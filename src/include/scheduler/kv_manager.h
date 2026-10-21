#pragma once

#include <memory>
#include <vector>
#include "base/base.h"
#include "block_allocator.h"
#include "tensor/tensor.h"

namespace scheduler {

// How the KV cache is quantized. There is exactly one quantized shape:
// S4 — both K and V at 4 bits per value (two values per byte), asymmetric
// around a zero point. V is quantized immediately per (token, kv head); K is
// quantized per (token group, channel group) after a full group of
// `group_size` successive tokens has accumulated, while the most recent
// group's K is held in a BF16 residual pool. The residual window is what lets
// K keep per-channel-group statistics without re-reading the already-quantized
// prefix, and it is the reason the K pool alone is not a complete cache.
//
// Both halves are always quantized together, so this is a switch rather than a
// mask. The spelling is 6: the old mask's 1/5 selected the removed
// K-int8+V-int4 shape, and those values are now hard errors (see
// kv_quant_from_env) so a stale command line cannot silently measure a
// different precision.
enum KvQuant : int {
  kKvQuantNone = 0,  // bf16 pool
  kKvQuantS4 = 6,    // K int4 (grouped, with BF16 residual) + V int4
};

// LLAMA_KV_QUANT — "0"/"off" (default, unset) / "6"/"s4".
// Anything else is a LOG(FATAL): a typo silently leaving quantization off would
// look like "no speedup and no PPL change", which is the most expensive
// possible way to find the mistake. In particular the removed shape's
// spellings ("1", "5") and the never-existing mask arms ("2", "3", "4") are
// errors rather than aliases: a run that quietly means a different precision
// produces a number nobody can attribute.
int kv_quant_from_env();

class KVManager {
 public:
  // max_seq_len: per-sequence KV capacity (prompt + generation, in tokens).
  // device: decides the layout — CUDA (+USE_PAGED_ATTENTION) gets the paged
  //   layout [num_layers, num_blocks, block_size, kv_dim], everything else
  //   keeps the continuous layout [num_layers, max_batch, kv_dim, max_seq_len].
  // kv_head_num: number of KV heads (kv_dim = kv_head_num * head_size). Used
  //   by the S4 path for the V scale/zero pools (per token and kv head) and to
  //   validate the K channel group.
  // block_size: physical block size of the paged pool (default 16). Only used
  //   in paged mode, where the constructor CHECKs it is a power of two >= 4
  //   dividing 256 (kernel geometry).
  // kv_quant: whether to store the cache quantized (KvQuant: 0 or 6). Only the
  //   paged CUDA layout can be quantized — the continuous layout puts the token
  //   index in the last (fastest) dim, so a per-token scale would be a strided
  //   access — and asking for it anywhere else is a LOG(FATAL) rather than a
  //   silent fallback to bf16.
  // group_size / channel_group: S4 grouping (0 = resolve from
  //   LLAMA_KV_GROUP_SIZE / LLAMA_KV_CHANNEL_GROUP, then to the defaults).
  //   group_size must divide block_size and be a multiple of 4; channel_group
  //   must divide head_size and be a multiple of 4. Both are validated here.
  // The constructor LOG(FATAL)s if the pool tensors fail to allocate, so a
  // successful construction guarantees a usable KV pool.
  KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim, int kv_head_num,
            std::shared_ptr<base::DeviceAllocator> alloc,
            base::DeviceType device, int block_size = 16,
            int kv_quant = kKvQuantNone, int group_size = 0, int channel_group = 0);

  // Slot/row API. Identical semantics in both modes: a "slot" is one
  // block-table row (paged) == one slot (continuous). allocate() reserves a
  // sequence's whole KV capacity — the continuous layout's only option, and
  // what its legacy admission still uses; the paged scheduler path allocates
  // the row (allocate_blocks(0)) and grows its blocks lazily instead.
  int allocate();       // Returns slot/row id, -1 if capacity exhausted
  void deallocate(int slot_or_row);
  bool has_free_slot() const;
  // Slots/rows a sequence could still be admitted into (paged: rows no
  // sequence owns — see the impl for why counting free *blocks* was wrong).
  int free_slot_count() const;
  int busy_slot_count() const;
  int max_seq_len() const { return max_seq_len_; }

  // Paged-mode accessors. In continuous mode block_allocator() is the
  // degenerate slot-mode pool (block_size == max_seq_len, table width 1) when
  // USE_PAGED_ATTENTION is on, and nullptr when it is off.
  BlockAllocator* block_allocator() { return block_allocator_.get(); }
  const BlockAllocator* block_allocator() const { return block_allocator_.get(); }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int block_size() const { return block_size_; }
  int num_blocks() const { return num_blocks_; }
  bool is_paged() const { return paged_; }

  // Bytes of the preallocated K + V pool (both modes), INCLUDING the S4
  // scale / zero / K-residual pools when quantized. This is the memory the
  // cache really occupies from construction on — a driver that prints a
  // token-count estimate should print this instead of recomputing it.
  long long pool_bytes() const;

  // S4 dequantization pools. Empty tensors unless quantized().
  //
  //   key_scale_cache_ / key_zero_cache_
  //       [num_layers, num_blocks, groups_per_block, channel_groups] fp32,
  //       per (token group, channel group). groups_per_block is
  //       block_size / group_size, so every scale belongs to exactly one
  //       physical page and prefix-cache sharing carries it automatically.
  //   value_scale_cache_ / value_zero_cache_
  //       [num_layers, num_blocks, kv_head_num, block_size] fp32,
  //       per (token, kv head) — V has no residual window, so every cached
  //       token owns a scale/zero from the moment it is written.
  //   key_residual_cache_
  //       [num_layers, max_batch, group_size, kv_dim] bf16, indexed by the
  //       sequence's block-table row (slot), `pos % group_size`, and dim.
  //       Only the current, not-yet-complete group is meaningful; a completed
  //       group's int4 copy in key_cache_ is authoritative for reads.
  //
  // They are SEPARATE pools rather than packed views so that each reader keeps
  // the addressing it wants: K's scale is consumed per 4 lane dims, V's per
  // token, and the residual pool is bf16.
  tensor::Tensor& key_scale_cache() { return key_scale_cache_; }
  tensor::Tensor& value_scale_cache() { return value_scale_cache_; }
  tensor::Tensor& key_zero_cache() { return key_zero_cache_; }
  tensor::Tensor& value_zero_cache() { return value_zero_cache_; }
  tensor::Tensor& key_residual_cache() { return key_residual_cache_; }
  const tensor::Tensor& key_scale_cache() const { return key_scale_cache_; }
  const tensor::Tensor& value_scale_cache() const { return value_scale_cache_; }
  const tensor::Tensor& key_zero_cache() const { return key_zero_cache_; }
  const tensor::Tensor& value_zero_cache() const { return value_zero_cache_; }
  const tensor::Tensor& key_residual_cache() const { return key_residual_cache_; }

  // S4 grouping, resolved and validated by the constructor. 0/0 when the
  // cache is not quantized.
  int group_size() const { return group_size_; }
  int channel_group() const { return channel_group_; }
  int channel_groups() const { return channel_groups_; }
  int groups_per_block() const { return groups_per_block_; }

  int kv_quant() const { return kv_quant_; }
  // One test: with a single quantized shape, "is K int4", "is V int4" and
  // "does this cache carry a residual window" are the same question (see the
  // KvQuant note). The attention kernels select their compile-time QUANT arm
  // on the pools' dtype, which is the one thing that follows a tensor into
  // every call.
  bool quantized() const { return kv_quant_ != kKvQuantNone; }

  tensor::Tensor& key_cache() { return key_cache_; }
  tensor::Tensor& value_cache() { return value_cache_; }
  const tensor::Tensor& key_cache() const { return key_cache_; }
  const tensor::Tensor& value_cache() const { return value_cache_; }

 private:
  int num_layers_ = 0;
  int max_batch_ = 0;
  int max_seq_len_ = 0;      // per-sequence KV capacity in tokens
  int kv_dim_ = 0;
  int kv_head_num_ = 0;
  int num_blocks_ = 0;       // paged: physical blocks per layer
  int block_size_ = 16;      // paged: block size; continuous: max_seq_len
  int max_blocks_per_seq_ = 1;  // paged: table width; continuous: 1
  bool paged_ = false;
  int kv_quant_ = kKvQuantNone;
  int group_size_ = 0;      // tokens per K quantization group (divides block_size)
  int channel_group_ = 0;   // K dims per scale/zero (divides head_size)
  int channel_groups_ = 0;  // kv_dim / channel_group
  int groups_per_block_ = 0;  // block_size / group_size

  // Paged layout: [num_layers, num_blocks, block_size, kv_dim].
  // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
  // Quantized (paged only) both pools are int8 packed rows: dim d lives in
  // byte d/2, the low nibble for even d. K and V therefore have the same
  // kv_dim/2-byte row and the same reader geometry; only their scale/zero
  // granularity differs.
  tensor::Tensor key_cache_;
  tensor::Tensor value_cache_;

  // S4 K scales (see the accessor note for the layouts). Indexed by the SAME
  // physical block id the block table yields, so a block shared through the
  // prefix cache carries its scales with it and no extra refcounting is
  // needed. For K the inner dims are (token group in block, channel group):
  // a decode position group of 4 always lies in one token group (group_size
  // is a multiple of 4), and each lane owns one channel group, so a lane
  // resolves its scale with two integer divides which the compiler folds once
  // group_size / channel_group are runtime kernel arguments.
  //
  // V scale/zero keep pos as the innermost dim, mirroring the old layout: the
  // decode kernel walks 4 positions of one kv head at a time, making the four
  // floats one 16-byte-aligned float4 load.
  tensor::Tensor key_scale_cache_;
  tensor::Tensor key_zero_cache_;
  tensor::Tensor value_scale_cache_;
  tensor::Tensor value_zero_cache_;

  // BF16 K residual window: [num_layers, max_batch, group_size, kv_dim],
  // indexed by the sequence's block-table row (slot), pos % group_size, dim.
  tensor::Tensor key_residual_cache_;

  std::vector<bool> slot_busy_;               // continuous layout only
  std::unique_ptr<BlockAllocator> block_allocator_;
};

}  // namespace scheduler
