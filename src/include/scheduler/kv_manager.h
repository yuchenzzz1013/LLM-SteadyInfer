#pragma once

#include <memory>
#include <vector>
#include "base/base.h"
#include "block_allocator.h"
#include "tensor/tensor.h"

namespace scheduler {

// How the KV cache is quantized. There is exactly one quantized shape left:
// K at 8 bits per value (one value per byte) and V at 4 bits per value (two
// values per byte, symmetric around zero, per-(token, kv head) fp32 scale),
// with the V pool row kv_dim/2 bytes long. Both halves are always quantized
// together, so this is a switch rather than a mask — and it is spelled 1, the
// way a switch is.
//
// The number is where the old bitmask left off rather than a fresh coinage:
// 1 was "K only" and 5 was "K int8 + V int4" while the mask existed, and 5 is
// what every log and reproduce command in results/ says. Both digits still
// select this arm (see kv_quant_from_env), so an old command line keeps
// working — but a script that exported 1 expecting K-only now gets K and V
// quantized, which is the one spelling whose meaning changed.
//
// K has no 4-bit mode: a K error moves the softmax itself, which no downstream
// scaling can undo, so K stays at 8 bits while V — whose error only scales the
// weighted sum after the softmax — takes the aggressive mode.
enum KvQuant : int {
  kKvQuantNone = 0,  // bf16 pool
  kKvQuantK8V4 = 1,  // K int8 + V int4
};

// LLAMA_KV_QUANT — "0"/"off" (default, unset) / "1"/"kv4"/"k+v4" (and the old
// spelling "5", which selected exactly this arm before the mask collapsed).
// Anything else is a LOG(FATAL): a typo silently leaving quantization off would
// look like "no speedup and no PPL change", which is the most expensive
// possible way to find the mistake. The removed arms' other spellings ("2",
// "3", "4", "k", "v", "v4", "kv", …) are an error rather than an alias, for the
// same reason — a run that quietly means a different precision produces a
// number nobody can attribute.
int kv_quant_from_env();

class KVManager {
 public:
  // max_seq_len: per-sequence KV capacity (prompt + generation, in tokens).
  // device: decides the layout — CUDA (+USE_PAGED_ATTENTION) gets the paged
  //   layout [num_layers, num_blocks, block_size, kv_dim], everything else
  //   keeps the continuous layout [num_layers, max_batch, kv_dim, max_seq_len].
  // kv_head_num: number of KV heads (kv_dim = kv_head_num * head_size). Only
  //   needed by the int8 path, whose per-(token, kv_head) scales are stored
  //   one per head; the bf16 pool is head-agnostic.
  // block_size: physical block size of the paged pool (default 16). Only used
  //   in paged mode, where the constructor CHECKs it is a power of two >= 4
  //   dividing 256 (kernel geometry).
  // kv_quant: whether to store the cache quantized (KvQuant: 0 or 5). Only the
  //   paged CUDA layout can be quantized — the continuous layout puts the token
  //   index in the last (fastest) dim, so a per-token scale would be a strided
  //   access — and asking for it anywhere else is a LOG(FATAL) rather than a
  //   silent fallback to bf16.
  // The constructor LOG(FATAL)s if the pool tensors fail to allocate, so a
  // successful construction guarantees a usable KV pool.
  KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim, int kv_head_num,
            std::shared_ptr<base::DeviceAllocator> alloc,
            base::DeviceType device, int block_size = 16,
            int kv_quant = kKvQuantNone);

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

  // Bytes of the preallocated K + V pool (both modes), INCLUDING the int8
  // scale pools when quantized. This is the memory the cache really occupies
  // from construction on — a driver that prints a token-count estimate should
  // print this instead of recomputing it.
  long long pool_bytes() const;

  // int8 dequantization scales, one fp32 per (block, kv_head, pos_in_block) —
  // see the layout note on the members. Empty tensors unless quantized().
  //
  // These are SEPARATE pools rather than one combined tensor so that the two
  // halves keep their own allocation and addressing: K's scale is consumed by
  // the score, V's by the weighted sum, and folding them into one tensor would
  // only make the two readers agree on a layout neither of them wants.
  tensor::Tensor& key_scale_cache() { return key_scale_cache_; }
  tensor::Tensor& value_scale_cache() { return value_scale_cache_; }
  const tensor::Tensor& key_scale_cache() const { return key_scale_cache_; }
  const tensor::Tensor& value_scale_cache() const { return value_scale_cache_; }

  int kv_quant() const { return kv_quant_; }
  // One test, not three: with a single quantized shape, "is K int8", "is V
  // quantized" and "is V 4-bit" are the same question (see the KvQuant note).
  // The attention kernels still detect the widths from the pools' own last dims
  // (see kv_pool_is_v4 in the CUDA kernels) rather than from a flag threaded
  // through every call — the shape is the one thing that follows the tensor
  // everywhere — so this predicate is only about the pool the manager built.
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

  // Paged layout: [num_layers, num_blocks, block_size, kv_dim].
  // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
  // Quantized (paged only) the same paged shape in int8 — except V's, whose
  // last dim is kv_dim/2 (two 4-bit values per byte). The last dim is what the
  // kernels read the width back from; it must stay the ONLY place the V pool's
  // width is expressed.
  tensor::Tensor key_cache_;
  tensor::Tensor value_cache_;

  // int8 scale pools: [num_layers, num_blocks, kv_head_num, block_size] fp32,
  // indexed by the SAME physical block id the block table yields, so a block
  // shared through the prefix cache carries its scales with it and no extra
  // refcounting is needed.
  //
  // Pos is the innermost dim (not kv_head, which would mirror the KV layout
  // more literally) because the decode kernel walks positions for ONE kv head
  // at a time in groups of 4: this layout makes a group's 4 scales one
  // contiguous, 16-byte-aligned float4 — a single load — while the mirrored
  // layout would make them a 4-way strided gather.
  tensor::Tensor key_scale_cache_;
  tensor::Tensor value_scale_cache_;

  std::vector<bool> slot_busy_;               // continuous layout only
  std::unique_ptr<BlockAllocator> block_allocator_;
};

}  // namespace scheduler
