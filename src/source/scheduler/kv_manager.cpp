#include "scheduler/kv_manager.h"

#include <cctype>
#include <cstdlib>
#include <string>

namespace scheduler {

int kv_quant_from_env() {
  const char* raw = std::getenv("LLAMA_KV_QUANT");
  if (raw == nullptr || raw[0] == '\0') return kKvQuantNone;
  // Case-insensitive: the value is typed by hand on a command line.
  std::string v(raw);
  for (char& c : v) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
  // "1" is the switch: the number is the arm's enum value, identical to the
  // Scheduler constructor argument and the benchmarks' --kv-quant, so the two
  // entry points cannot be read against each other. Every other value is a
  // LOG(FATAL) rather than an alias — in particular 5 named the removed
  // K-int8 + V-int4 shape, and 2/3/4 named mask arms that no longer exist.
  // Quietly resolving any of them to a different precision would produce a
  // number nobody can attribute.
  if (v == "0" || v == "off" || v == "none" || v == "false") return kKvQuantNone;
  if (v == "1" || v == "s4") return kKvQuantS4;
  LOG(FATAL) << "LLAMA_KV_QUANT: unrecognized value '" << raw
             << "' (expected 0/off or 1/s4 — S4: K int4 grouped with a BF16 residual "
                "window + V int4 asymmetric). The old K-int8 + V-int4 spelling 5, "
                "and the removed mask arms 2/3/4, are hard errors: rewrite reproduce "
                "commands to use 1.";
  return kKvQuantNone;
}

// Read a non-negative integer tunable from the environment. Unset, empty and
// "0" all mean "use the built-in default" (the caller passes 0). Anything that
// does not parse as a non-negative integer is a hard error: a typo must not
// silently fall back to a different grouping.
static int grouping_from_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return 0;
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  CHECK(end != nullptr && *end == '\0' && value >= 0)
      << name << ": expected a non-negative integer, got '" << raw << "'";
  CHECK(value <= 4096) << name << "=" << value << " is unreasonably large";
  return static_cast<int>(value);
}

KVManager::KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
                     int kv_head_num, std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DeviceType device, int block_size, int kv_quant, int group_size,
                     int channel_group)
    : num_layers_(num_layers),
      max_batch_(max_batch),
      max_seq_len_(max_seq_len),
      kv_dim_(kv_dim),
      kv_head_num_(kv_head_num) {
#ifdef USE_PAGED_ATTENTION
  paged_ = (device == base::DeviceType::kDeviceCUDA);
#else
  paged_ = false;
#endif

  // Quantization requires the paged layout (see the header). Reject rather
  // than downgrade: a caller that asked for quantization and silently got bf16
  // would measure no speedup, no PPL change, and no error.
  CHECK(kv_quant == kKvQuantNone || paged_)
      << "KV quantization (LLAMA_KV_QUANT) is only implemented for the paged CUDA layout "
         "(USE_PAGED_ATTENTION + device=kDeviceCUDA); requested mask="
      << kv_quant << ", paged=" << paged_;
  // Value validation: exactly two values, and the removed ones are an error
  // rather than an alias for S4. A driver or a script that still passes
  // 1/2/3/4/5 is asking for a precision this build does not have; resolving it
  // to S4 would produce a PPL number attributed to an arm that never ran.
  CHECK(kv_quant == kKvQuantNone || kv_quant == kKvQuantS4)
      << "Invalid KV quantization value " << kv_quant
      << " (expected 0 for off or 6 for S4: K int4 grouped with a BF16 residual window + "
         "V int4 asymmetric). The old K-int8 + V-int4 values 1 and 5 and the removed mask "
         "arms 2/3/4 are hard errors.";
  kv_quant_ = kv_quant;

  // LLAMA_KV_WINDOW belonged to the removed V residual-window experiment. S4's
  // K residual window is sized by group_size, not by this variable, so a run
  // that still exports it would silently measure a different configuration
  // while its logs claim otherwise. Hard error rather than ignored.
  const char* window_env = std::getenv("LLAMA_KV_WINDOW");
  CHECK(window_env == nullptr)
      << "LLAMA_KV_WINDOW=" << window_env
      << " is no longer supported: the removed V residual window has no replacement, and "
         "S4's K residual window is controlled by LLAMA_KV_GROUP_SIZE. Unset it — any value "
         "here, including 0, is a leftover.";

  if (paged_) {
    // Paged layout: [num_layers, num_blocks, block_size, kv_dim]. One block
    // pool serves all layers (block i of layer l is key_cache[l][i]).
    // Pool capacity matches the continuous layout within one block per seq:
    //   num_blocks = max_batch * ceil(max_seq_len / block_size)
    //
    // Validate the block size here, where the value and its origin (Scheduler
    // argument or LLAMA_BLOCK_SIZE) are still attributable: the kernels assume
    // a power-of-two page that fits a warp's 4-position group
    // (paged_decode_geometry_ok requires block_size >= 4) and the pool caps a
    // page at 256 tokens. Anything else would be rejected by the kernels'
    // geometry CHECK or mis-address the pool.
    CHECK(block_size >= 4 && (block_size & (block_size - 1)) == 0 && 256 % block_size == 0)
        << "Invalid paged KV block_size=" << block_size
        << " (must be a power of two >= 4 dividing 256: 8, 16, 32, ...); "
        << "set by the Scheduler block_size argument or LLAMA_BLOCK_SIZE";
    block_size_ = block_size;
    max_blocks_per_seq_ = (max_seq_len_ + block_size_ - 1) / block_size_;
    num_blocks_ = max_batch_ * max_blocks_per_seq_;
    block_allocator_ = std::make_unique<BlockAllocator>(num_blocks_, max_batch_,
                                                        max_blocks_per_seq_, block_size_);
  } else {
    // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
    block_size_ = max_seq_len_;
    max_blocks_per_seq_ = 1;
    slot_busy_.assign(max_batch_, false);
#ifdef USE_PAGED_ATTENTION
    // CPU device (or paging disabled): degenerate slot-mode pool so the
    // scheduler's block-table code path stays uniform — one block == one
    // whole slot, block table width 1.
    block_allocator_ = std::make_unique<BlockAllocator>(max_batch_, max_batch_, 1, max_seq_len_);
#endif
  }

  // Resolve the S4 grouping. The defaults are chosen so that a default
  // configuration is always valid: group_size == block_size makes every
  // physical page exactly one quantization group, and channel_group is 32 when
  // the head is a multiple of it (the served head sizes are 64 and 128),
  // otherwise the whole head is one group.
  if (quantized()) {
    const int resolved_group =
        group_size > 0 ? group_size : grouping_from_env("LLAMA_KV_GROUP_SIZE");
    const int resolved_channel =
        channel_group > 0 ? channel_group : grouping_from_env("LLAMA_KV_CHANNEL_GROUP");
    CHECK(kv_head_num_ > 0 && kv_dim_ % kv_head_num_ == 0)
        << "S4 K quantization needs kv_dim divisible by kv_head_num, got kv_dim=" << kv_dim_
        << " kv_head_num=" << kv_head_num_;
    const int head_size = kv_dim_ / kv_head_num_;
    group_size_ = resolved_group > 0 ? resolved_group : block_size_;
    // Default to 8 channels per scale pair: the PPL sweep on Qwen3-4B
    // (32x4096-equivalent sample) put it closest to BF16, while 32 was
    // measurably coarser. head_size is a multiple of 8 on every paged
    // geometry, so 8 always divides.
    channel_group_ = resolved_channel > 0
                         ? resolved_channel
                         : (head_size % 8 == 0 ? 8 : head_size);
    CHECK(group_size_ % 4 == 0 && block_size_ % group_size_ == 0)
        << "S4 group_size=" << group_size_ << " must be a multiple of 4 dividing block_size="
        << block_size_ << " (LLAMA_KV_GROUP_SIZE)";
    CHECK(channel_group_ % 4 == 0 && head_size % channel_group_ == 0)
        << "S4 channel_group=" << channel_group_
        << " must be a multiple of 4 dividing head_size=" << head_size
        << " (LLAMA_KV_CHANNEL_GROUP)";
    CHECK(kv_dim_ % 2 == 0) << "4-bit K/V need an even kv_dim, got " << kv_dim_;
    groups_per_block_ = block_size_ / group_size_;
    channel_groups_ = kv_dim_ / channel_group_;
  }

  // All CUDA kernels (scatter / attention / decode) address the cache as
  // raw bf16, so allocate BF16 on CUDA and keep FP32 only on the CPU path
  // (whose FP32 fallback kernels read float pointers). This halves KV
  // memory on device.
  const base::DataType kv_dtype =
      (device == base::DeviceType::kDeviceCUDA) ? base::DataType::kDataTypeBF16
                                                : base::DataType::kDataTypeFp32;

  if (paged_) {
    // Element (layer, block, pos_in_block, d) at
    //   layer * (num_blocks * block_size * row_bytes)
    // + block * (block_size * row_bytes)
    // + pos_in_block * row_bytes + d
    // d-innermost within a page: the flash-decoding kernels walk consecutive
    // positions contiguously, crossing page boundaries through block_table.
    // Quantized, both pools are int8 with row_bytes = kv_dim/2: dim d lives in
    // byte d/2, the low nibble for even d. The last dim is the width the
    // kernels read back, so it is set here and nowhere else.
    const base::DataType cached_dtype =
        quantized() ? base::DataType::kDataTypeInt8 : kv_dtype;
    const int row_elements = quantized() ? kv_dim_ / 2 : kv_dim_;
    key_cache_ = tensor::Tensor(cached_dtype, num_layers_, num_blocks_, block_size_, row_elements,
                                true, alloc);
    value_cache_ = tensor::Tensor(cached_dtype, num_layers_, num_blocks_, block_size_,
                                  row_elements, true, alloc);

    // S4 metadata pools (see the layout note on the members). Only allocated
    // when quantized.
    if (quantized()) {
      key_scale_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_, num_blocks_,
                                        groups_per_block_, channel_groups_, true, alloc);
      key_zero_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_, num_blocks_,
                                       groups_per_block_, channel_groups_, true, alloc);
      value_scale_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_,
                                          num_blocks_, kv_head_num_, block_size_, true, alloc);
      value_zero_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32, num_layers_,
                                         num_blocks_, kv_head_num_, block_size_, true, alloc);
      key_residual_cache_ = tensor::Tensor(base::DataType::kDataTypeBF16, num_layers_, max_batch_,
                                           group_size_, kv_dim_, true, alloc);
    }
  } else {
    // Head-dim-contiguous layout: cache[layer][slot][d][pos], position
    // innermost so decode MHA reads coalesce across consecutive positions.
    key_cache_ = tensor::Tensor(kv_dtype, num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                true, alloc);
    value_cache_ = tensor::Tensor(kv_dtype, num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                  true, alloc);
  }

  // The KV pool is by far the largest allocation in the process and it is
  // requested here, once, up front. If the device is out of memory the
  // allocator returns an empty tensor; without this check that surfaces much
  // later as a CUDA illegal address (or a null-pointer memcpy on CPU), with
  // nothing pointing at the pool size or the knobs that control it. Fail at
  // construction instead, with the numbers needed to size it down.
  const bool metadata_ok =
      !quantized() ||
      (!key_scale_cache_.is_empty() && !key_zero_cache_.is_empty() &&
       !value_scale_cache_.is_empty() && !value_zero_cache_.is_empty() &&
       !key_residual_cache_.is_empty());
  if (key_cache_.is_empty() || value_cache_.is_empty() || !metadata_ok) {
    const double pool_gb =
        static_cast<double>(pool_bytes()) / (1024.0 * 1024.0 * 1024.0);
    LOG(FATAL) << "KV pool allocation failed: tried to allocate " << pool_gb << " GB ("
               << (paged_ ? "paged" : "continuous") << " layout, layers=" << num_layers_
               << " max_batch=" << max_batch_ << " max_seq_len=" << max_seq_len_
               << " kv_dim=" << kv_dim_
               << (paged_ ? " block_size=" + std::to_string(block_size_) : std::string())
               << (quantized() ? " kv_quant=" + std::to_string(kv_quant_) +
                                         " (S4: K/V int4 packed to kv_dim/2 bytes, K "
                                         "group_size=" + std::to_string(group_size_) +
                                         " channel_group=" + std::to_string(channel_group_) +
                                         ", plus the scale/zero and K-residual pools)"
                               : std::string())
               << "). Lower max_batch / max_seq_len (or the driver's --gpu-mem-fraction) "
                  "and retry.";
  }
}

long long KVManager::pool_bytes() const {
  // Derived from the dims, so a failed (empty) allocation still reports the
  // size that was attempted — see the constructor's failure message. The S4
  // metadata pools are included: they are real cache memory (scale/zero plus
  // the BF16 residual window), and a driver sizing the pool from the packed
  // K/V rows alone would understate it by exactly that much.
  return static_cast<long long>(key_cache_.byte_size()) +
         static_cast<long long>(value_cache_.byte_size()) +
         static_cast<long long>(key_scale_cache_.byte_size()) +
         static_cast<long long>(key_zero_cache_.byte_size()) +
         static_cast<long long>(value_scale_cache_.byte_size()) +
         static_cast<long long>(value_zero_cache_.byte_size()) +
         static_cast<long long>(key_residual_cache_.byte_size());
}

int KVManager::allocate() {
#ifdef USE_PAGED_ATTENTION
  return block_allocator_->allocate_blocks(max_blocks_per_seq_);
#else
  for (int i = 0; i < max_batch_; ++i) {
    if (!slot_busy_[i]) {
      slot_busy_[i] = true;
      return i;
    }
  }
  return -1;
#endif
}

void KVManager::deallocate(int slot_or_row) {
#ifdef USE_PAGED_ATTENTION
  block_allocator_->free_all(slot_or_row);
#else
  if (slot_or_row >= 0 && slot_or_row < max_batch_) {
    slot_busy_[slot_or_row] = false;
  }
#endif
}

bool KVManager::has_free_slot() const {
  return free_slot_count() > 0;
}

int KVManager::busy_slot_count() const {
#ifdef USE_PAGED_ATTENTION
  // Rows owned by a sequence — NOT "rows with at least one block": a fully
  // prefix-shared prompt owns an empty (or shared-only) table but is still in
  // flight, and counting it would understate the KV cache in use.
  return block_allocator_->num_busy_rows();
#else
  int count = 0;
  for (bool busy : slot_busy_) {
    if (busy) ++count;
  }
  return count;
#endif
}

int KVManager::free_slot_count() const {
#ifdef USE_PAGED_ATTENTION
  // Rows, not blocks. One sequence owns one block-table row for its whole
  // lifetime, so the pool accepts exactly (num_rows - busy_rows) more
  // sequences — that is what "free slots" means in the continuous layout this
  // mirrors, and what an admission check needs. The old free_blocks /
  // max_blocks_per_seq counted full-window sequences' worth of blocks instead:
  // it read 0 with rows still free (blocks held by running rows or pinned by
  // the prefix cache), and > 0 with every row taken by sequences that had used
  // little of their capacity. Block pressure is a different question, and one
  // the allocator answers per allocation (append_blocks reclaims before it
  // fails), not from a precomputed count.
  return block_allocator_->num_rows() - block_allocator_->num_busy_rows();
#else
  int count = 0;
  for (bool busy : slot_busy_) {
    if (!busy) ++count;
  }
  return count;
#endif
}

}  // namespace scheduler
