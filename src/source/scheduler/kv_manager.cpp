#include "scheduler/kv_manager.h"

namespace scheduler {

KVManager::KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
                     std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DeviceType device, int block_size)
    : num_layers_(num_layers),
      max_batch_(max_batch),
      max_seq_len_(max_seq_len),
      kv_dim_(kv_dim) {
#ifdef USE_PAGED_ATTENTION
  paged_ = (device == base::DeviceType::kDeviceCUDA);
#else
  paged_ = false;
#endif

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

  // All CUDA kernels (scatter / attention / decode) address the cache as
  // raw bf16, so allocate BF16 on CUDA and keep FP32 only on the CPU path
  // (whose FP32 fallback kernels read float pointers). This halves KV
  // memory on device.
  const base::DataType kv_dtype =
      (device == base::DeviceType::kDeviceCUDA) ? base::DataType::kDataTypeBF16
                                                : base::DataType::kDataTypeFp32;

  if (paged_) {
    // Element (layer, block, pos_in_block, d) at
    //   layer * (num_blocks * block_size * kv_dim)
    // + block * (block_size * kv_dim)
    // + pos_in_block * kv_dim + d
    // d-innermost within a page: the flash-decoding kernels walk consecutive
    // positions contiguously, crossing page boundaries through block_table.
    key_cache_ = tensor::Tensor(kv_dtype, num_layers_, num_blocks_, block_size_, kv_dim_,
                                true, alloc);
    value_cache_ = tensor::Tensor(kv_dtype, num_layers_, num_blocks_, block_size_, kv_dim_,
                                  true, alloc);
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
  if (key_cache_.is_empty() || value_cache_.is_empty()) {
    const double pool_gb =
        static_cast<double>(pool_bytes()) / (1024.0 * 1024.0 * 1024.0);
    LOG(FATAL) << "KV pool allocation failed: tried to allocate " << pool_gb << " GB ("
               << (paged_ ? "paged" : "continuous") << " layout, layers=" << num_layers_
               << " max_batch=" << max_batch_ << " max_seq_len=" << max_seq_len_
               << " kv_dim=" << kv_dim_
               << (paged_ ? " block_size=" + std::to_string(block_size_) : std::string())
               << "). Lower max_batch / max_seq_len (or the driver's --gpu-mem-fraction) "
                  "and retry.";
  }
}

long long KVManager::pool_bytes() const {
  // Derived from the dims, so a failed (empty) allocation still reports the
  // size that was attempted — see the constructor's failure message.
  return static_cast<long long>(key_cache_.byte_size()) +
         static_cast<long long>(value_cache_.byte_size());
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
  return block_allocator_->free_block_count() / max_blocks_per_seq_;
#else
  int count = 0;
  for (bool busy : slot_busy_) {
    if (!busy) ++count;
  }
  return count;
#endif
}

}  // namespace scheduler
