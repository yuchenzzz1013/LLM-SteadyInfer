#include "scheduler/kv_manager.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>

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

  size_t kv_bytes = 0;
  if (paged_) {
    // Paged layout: [num_layers, num_blocks, block_size, kv_dim]. One block
    // pool serves all layers (block i of layer l is key_cache[l][i]).
    // Pool capacity matches the continuous layout within one block per seq:
    //   num_blocks = max_batch * ceil(max_seq_len / block_size)
    block_size_ = block_size;
    max_blocks_per_seq_ = (max_seq_len_ + block_size_ - 1) / block_size_;
    num_blocks_ = max_batch_ * max_blocks_per_seq_;
    kv_bytes = static_cast<size_t>(num_layers_) * num_blocks_ * block_size_ * kv_dim_ * 4;
    LOG(INFO) << "[KVMGR] Allocating PAGED KV caches:"
              << " num_layers=" << num_layers_
              << " num_blocks=" << num_blocks_
              << " block_size=" << block_size_
              << " max_blocks_per_seq=" << max_blocks_per_seq_
              << " kv_dim=" << kv_dim_
              << " each_size=" << (kv_bytes >> 20) << "MB"
              << " total=" << ((kv_bytes * 2) >> 20) << "MB"
              << " (continuous equivalent:"
              << " each=" << ((static_cast<size_t>(num_layers_) * max_batch_ * max_seq_len_ *
                               kv_dim_ * 4) >> 20) << "MB)";
    block_allocator_ = std::make_unique<BlockAllocator>(num_blocks_, max_batch_,
                                                        max_blocks_per_seq_, block_size_);
  } else {
    // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
    block_size_ = max_seq_len_;
    max_blocks_per_seq_ = 1;
    slot_busy_.assign(max_batch_, false);
    kv_bytes = static_cast<size_t>(num_layers_) * max_batch_ * max_seq_len_ * kv_dim_ * 4;
    LOG(INFO) << "[KVMGR] Allocating continuous KV caches:"
              << " num_layers=" << num_layers_
              << " max_batch=" << max_batch_
              << " max_seq_len=" << max_seq_len_
              << " kv_dim=" << kv_dim_
              << " each_size=" << (kv_bytes >> 20) << "MB"
              << " total=" << ((kv_bytes * 2) >> 20) << "MB";
#ifdef USE_PAGED_ATTENTION
    // CPU device (or paging disabled): degenerate slot-mode pool so the
    // scheduler's block-table code path stays uniform — one block == one
    // whole slot, block table width 1.
    block_allocator_ = std::make_unique<BlockAllocator>(max_batch_, max_batch_, 1, max_seq_len_);
#endif
  }

  // Log GPU memory before allocation
  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    LOG(INFO) << "[KVMGR] GPU before alloc: free=" << (free_mem >> 20)
              << "MB total=" << (total_mem >> 20) << "MB";
  }

  if (paged_) {
    // Element (layer, block, pos_in_block, d) at
    //   layer * (num_blocks * block_size * kv_dim)
    // + block * (block_size * kv_dim)
    // + pos_in_block * kv_dim + d
    // d-innermost within a page: the flash-decoding kernels walk consecutive
    // positions contiguously, crossing page boundaries through block_table.
    key_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                                 num_layers_, num_blocks_, block_size_, kv_dim_,
                                 true, alloc);
    value_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                                   num_layers_, num_blocks_, block_size_, kv_dim_,
                                   true, alloc);
  } else {
    // Head-dim-contiguous layout: cache[layer][slot][d][pos], position
    // innermost so decode MHA reads coalesce across consecutive positions.
    key_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                                 num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                 true, alloc);
    value_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                                   num_layers_, max_batch_, kv_dim_, max_seq_len_,
                                   true, alloc);
  }

  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    LOG(INFO) << "[KVMGR] GPU after alloc: free=" << (free_mem >> 20)
              << "MB total=" << (total_mem >> 20) << "MB";
  }

  LOG(INFO) << "[KVMGR] key_cache ptr=" << key_cache_.ptr<float>()
            << " value_cache ptr=" << value_cache_.ptr<float>();
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
  int count = 0;
  for (int r = 0; r < max_batch_; ++r) {
    if (block_allocator_->num_used_blocks(r) > 0) ++count;
  }
  return count;
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
