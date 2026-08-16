#pragma once

#include <memory>
#include <vector>
#include "base/base.h"
#include "block_allocator.h"
#include "tensor/tensor.h"

namespace scheduler {

class KVManager {
 public:
  // max_seq_len: per-sequence KV capacity (prompt + generation, in tokens).
  // device: decides the layout — CUDA (+USE_PAGED_ATTENTION) gets the paged
  //   layout [num_layers, num_blocks, block_size, kv_dim], everything else
  //   keeps the continuous layout [num_layers, max_batch, kv_dim, max_seq_len].
  // block_size: physical block size of the paged pool (default 16).
  KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
            std::shared_ptr<base::DeviceAllocator> alloc,
            base::DeviceType device, int block_size = 16);

  // Slot/row API. Identical semantics in both modes: allocate() reserves a
  // sequence's whole KV capacity — one block-table row (paged) == one slot
  // (continuous), so M1 keeps the legacy full-capacity admission exactly.
  int allocate();       // Returns slot/row id, -1 if capacity exhausted
  void deallocate(int slot_or_row);
  bool has_free_slot() const;
  int free_slot_count() const;
  int busy_slot_count() const;
  int max_batch() const { return max_batch_; }
  int max_seq_len() const { return max_seq_len_; }

  // Paged-mode accessors. In continuous mode block_allocator() is the
  // degenerate slot-mode pool (block_size == max_seq_len, table width 1) when
  // USE_PAGED_ATTENTION is on, and nullptr when it is off.
  BlockAllocator* block_allocator() { return block_allocator_.get(); }
  const BlockAllocator* block_allocator() const { return block_allocator_.get(); }
  int max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int block_size() const { return block_size_; }
  bool is_paged() const { return paged_; }

  tensor::Tensor& key_cache() { return key_cache_; }
  tensor::Tensor& value_cache() { return value_cache_; }
  const tensor::Tensor& key_cache() const { return key_cache_; }
  const tensor::Tensor& value_cache() const { return value_cache_; }

 private:
  int num_layers_ = 0;
  int max_batch_ = 0;
  int max_seq_len_ = 0;      // per-sequence KV capacity in tokens
  int kv_dim_ = 0;
  int num_blocks_ = 0;       // paged: physical blocks per layer
  int block_size_ = 16;      // paged: block size; continuous: max_seq_len
  int max_blocks_per_seq_ = 1;  // paged: table width; continuous: 1
  bool paged_ = false;

  // Paged layout: [num_layers, num_blocks, block_size, kv_dim].
  // Continuous layout: [num_layers, max_batch, kv_dim, max_seq_len].
  tensor::Tensor key_cache_;
  tensor::Tensor value_cache_;
  std::vector<bool> slot_busy_;               // continuous layout only
  std::unique_ptr<BlockAllocator> block_allocator_;
};

}  // namespace scheduler
