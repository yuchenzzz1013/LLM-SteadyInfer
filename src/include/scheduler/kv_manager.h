#pragma once

#include <memory>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace scheduler {

class KVManager {
 public:
  KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
            std::shared_ptr<base::DeviceAllocator> alloc);

  int allocate();       // Returns slot id, -1 if no free slot
  void deallocate(int slot);
  bool has_free_slot() const;
  int free_slot_count() const;
  int max_batch() const { return max_batch_; }
  int max_seq_len() const { return max_seq_len_; }
  int busy_slot_count() const;

  // Copy KV data from source (model internal cache) to a slot for given positions
  void copy_to_slot(int slot, const tensor::Tensor& src_key, const tensor::Tensor& src_val,
                    int layer_idx, int start_pos, int num_positions);

  // Get pointer to element (layer, slot, d, pos) of the head-dim-contiguous
  // KV cache [num_layers, max_batch, kv_dim, max_seq_len].
  float* key_slot_ptr(int layer_idx, int slot, int pos, int d = 0);
  float* value_slot_ptr(int layer_idx, int slot, int pos, int d = 0);

  tensor::Tensor& key_cache() { return key_cache_; }
  tensor::Tensor& value_cache() { return value_cache_; }
  const tensor::Tensor& key_cache() const { return key_cache_; }
  const tensor::Tensor& value_cache() const { return value_cache_; }

 private:
  int num_layers_ = 0;
  int max_batch_ = 0;
  int max_seq_len_ = 0;
  int kv_dim_ = 0;
  tensor::Tensor key_cache_;    // [num_layers, max_batch, kv_dim, max_seq_len]
  tensor::Tensor value_cache_;
  std::vector<bool> slot_busy_;
};

}  // namespace scheduler
