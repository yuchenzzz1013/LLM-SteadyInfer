#include "scheduler/kv_manager.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <glog/logging.h>

namespace scheduler {

KVManager::KVManager(int num_layers, int max_batch, int max_seq_len, int kv_dim,
                     std::shared_ptr<base::DeviceAllocator> alloc)
    : num_layers_(num_layers),
      max_batch_(max_batch),
      max_seq_len_(max_seq_len),
      kv_dim_(kv_dim),
      slot_busy_(max_batch, false) {
  size_t kv_bytes = static_cast<size_t>(num_layers_) * max_batch_ * max_seq_len_ * kv_dim_ * 4;
  LOG(INFO) << "[KVMGR] Allocating KV caches:"
            << " num_layers=" << num_layers_
            << " max_batch=" << max_batch_
            << " max_seq_len=" << max_seq_len_
            << " kv_dim=" << kv_dim_
            << " each_size=" << (kv_bytes >> 20) << "MB"
            << " total=" << ((kv_bytes * 2) >> 20) << "MB";

  // Log GPU memory before allocation
  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    LOG(INFO) << "[KVMGR] GPU before alloc: free=" << (free_mem >> 20)
              << "MB total=" << (total_mem >> 20) << "MB";
  }

  // Allocate KV cache: [num_layers, max_batch, max_seq_len, kv_dim]
  key_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                               num_layers_, max_batch_, max_seq_len_, kv_dim_,
                               true, alloc);
  value_cache_ = tensor::Tensor(base::DataType::kDataTypeFp32,
                                 num_layers_, max_batch_, max_seq_len_, kv_dim_,
                                 true, alloc);

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
  for (int i = 0; i < max_batch_; ++i) {
    if (!slot_busy_[i]) {
      slot_busy_[i] = true;
      return i;
    }
  }
  return -1;
}

void KVManager::deallocate(int slot) {
  if (slot >= 0 && slot < max_batch_) {
    slot_busy_[slot] = false;
  }
}

void KVManager::copy_to_slot(int slot, const tensor::Tensor& src_key,
                              const tensor::Tensor& src_val,
                              int layer_idx, int start_pos, int num_positions) {
  // Bounds check: prevent buffer overflow when prompt is longer than max_seq_len_
  int end_pos = start_pos + num_positions;
  if (end_pos > max_seq_len_) {
    LOG(WARNING) << "[KVMGR] copy_to_slot truncating: start=" << start_pos
                 << " num=" << num_positions << " max_seq_len=" << max_seq_len_
                 << " slot=" << slot;
    num_positions = max_seq_len_ - start_pos;
    if (num_positions <= 0) return;
  }

  bool is_cuda = (src_key.device_type() == base::DeviceType::kDeviceCUDA);
  int src_seq_len = src_key.get_dim(1);
  size_t copy_bytes = kv_dim_ * sizeof(float);

  for (int p = 0; p < num_positions; ++p) {
    float* dst_k = key_slot_ptr(layer_idx, slot, start_pos + p);
    float* dst_v = value_slot_ptr(layer_idx, slot, start_pos + p);
    const float* src_k = src_key.ptr<float>(
        layer_idx * src_seq_len * kv_dim_ + (start_pos + p) * kv_dim_);
    const float* src_v = src_val.ptr<float>(
        layer_idx * src_seq_len * kv_dim_ + (start_pos + p) * kv_dim_);

    if (is_cuda) {
      cudaMemcpy(dst_k, src_k, copy_bytes, cudaMemcpyDeviceToDevice);
      cudaMemcpy(dst_v, src_v, copy_bytes, cudaMemcpyDeviceToDevice);
    } else {
      std::memcpy(dst_k, src_k, copy_bytes);
      std::memcpy(dst_v, src_v, copy_bytes);
    }
  }
}

float* KVManager::key_slot_ptr(int layer_idx, int slot, int pos) {
  int offset = layer_idx * max_batch_ * max_seq_len_ * kv_dim_ +
               slot * max_seq_len_ * kv_dim_ +
               pos * kv_dim_;
  return const_cast<float*>(key_cache_.ptr<float>(offset));
}

float* KVManager::value_slot_ptr(int layer_idx, int slot, int pos) {
  int offset = layer_idx * max_batch_ * max_seq_len_ * kv_dim_ +
               slot * max_seq_len_ * kv_dim_ +
               pos * kv_dim_;
  return const_cast<float*>(value_cache_.ptr<float>(offset));
}

bool KVManager::has_free_slot() const {
  return free_slot_count() > 0;
}

int KVManager::busy_slot_count() const {
  int count = 0;
  for (bool busy : slot_busy_) {
    if (busy) ++count;
  }
  return count;
}

int KVManager::free_slot_count() const {
  int count = 0;
  for (bool busy : slot_busy_) {
    if (!busy) ++count;
  }
  return count;
}

}  // namespace scheduler
