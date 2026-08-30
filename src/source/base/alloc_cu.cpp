#include <algorithm>
#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

// Flush-idle threshold per device: max(total_mem * 5%, 1GB). Fixed 1GB is
// wasteful on small (24GB) cards; on large cards the 5% floor keeps the pool
// warm enough to absorb allocation spikes.
size_t CUDADeviceAllocator::idle_flush_threshold(int device_id) const {
  auto it = idle_thresholds_.find(device_id);
  if (it != idle_thresholds_.end()) {
    return it->second;
  }
  size_t total_mem = 0, free_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
    total_mem = 0;
  }
  const size_t threshold = std::max<size_t>(total_mem / 20, 1024ull * 1024 * 1024);
  idle_thresholds_[device_id] = threshold;
  return threshold;
}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  if (byte_size == 0) {
    LOG(WARNING) << "[CUDA_ALLOC] Attempted to allocate 0 bytes; returning nullptr";
    return nullptr;
  }
  // On OOM, recycle idle pooled buffers once and retry before giving up
  // (spin-reclaim: frees what the pool hoards before surfacing the error).
  auto malloc_with_retry = [this](void** ptr, size_t size) -> cudaError_t {
    cudaError_t st = cudaMalloc(ptr, size);
    if (st != cudaSuccess) {
      LOG(WARNING) << "[CUDA_ALLOC] cudaMalloc(" << (size >> 20)
                   << "MB) failed; recycling idle pool buffers and retrying once.";
      free_idle();
      st = cudaMalloc(ptr, size);
    }
    return st;
  };

  if (byte_size > 1024 * 1024) {
    auto& big_buffers = big_buffers_map_[id];
    int sel_id = -1;
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
          big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
        if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
          sel_id = i;
        }
      }
    }
    if (sel_id != -1) {
      big_buffers[sel_id].busy = true;
#ifndef NDEBUG
      VLOG(1) << "[CUDA_ALLOC] BIG reuse id=" << sel_id
                << " size=" << (byte_size >> 10) << "KB"
                << " (pool big_buffers cnt=" << big_buffers.size() << ")";
#endif
      return big_buffers[sel_id].data;
    }

    void* ptr = nullptr;
    state = malloc_with_retry(&ptr, byte_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB: %s (%d)! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20, cudaGetErrorString(state), static_cast<int>(state));
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
#ifndef NDEBUG
    VLOG(1) << "[CUDA_ALLOC] BIG new size=" << (byte_size >> 10) << "KB"
              << " (pool big_buffers cnt=" << big_buffers.size() << ")";
#endif
    return ptr;
  }

  auto& cuda_buffers = cuda_buffers_map_[id];
  // Best-fit: hand out the smallest idle buffer that satisfies the request.
  // First-fit instead lets a small tensor consume a large pooled buffer, so
  // the next large-tensor request finds nothing reusable and cudaMallocs a
  // fresh one every step — the pool then grows without bound until the
  // device OOMs (observed: +808KB/step of partial_batch buffers).
  int sel_id = -1;
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy &&
        (sel_id == -1 || cuda_buffers[i].byte_size < cuda_buffers[sel_id].byte_size)) {
      sel_id = i;
    }
  }
  if (sel_id != -1) {
    cuda_buffers[sel_id].busy = true;
    no_busy_cnt_[id] -= cuda_buffers[sel_id].byte_size;
#ifndef NDEBUG
    VLOG(1) << "[CUDA_ALLOC] SMALL reuse id=" << sel_id
              << " asked=" << byte_size << "B"
              << " have=" << cuda_buffers[sel_id].byte_size << "B"
              << " (pool small cnt=" << cuda_buffers.size()
              << " idle=" << (no_busy_cnt_[id] >> 10) << "KB)";
#endif
    return cuda_buffers[sel_id].data;
  }
  void* ptr = nullptr;
  state = malloc_with_retry(&ptr, byte_size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB: %s (%d)! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20, cudaGetErrorString(state), static_cast<int>(state));
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
#ifndef NDEBUG
  VLOG(1) << "[CUDA_ALLOC] SMALL new size=" << byte_size << "B"
            << " (pool small cnt=" << cuda_buffers.size()
            << " idle=" << (no_busy_cnt_[id] >> 10) << "KB)";
#endif
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  if (cuda_buffers_map_.empty()) {
    LOG(WARNING) << "[CUDA_FREE] release called but cuda_buffers_map_ is empty, cudaFree directly";
    cudaError_t state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
    return;
  }
  cudaError_t state = cudaSuccess;
  for (auto& it : cuda_buffers_map_) {
    if (no_busy_cnt_[it.first] > idle_flush_threshold(it.first)) {
#ifndef NDEBUG
      VLOG(1) << "[CUDA_FREE] idle above waterline ("
              << (idle_flush_threshold(it.first) >> 20) << "MB), flushing all non-busy small buffers";
#endif
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for (int i = 0; i < cuda_buffers.size(); i++) {
        if (!cuda_buffers[i].busy) {
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
          CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << it.first;
        } else {
          temp.push_back(cuda_buffers[i]);
        }
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
#ifndef NDEBUG
      VLOG(1) << "[CUDA_FREE] flush done, small pool now " << temp.size() << " busy buffers";
#endif
    }
  }

  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
#ifndef NDEBUG
        VLOG(1) << "[CUDA_FREE] SMALL release idx=" << i
                  << " size=" << cuda_buffers[i].byte_size << "B"
                  << " idle_now=" << (no_busy_cnt_[it.first] >> 10) << "KB";
#endif
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
#ifndef NDEBUG
        VLOG(1) << "[CUDA_FREE] BIG release idx=" << i
                  << " size=" << (big_buffers[i].byte_size >> 10) << "KB";
#endif
        return;
      }
    }
  }
  // Not found in pool — free directly
  LOG(WARNING) << "[CUDA_FREE] ptr not found in pool, cudaFree directly";
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}
void CUDADeviceAllocator::free_idle() const {
  for (auto& it : cuda_buffers_map_) {
    int device_id = it.first;
    auto& buffers = it.second;
    cudaSetDevice(device_id);
    std::vector<CudaMemoryBuffer> kept;
    for (auto& buf : buffers) {
      if (!buf.busy) {
        cudaFree(buf.data);
      } else {
        kept.push_back(buf);
      }
    }
    buffers = std::move(kept);
    no_busy_cnt_[device_id] = 0;
  }
  for (auto& it : big_buffers_map_) {
    int device_id = it.first;
    auto& buffers = it.second;
    cudaSetDevice(device_id);
    std::vector<CudaMemoryBuffer> kept;
    for (auto& buf : buffers) {
      if (!buf.busy) {
        cudaFree(buf.data);
      } else {
        kept.push_back(buf);
      }
    }
    buffers = std::move(kept);
  }
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base