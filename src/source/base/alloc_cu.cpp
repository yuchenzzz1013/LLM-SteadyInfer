#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError_t state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess);
  if (byte_size == 0) {
    LOG(WARNING) << "[CUDA_ALLOC] Attempted to allocate 0 bytes; returning nullptr";
    return nullptr;
  }
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
      VLOG(1) << "[CUDA_ALLOC] BIG reuse id=" << sel_id
                << " size=" << (byte_size >> 10) << "KB"
                << " (pool big_buffers cnt=" << big_buffers.size() << ")";
      return big_buffers[sel_id].data;
    }

    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(ptr, byte_size, true);
    VLOG(1) << "[CUDA_ALLOC] BIG new size=" << (byte_size >> 10) << "KB"
              << " (pool big_buffers cnt=" << big_buffers.size() << ")";
    return ptr;
  }

  auto& cuda_buffers = cuda_buffers_map_[id];
  for (int i = 0; i < cuda_buffers.size(); i++) {
    if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
      cuda_buffers[i].busy = true;
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
      VLOG(1) << "[CUDA_ALLOC] SMALL reuse id=" << i
                << " asked=" << byte_size << "B"
                << " have=" << cuda_buffers[i].byte_size << "B"
                << " (pool small cnt=" << cuda_buffers.size()
                << " idle=" << (no_busy_cnt_[id] >> 10) << "KB)";
      return cuda_buffers[i].data;
    }
  }
  void* ptr = nullptr;
  state = cudaMalloc(&ptr, byte_size);
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(ptr, byte_size, true);
  VLOG(1) << "[CUDA_ALLOC] SMALL new size=" << byte_size << "B"
            << " (pool small cnt=" << cuda_buffers.size()
            << " idle=" << (no_busy_cnt_[id] >> 10) << "KB)";
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
    if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
      VLOG(1) << "[CUDA_FREE] idle > 1GB, flushing all non-busy small buffers";
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
      VLOG(1) << "[CUDA_FREE] flush done, small pool now " << temp.size() << " busy buffers";
    }
  }

  for (auto& it : cuda_buffers_map_) {
    auto& cuda_buffers = it.second;
    for (int i = 0; i < cuda_buffers.size(); i++) {
      if (cuda_buffers[i].data == ptr) {
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        VLOG(1) << "[CUDA_FREE] SMALL release idx=" << i
                  << " size=" << cuda_buffers[i].byte_size << "B"
                  << " idle_now=" << (no_busy_cnt_[it.first] >> 10) << "KB";
        return;
      }
    }
    auto& big_buffers = big_buffers_map_[it.first];
    for (int i = 0; i < big_buffers.size(); i++) {
      if (big_buffers[i].data == ptr) {
        big_buffers[i].busy = false;
        VLOG(1) << "[CUDA_FREE] BIG release idx=" << i
                  << " size=" << (big_buffers[i].byte_size >> 10) << "KB";
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