#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

PinnedCPUDeviceAllocator::PinnedCPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* PinnedCPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
  void* data = nullptr;
  const cudaError_t err = cudaHostAlloc(&data, byte_size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    LOG(ERROR) << "cudaHostAlloc(" << byte_size << ") failed: " << cudaGetErrorString(err);
    return nullptr;
  }
  return data;
}

void PinnedCPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    // Errors can only mean the context is already gone (teardown ordering),
    // so there is nothing useful to do with them here.
    cudaFreeHost(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
std::shared_ptr<PinnedCPUDeviceAllocator> PinnedCPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base