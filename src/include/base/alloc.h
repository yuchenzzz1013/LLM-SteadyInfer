#ifndef SRC_INCLUDE_BASE_ALLOC_H_
#define SRC_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"
namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  virtual DeviceType device_type() const { return device_type_; }

  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t byte_size) const = 0;

  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;

  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool is_busy)
      : data(data), byte_size(byte_size), busy(is_busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

  // Free all idle (non-busy) buffers back to the GPU
  void free_idle() const;

 private:
  // Waterline above which release() flushes idle buffers: max(total_mem * 5%,
  // 1GB), cached per device.
  size_t idle_flush_threshold(int device_id) const;

  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
  mutable std::map<int, size_t> idle_thresholds_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

// Page-locked (pinned) host memory. Tensors built on it behave as ordinary
// CPU tensors (same device type, index<>/ptr<> work as usual), but the
// storage is page-locked, so cudaMemcpyAsync to/from it is a true
// asynchronous DMA. With pageable memory the driver first copies through an
// internal bounce buffer, which blocks the calling thread for the duration of
// the transfer — making the "async" staging copies on the decode hot path
// effectively synchronous.
class PinnedCPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit PinnedCPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

class PinnedCPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<PinnedCPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<PinnedCPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<PinnedCPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};
}  // namespace base
#endif