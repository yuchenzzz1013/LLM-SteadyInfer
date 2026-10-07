#ifndef SRC_INCLUDE_BASE_ALLOC_H_
#define SRC_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include <mutex>
#include <vector>
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

  // Allocators are held and passed around as shared_ptr<DeviceAllocator>;
  // without a virtual destructor, destroying a derived allocator through the
  // base pointer is undefined behaviour.
  virtual ~DeviceAllocator() = default;

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

// Caching CUDA device allocator: one pool per device, with splitting and
// coalescing.
//
// Every pooled block remembers the cudaMalloc'd *region* it was carved out of,
// which turns the pool into a real allocator instead of a cache keyed by exact
// size:
//   * allocate() is best-fit + split — the smallest idle block that fits is
//     handed out and an oversized tail is split off and kept for reuse. A
//     request therefore never reaches cudaMalloc while *any* idle block big
//     enough exists.
//   * release() merges the returned block with idle neighbours from the same
//     region, so a region that goes fully idle collapses back into a single
//     block.
//   * Idle bytes of every block — small and large alike — count toward the
//     waterline, and both the waterline flush in release() and free_idle()
//     hand fully idle regions back to the driver.
//   * ~CUDADeviceAllocator returns every region the pool still holds.
//   * All pool state is guarded by a mutex, so concurrent workers cannot
//     corrupt the ledger or hand one block to two callers.
class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();
  ~CUDADeviceAllocator() override;

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

  // Return every fully idle region to the driver. A block whose region still
  // contains a live allocation cannot be released (the driver has no partial
  // cudaFree); it stays pooled for reuse.
  void free_idle() const;

  // Pool occupancy, for metrics and leak checks. device_id < 0 means the
  // calling thread's current device.
  struct DeviceStats {
    size_t reserved_bytes = 0;  // device memory cudaMalloc'd by the pool
    size_t busy_bytes = 0;      // handed out to live buffers
    size_t idle_bytes = 0;      // cached and reusable
    size_t blocks = 0;          // pool entries
    size_t regions = 0;         // cudaMalloc'd regions
  };

  DeviceStats stats(int device_id = -1) const;

 private:
  // A slice of a cudaMalloc'd region. Blocks of one region tile it exactly and
  // never overlap; two idle blocks of the same region are always merged.
  struct Block {
    void* data = nullptr;
    size_t byte_size = 0;
    bool busy = false;
    void* region = nullptr;  // cudaMalloc'd base pointer this block belongs to
  };

  struct DevicePool {
    std::vector<Block> blocks;
    std::map<void*, size_t> regions;  // region base -> region size
    size_t idle_bytes = 0;
    size_t reserved_bytes = 0;
    size_t flush_threshold = 0;  // cached waterline, 0 until first computed
  };

  // Waterline above which release() flushes idle regions: max(total_mem * 5%,
  // 1GB), cached per device.
  size_t idle_flush_threshold(int device_id, DevicePool& pool) const;

  void* allocate_locked(int device_id, DevicePool& pool, size_t byte_size) const;

  // Marks the block owning ptr idle and merges it with its neighbours. Returns
  // false when ptr does not belong to this pool.
  bool release_locked(DevicePool& pool, void* ptr) const;

  void merge_idle_neighbours_locked(DevicePool& pool, size_t index) const;

  // Coalesces every adjacent pair of idle blocks (cold path only).
  void merge_all_idle_locked(DevicePool& pool) const;

  // Frees the regions that are entirely idle; returns the bytes released.
  size_t free_regions_locked(int device_id, DevicePool& pool) const;

  void flush_over_waterline_locked() const;

  void check_ledger_locked() const;

  mutable std::mutex mutex_;
  mutable std::map<int, DevicePool> pools_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    // Function-local static: the C++11 magic-static rule makes initialization
    // run exactly once even under concurrent first calls. The previous
    // `if (instance == nullptr) instance = ...` raced: two threads could both
    // see nullptr, each build an allocator, and one of them was then dropped
    // on the floor (its pool leaked) while a thread could also observe a
    // half-assigned shared_ptr.
    static std::shared_ptr<CPUDeviceAllocator> instance = std::make_shared<CPUDeviceAllocator>();
    return instance;
  }
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
    // Thread-safe magic static — see CPUDeviceAllocatorFactory::get_instance.
    static std::shared_ptr<PinnedCPUDeviceAllocator> instance =
        std::make_shared<PinnedCPUDeviceAllocator>();
    return instance;
  }
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    // Thread-safe magic static — see CPUDeviceAllocatorFactory::get_instance.
    // The instance is destroyed when the last shared_ptr dies, so tensors that
    // still hold a Buffer keep the allocator (and its pooled memory) alive.
    static std::shared_ptr<CUDADeviceAllocator> instance =
        std::make_shared<CUDADeviceAllocator>();
    return instance;
  }
};
}  // namespace base
#endif