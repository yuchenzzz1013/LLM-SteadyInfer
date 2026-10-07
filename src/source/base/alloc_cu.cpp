#include <algorithm>
#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base {

namespace {
// cudaMalloc hands out 256-byte aligned pointers; rounding requests up to that
// keeps blocks of one region tiling it exactly.
constexpr size_t kAlignment = 256;
// Do not leave a splinter smaller than one alignment unit behind when
// splitting a block.
constexpr size_t kMinSplitBytes = kAlignment;
// Never cache less than this much idle memory: the pool exists to absorb
// per-step allocation spikes, so the waterline is at least 1GB even on small
// cards (on large cards the 5%-of-total term dominates).
constexpr size_t kMinIdleFlushBytes = 1024ull * 1024 * 1024;

size_t round_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

// cudaMalloc/cudaFree act on the calling thread's *current* device. Switch to
// `device_id` for the duration of one pool operation and put the thread back
// where it was: without the restore, reclaiming memory inside allocate() would
// leave the thread on another device and the retried cudaMalloc would land
// there while the pool booked the buffer under the original device.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    saved_ = current;
    if (current != device_id) {
      switched_ = (cudaSetDevice(device_id) == cudaSuccess);
    }
  }
  ~DeviceGuard() {
    if (switched_) {
      cudaSetDevice(saved_);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int saved_ = -1;
  bool switched_ = false;
};
}  // namespace

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

CUDADeviceAllocator::~CUDADeviceAllocator() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) {
    DevicePool& pool = entry.second;
    size_t live_bytes = 0;
    for (const Block& block : pool.blocks) {
      if (block.busy) {
        live_bytes += block.byte_size;
      }
    }
    if (live_bytes != 0) {
      // The pool owns the memory, so it is released either way; a live block
      // at destruction means a Buffer outlived its owner.
      LOG(WARNING) << "[CUDA_ALLOC] destroying allocator on device " << entry.first << " with "
                   << (live_bytes >> 20) << "MB still handed out to live buffers";
    }
    DeviceGuard guard(entry.first);
    for (const auto& region : pool.regions) {
      if (cudaFree(region.first) != cudaSuccess) {
        // Expected on teardown after the context is gone; nothing to do.
        VLOG(1) << "[CUDA_ALLOC] cudaFree(" << region.first
                << ") failed during allocator destruction: " << cudaGetErrorString(cudaGetLastError());
        cudaGetLastError();
      }
    }
    pool.blocks.clear();
    pool.regions.clear();
    pool.idle_bytes = 0;
    pool.reserved_bytes = 0;
  }
}

size_t CUDADeviceAllocator::idle_flush_threshold(int device_id, DevicePool& pool) const {
  if (pool.flush_threshold != 0) {
    return pool.flush_threshold;
  }
  size_t total_mem = 0, free_mem = 0;
  {
    DeviceGuard guard(device_id);
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
      cudaGetLastError();
      total_mem = 0;
    }
  }
  pool.flush_threshold = std::max<size_t>(total_mem / 20, kMinIdleFlushBytes);
  return pool.flush_threshold;
}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  if (byte_size == 0) {
    LOG(WARNING) << "[CUDA_ALLOC] Attempted to allocate 0 bytes; returning nullptr";
    return nullptr;
  }
  int device_id = -1;
  cudaError_t state = cudaGetDevice(&device_id);
  CHECK(state == cudaSuccess) << "cudaGetDevice failed: " << cudaGetErrorString(state);

  std::lock_guard<std::mutex> lock(mutex_);
  void* ptr = allocate_locked(device_id, pools_[device_id], byte_size);
#ifndef NDEBUG
  check_ledger_locked();
#endif
  return ptr;
}

void* CUDADeviceAllocator::allocate_locked(int device_id, DevicePool& pool,
                                           size_t byte_size) const {
  const size_t wanted = round_up(byte_size, kAlignment);

  // Best fit: the smallest idle block that still satisfies the request. The
  // old code required |have - asked| < 1MB, so once allocation sizes drifted
  // apart every request cudaMalloc'd a fresh block and the pool grew without
  // bound instead of reusing what it already held.
  size_t best = pool.blocks.size();
  for (size_t i = 0; i < pool.blocks.size(); ++i) {
    const Block& block = pool.blocks[i];
    if (block.busy || block.byte_size < wanted) {
      continue;
    }
    if (best == pool.blocks.size() || block.byte_size < pool.blocks[best].byte_size) {
      best = i;
    }
  }
  if (best != pool.blocks.size()) {
    const size_t have = pool.blocks[best].byte_size;
    const size_t leftover = have - wanted;
    void* const data = pool.blocks[best].data;
    void* const region = pool.blocks[best].region;
    if (leftover >= kMinSplitBytes) {
      // Split: hand out the head, keep the tail pooled and reusable. The
      // block was idle and only its head becomes busy, so the ledger drops by
      // exactly the requested size.
      pool.blocks[best].byte_size = wanted;
      pool.blocks[best].busy = true;
      pool.idle_bytes -= wanted;
      pool.blocks.push_back(Block{static_cast<char*>(data) + wanted, leftover, false, region});
    } else {
      // Internal fragmentation smaller than one alignment unit: hand out the
      // whole block.
      pool.blocks[best].busy = true;
      pool.idle_bytes -= have;
    }
#ifndef NDEBUG
    VLOG(1) << "[CUDA_ALLOC] reuse asked=" << byte_size << "B have=" << have << "B"
              << " (pool blocks=" << pool.blocks.size() << " idle=" << (pool.idle_bytes >> 10)
              << "KB)";
#endif
    return data;
  }

  // Miss: grow the pool with a region holding exactly this block. Sizes are
  // tracked per region so the whole region can later go back to the driver.
  void* ptr = nullptr;
  cudaError_t state = cudaSuccess;
  {
    DeviceGuard guard(device_id);
    state = cudaMalloc(&ptr, wanted);
    if (state != cudaSuccess) {
      cudaGetLastError();
      LOG(WARNING) << "[CUDA_ALLOC] cudaMalloc(" << (wanted >> 20)
                   << "MB) failed; returning idle pool regions and retrying once.";
      // Reclaiming idle regions (small and large) before surfacing the error
      // is what keeps a fragmented pool from failing a request the device
      // could still satisfy.
      free_regions_locked(device_id, pool);
      state = cudaMalloc(&ptr, wanted);
    }
  }
  if (cudaSuccess != state) {
    char buf[256];
    snprintf(buf, 256,
             "Error: CUDA error when allocating %lu MB: %s (%d)! maybe there's no enough memory "
             "left on  device.",
             byte_size >> 20, cudaGetErrorString(state), static_cast<int>(state));
    LOG(ERROR) << buf;
    cudaGetLastError();
    return nullptr;
  }
  pool.regions[ptr] = wanted;
  pool.reserved_bytes += wanted;
  pool.blocks.push_back(Block{ptr, wanted, true, ptr});
#ifndef NDEBUG
  VLOG(1) << "[CUDA_ALLOC] new size=" << byte_size << "B"
            << " (pool blocks=" << pool.blocks.size() << " reserved="
            << (pool.reserved_bytes >> 20) << "MB)";
#endif
  return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
  if (!ptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) {
    if (release_locked(entry.second, ptr)) {
#ifndef NDEBUG
      check_ledger_locked();
#endif
      flush_over_waterline_locked();
      return;
    }
  }

  // No pool knows this pointer: it did not come from this allocator (or the
  // region was already handed back). Free it directly.
  LOG(WARNING) << "[CUDA_FREE] ptr not found in pool, cudaFree directly";
  if (cudaFree(ptr) != cudaSuccess) {
    // The context can already be gone during process teardown.
    cudaGetLastError();
  }
}

bool CUDADeviceAllocator::release_locked(DevicePool& pool, void* ptr) const {
  for (size_t i = 0; i < pool.blocks.size(); ++i) {
    Block& block = pool.blocks[i];
    if (block.data != ptr) {
      continue;
    }
    if (!block.busy) {
      // The pool already owns this block: a second release would make the
      // allocator hand the same memory to two live buffers (silent aliasing).
      // Surface it loudly and leave the pool untouched.
      LOG(ERROR) << "[CUDA_FREE] double release of pooled buffer " << ptr << " ("
                 << (block.byte_size >> 10) << "KB); ignoring";
      return true;
    }
    block.busy = false;
    pool.idle_bytes += block.byte_size;
    merge_idle_neighbours_locked(pool, i);
#ifndef NDEBUG
    VLOG(1) << "[CUDA_FREE] release size=" << (block.byte_size >> 10)
              << "KB idle_now=" << (pool.idle_bytes >> 10) << "KB";
#endif
    return true;
  }
  return false;
}

void CUDADeviceAllocator::merge_idle_neighbours_locked(DevicePool& pool, size_t index) const {
  // Repeatedly absorb adjacent idle blocks of the same region. Merging one
  // side can make the other side adjacent to the next block, hence the loop.
  bool merged = true;
  while (merged) {
    merged = false;
    char* begin = static_cast<char*>(pool.blocks[index].data);
    char* end = begin + pool.blocks[index].byte_size;
    const void* region = pool.blocks[index].region;
    for (size_t j = 0; j < pool.blocks.size(); ++j) {
      if (j == index) {
        continue;
      }
      const Block& other = pool.blocks[j];
      if (other.busy || other.region != region) {
        continue;
      }
      char* other_begin = static_cast<char*>(other.data);
      char* other_end = other_begin + other.byte_size;
      if (other_end == begin) {
        pool.blocks[index].data = other_begin;
        pool.blocks[index].byte_size += other.byte_size;
      } else if (end == other_begin) {
        pool.blocks[index].byte_size += other.byte_size;
      } else {
        continue;
      }
      pool.blocks.erase(pool.blocks.begin() + static_cast<std::ptrdiff_t>(j));
      if (j < index) {
        --index;
      }
      merged = true;
      break;
    }
  }
}

void CUDADeviceAllocator::merge_all_idle_locked(DevicePool& pool) const {
  // Cold-path safety net: release() already merges on every free, so this only
  // has work if the invariant was broken elsewhere.
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t i = 0; i < pool.blocks.size() && !merged; ++i) {
      if (pool.blocks[i].busy) {
        continue;
      }
      const size_t before = pool.blocks.size();
      merge_idle_neighbours_locked(pool, i);
      merged = pool.blocks.size() != before;
    }
  }
}

size_t CUDADeviceAllocator::free_regions_locked(int device_id, DevicePool& pool) const {
  DeviceGuard guard(device_id);
  size_t freed_bytes = 0;
  std::vector<Block> kept;
  kept.reserve(pool.blocks.size());
  for (const Block& block : pool.blocks) {
    // Only a block covering its whole region can go back to the driver:
    // cudaFree cannot release part of a cudaMalloc.
    const auto region = pool.regions.find(block.region);
    const bool whole_region_idle = !block.busy && block.data == block.region &&
                                   region != pool.regions.end() &&
                                   region->second == block.byte_size;
    if (whole_region_idle && cudaFree(block.data) == cudaSuccess) {
      pool.reserved_bytes -= block.byte_size;
      pool.idle_bytes -= block.byte_size;
      pool.regions.erase(block.region);
      freed_bytes += block.byte_size;
      continue;
    }
    if (whole_region_idle) {
      LOG(WARNING) << "[CUDA_FREE] cudaFree(" << block.data
                   << ") failed: " << cudaGetErrorString(cudaGetLastError());
      cudaGetLastError();
    }
    kept.push_back(block);
  }
  pool.blocks = std::move(kept);
#ifndef NDEBUG
  if (freed_bytes != 0) {
    VLOG(1) << "[CUDA_FREE] flushed " << (freed_bytes >> 20) << "MB idle to the driver"
              << " (pool now blocks=" << pool.blocks.size() << " reserved="
              << (pool.reserved_bytes >> 20) << "MB)";
  }
#endif
  return freed_bytes;
}

void CUDADeviceAllocator::flush_over_waterline_locked() const {
  for (auto& entry : pools_) {
    DevicePool& pool = entry.second;
    if (pool.idle_bytes <= idle_flush_threshold(entry.first, pool)) {
      continue;
    }
#ifndef NDEBUG
    VLOG(1) << "[CUDA_FREE] idle above waterline (" << (idle_flush_threshold(entry.first, pool) >> 20)
              << "MB), flushing idle regions of device " << entry.first;
#endif
    free_regions_locked(entry.first, pool);
  }
}

void CUDADeviceAllocator::free_idle() const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : pools_) {
    merge_all_idle_locked(entry.second);
    free_regions_locked(entry.first, entry.second);
  }
}

CUDADeviceAllocator::DeviceStats CUDADeviceAllocator::stats(int device_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_id < 0) {
    const cudaError_t state = cudaGetDevice(&device_id);
    if (state != cudaSuccess) {
      cudaGetLastError();
      return DeviceStats{};
    }
  }
  DeviceStats stats;
  const auto it = pools_.find(device_id);
  if (it == pools_.end()) {
    return stats;
  }
  const DevicePool& pool = it->second;
  stats.reserved_bytes = pool.reserved_bytes;
  stats.idle_bytes = pool.idle_bytes;
  stats.busy_bytes = pool.reserved_bytes - pool.idle_bytes;
  stats.blocks = pool.blocks.size();
  stats.regions = pool.regions.size();
  return stats;
}

void CUDADeviceAllocator::check_ledger_locked() const {
#ifndef NDEBUG
  for (const auto& entry : pools_) {
    size_t idle = 0;
    size_t reserved = 0;
    for (const Block& block : entry.second.blocks) {
      if (!block.busy) {
        idle += block.byte_size;
      }
    }
    for (const auto& region : entry.second.regions) {
      reserved += region.second;
    }
    CHECK_EQ(idle, entry.second.idle_bytes)
        << "idle ledger desync on device " << entry.first;
    CHECK_EQ(reserved, entry.second.reserved_bytes)
        << "reserved ledger desync on device " << entry.first;
  }
#endif
}

}  // namespace base
