#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      CHECK_EQ(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice), cudaSuccess);
    } else {
      CHECK_EQ(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_),
               cudaSuccess);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      CHECK_EQ(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost), cudaSuccess);
    } else {
      CHECK_EQ(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_),
               cudaSuccess);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      CHECK_EQ(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice), cudaSuccess);
    } else {
      CHECK_EQ(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_),
               cudaSuccess);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  // Red line: NEVER call cudaDeviceSynchronize() on the inference hot path —
  // a global sync blocks every stream and serializes concurrent work.
  // Sync only the stream that owns this transfer (cudaStreamSynchronize).
  if (need_sync) {
    if (stream_) {
      CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    // Without a stream the copy above was issued as a synchronous cudaMemcpy /
    // cudaMemset, which already blocks until completion — nothing to sync.
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  CHECK_NE(ptr, nullptr);
  if (!byte_size) {
    return;
  }

  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else if (stream_) {
    CHECK_EQ(cudaMemsetAsync(ptr, 0, byte_size, stream_), cudaSuccess);
  } else {
    CHECK_EQ(cudaMemset(ptr, 0, byte_size), cudaSuccess);
  }
  // Stream-scoped sync only — see the note in memcpy above.
  if (need_sync && stream_) {
    CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }
}

}  // namespace base