#include "../kernels_interface.h"
#include "argmax_kernel.cuh"
#include "tensor/tensor.h"
#include <cfloat>
namespace kernel {
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

// Widen the upper 16 bits of the bit pattern: bfloat16 is the top half of
// IEEE float32, so the result is the exact float value of the bfloat16.
__forceinline__ __device__ float bf16_bits_to_fp32(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = static_cast<uint32_t>(bits) << 16;
  return conv.f;
}

__global__ void argmax_kernel(const uint16_t* input_ptr, size_t size, size_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  uint32_t tid = threadIdx.x;
  // Do NOT early-return: block_reduce_argmax uses a full-mask __ballot_sync,
  // which requires every thread in the block to participate. Guard the reads
  // instead so the kernel is correct for any `size` (incl. size < blockDim).
  bool valid = tid < size;
  size_t max_index = threadIdx.x;
  float max_value = valid ? bf16_bits_to_fp32(input_ptr[max_index]) : -FLT_MAX;
  for (size_t i = tid; i < size; i += blockDim.x) {
    const float v = bf16_bits_to_fp32(input_ptr[i]);
    if (v > max_value) {
      max_index = i;
      max_value = v;
    }
  }

  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    *output_idx = max_index;
  }
}

__global__ void argmax_kernel_batch(const uint16_t* input_ptr, size_t row_stride, size_t size,
                                    int32_t* output_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  uint32_t tid = threadIdx.x;
  // Same rule as argmax_kernel: no early return (full-mask ballot),
  // guard the reads instead.
  const uint16_t* row = input_ptr + blockIdx.x * row_stride;
  bool valid = tid < size;
  size_t max_index = threadIdx.x;
  float max_value = valid ? bf16_bits_to_fp32(row[max_index]) : -FLT_MAX;
  for (size_t i = tid; i < size; i += blockDim.x) {
    const float v = bf16_bits_to_fp32(row[i]);
    if (v > max_value) {
      max_index = i;
      max_value = v;
    }
  }

  block_reduce_argmax(max_value, max_index, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    // Token indices are int32 in this codebase; max_index < size <= 2^31.
    output_idx[blockIdx.x] = static_cast<int32_t>(max_index);
  }
}

size_t argmax_kernel_cu(const uint16_t* input_ptr, size_t size, void* stream) {
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  size_t* index = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
  size_t output_index = 0;
  if (!stream) {
    argmax_kernel<<<1, 512>>>(input_ptr, size, index);
    cudaMemcpy(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost);
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    argmax_kernel<<<1, 512, 0, stream_>>>(input_ptr, size, index);
    cudaMemcpyAsync(&output_index, index, sizeof(size_t), cudaMemcpyDeviceToHost, stream_);
    // The async copy is still in flight when this function returns the host
    // value; sync before reading output_index.
    cudaStreamSynchronize(stream_);
  }
  return output_index;
}

void argmax_kernel_cu_batch(const uint16_t* input_ptr, size_t row_stride, size_t size,
                            int32_t batch, int32_t* out_tokens, void* stream) {
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  int32_t* dev_idx =
      static_cast<int32_t*>(alloc_cu->allocate(static_cast<size_t>(batch) * sizeof(int32_t)));
  CHECK(dev_idx != nullptr) << "Failed to allocate device argmax index buffer";

  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    argmax_kernel_batch<<<batch, 512, 0, stream_>>>(input_ptr, row_stride, size, dev_idx);
    cudaMemcpyAsync(out_tokens, dev_idx, static_cast<size_t>(batch) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  } else {
    argmax_kernel_batch<<<batch, 512>>>(input_ptr, row_stride, size, dev_idx);
    cudaMemcpy(out_tokens, dev_idx, static_cast<size_t>(batch) * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
  }
}
}  // namespace kernel
