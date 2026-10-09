// CUDA element-wise add (BF16): raw bfloat16 operands, math in float,
// bf16-rounded result.
//
// Both kernels read/write 8 bf16 (one uint4) per thread step when the pointers
// and the element count allow it: a [batch, 2560] residual add is a pure
// bandwidth kernel, and 2-byte scalar accesses issue 8x the instructions for
// the same traffic. The scalar path stays as the fallback for unaligned views
// (16 B requires an element offset that is a multiple of 8) and odd tails, so
// the vector path can never fault on a caller's sub-view.
#include <base/cuda_check.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "add_kernel.cuh"
namespace kernel {
namespace {
// True when every pointer is 16-byte aligned, the requirement of the uint4
// loads (element offset % 8 == 0, since bf16 is 2 bytes).
__device__ __forceinline__ bool vec8_ok(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}
}  // namespace

__global__ void add_kernel_bf16_kernel(int32_t size, const __nv_bfloat16* in1,
                                       const __nv_bfloat16* in2, __nv_bfloat16* out) {
  const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const bool vec =
      size >= 8 && vec8_ok(in1) && vec8_ok(in2) && vec8_ok(out);
  const int32_t nvec = vec ? (size >> 3) : 0;
  if (tid < nvec) {
    const uint4 a = reinterpret_cast<const uint4*>(in1)[tid];
    const uint4 b = reinterpret_cast<const uint4*>(in2)[tid];
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[j] = __floats2bfloat162_rn(
          __bfloat162float(ha[j].x) + __bfloat162float(hb[j].x),
          __bfloat162float(ha[j].y) + __bfloat162float(hb[j].y));
    }
    reinterpret_cast<uint4*>(out)[tid] = *reinterpret_cast<const uint4*>(o);
    return;
  }
  // Tail (or the whole range, on the scalar path): threads past nvec cover the
  // remaining elements one at a time.
  const int32_t i = (nvec << 3) + (tid - nvec);
  if (tid >= nvec && i < size) {
    out[i] = __float2bfloat16(__bfloat162float(in1[i]) + __bfloat162float(in2[i]));
  }
}

// Broadcast bias add: out[rows, cols] += bias[cols] in one launch. The matmul
// layer used to emit one add launch per row here (a [1, K] add per row), i.e.
// `batch` launches per biased projection per layer.
__global__ void add_bias_kernel_bf16_kernel(int32_t rows, int32_t cols,
                                            const __nv_bfloat16* bias,
                                            __nv_bfloat16* out) {
  const bool vec = cols >= 8 && (cols & 7) == 0 && vec8_ok(bias) && vec8_ok(out);
  for (int32_t r = blockIdx.y; r < rows; r += gridDim.y) {
    __nv_bfloat16* row = out + static_cast<int64_t>(r) * cols;
    if (vec) {
      const int32_t nvec = cols >> 3;
      for (int32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < nvec;
           c += gridDim.x * blockDim.x) {
        const uint4 b = reinterpret_cast<const uint4*>(bias)[c];
        const uint4 a = reinterpret_cast<const uint4*>(row)[c];
        const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
        const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
        __nv_bfloat162 o[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          o[j] = __floats2bfloat162_rn(
              __bfloat162float(ha[j].x) + __bfloat162float(hb[j].x),
              __bfloat162float(ha[j].y) + __bfloat162float(hb[j].y));
        }
        reinterpret_cast<uint4*>(row)[c] = *reinterpret_cast<const uint4*>(o);
      }
    } else {
      for (int32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < cols;
           c += gridDim.x * blockDim.x) {
        row[c] = __float2bfloat16(__bfloat162float(row[c]) + __bfloat162float(bias[c]));
      }
    }
  }
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  // One thread per vec8 chunk (plus up to 7 tail lanes); the same predicate the
  // kernel applies internally decides whether that layout is usable.
  const bool vec = size >= 8 &&
                   (reinterpret_cast<uintptr_t>(input1.ptr<__nv_bfloat16>()) & 0xF) == 0 &&
                   (reinterpret_cast<uintptr_t>(input2.ptr<__nv_bfloat16>()) & 0xF) == 0 &&
                   (reinterpret_cast<uintptr_t>(output.ptr<__nv_bfloat16>()) & 0xF) == 0;
  int32_t lanes = vec ? (size >> 3) + 8 : size;
  int32_t block_num = (lanes + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_bf16_kernel<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
    CUDA_KERNEL_CHECK();
  } else {
    add_kernel_bf16_kernel<<<block_num, thread_num>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
    CUDA_KERNEL_CHECK();
  }
}

void add_bias_kernel_cu(const tensor::Tensor& output, const tensor::Tensor& bias,
                        int32_t rows, int32_t cols, void* stream) {
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(bias.is_empty(), false);
  CHECK_GT(rows, 0);
  CHECK_GT(cols, 0);
  CHECK_EQ(bias.size(), static_cast<int64_t>(cols));
  CHECK_EQ(output.size(), static_cast<int64_t>(rows) * cols);
  const int32_t threads = 256;
  const int32_t vec_lanes = (cols + 7) / 8;
  dim3 grid((vec_lanes + threads - 1) / threads, rows);
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
  add_bias_kernel_bf16_kernel<<<grid, threads, 0, stream_>>>(
      rows, cols, bias.ptr<__nv_bfloat16>(),
      const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
  CUDA_KERNEL_CHECK();
}
}  // namespace kernel
