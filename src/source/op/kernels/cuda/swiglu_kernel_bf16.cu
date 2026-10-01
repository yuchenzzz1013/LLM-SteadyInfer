// CUDA SwiGLU (BF16): out = silu(in1) * in2 with bf16 operands, sigmoid and
// product in float, bf16-rounded result.
#include <cuda_bf16.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_bf16_kernel(int size, const __nv_bfloat16* in1,
                                             const __nv_bfloat16* in2, __nv_bfloat16* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) {
    return;
  }
  const float x1 = __bfloat162float(in1[idx]);
  const float x2 = __bfloat162float(in2[idx]);
  const float sig = 1.0f / (1.0f + expf(-x1));
  out[idx] = __float2bfloat16(x1 * sig * x2);
}

// Fused gate/up GEMM variant: the FFN's w1 and w3 projections are stacked into
// one [2 * ffn, hidden] weight (see Qwen3Model::build_fused_w13_layers), so the
// GEMM emits one [rows, 2 * ffn] row whose first ffn values are the gate and
// whose second ffn values are the up projection — the same two operands the
// pair kernel above reads from two separate buffers. Reading them as two uint4
// (8 bf16 each) from one row keeps the per-element arithmetic identical
// (silu(x1) * x2, one bf16 rounding), so the fused FFN changes only the GEMM
// shape, not the activation's values.
__global__ void swiglu_fused_pair_kernel_bf16(const __nv_bfloat16* __restrict__ fused_in,
                                              __nv_bfloat16* __restrict__ out,
                                              int32_t chunks_per_row) {
  const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunks_per_row) {
    return;
  }
  const int64_t in_row = static_cast<int64_t>(blockIdx.y) * 2 * chunks_per_row * 8;
  const __nv_bfloat16* gate = fused_in + in_row + static_cast<int64_t>(c) * 8;
  const __nv_bfloat16* up = gate + static_cast<int64_t>(chunks_per_row) * 8;
  const uint4 graw = *reinterpret_cast<const uint4*>(gate);
  const uint4 uraw = *reinterpret_cast<const uint4*>(up);
  const __nv_bfloat162* gh = reinterpret_cast<const __nv_bfloat162*>(&graw);
  const __nv_bfloat162* uh = reinterpret_cast<const __nv_bfloat162*>(&uraw);
  __nv_bfloat162 o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    // Same expression as swiglu_kernel_bf16_kernel above (x * sig * y, not
    // x / (1 + exp): the reciprocal form rounds differently in fp32).
    const float x = __bfloat162float(gh[j].x);
    const float y = __bfloat162float(gh[j].y);
    const float sx = 1.0f / (1.0f + expf(-x));
    const float sy = 1.0f / (1.0f + expf(-y));
    o[j] = __floats2bfloat162_rn(x * sx * __bfloat162float(uh[j].x),
                                 y * sy * __bfloat162float(uh[j].y));
  }
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(blockIdx.y) * chunks_per_row * 8 +
                            static_cast<int64_t>(c) * 8) = *reinterpret_cast<const uint4*>(o);
}

void swiglu_kernel_cu_fused(const tensor::Tensor& fused_in, const tensor::Tensor& output,
                            int32_t rows, int32_t ffn_dim, void* stream) {
  CHECK(!fused_in.is_empty() && !output.is_empty());
  CHECK(fused_in.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(ffn_dim % 8, 0) << "Fused SwiGLU needs a gate width that is a multiple of 8.";
  CHECK_EQ(fused_in.size(), static_cast<int64_t>(rows) * 2 * ffn_dim);
  CHECK_EQ(output.size(), static_cast<int64_t>(rows) * ffn_dim);
  const int32_t chunks_per_row = ffn_dim / 8;
  const int threads = 128;
  dim3 grid((chunks_per_row + threads - 1) / threads, rows);
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
  swiglu_fused_pair_kernel_bf16<<<grid, threads, 0, stream_>>>(
      fused_in.ptr<__nv_bfloat16>(), const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()),
      chunks_per_row);
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                           const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  if (!stream) {
    swiglu_kernel_bf16_kernel<<<blocks, threads>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_bf16_kernel<<<blocks, threads, 0, stream_>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
  }
}
}  // namespace kernel
