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
