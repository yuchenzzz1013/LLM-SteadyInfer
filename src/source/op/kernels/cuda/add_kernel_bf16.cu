// CUDA element-wise add (BF16): raw bfloat16 operands, math in float,
// bf16-rounded result.
#include <cuda_bf16.h>
#include "add_kernel.cuh"
namespace kernel {
__global__ void add_kernel_bf16_kernel(int32_t size, const __nv_bfloat16* in1,
                                          const __nv_bfloat16* in2, __nv_bfloat16* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  out[tid] = __float2bfloat16(__bfloat162float(in1[tid]) + __bfloat162float(in2[tid]));
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
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_bf16_kernel<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
  } else {
    add_kernel_bf16_kernel<<<block_num, thread_num>>>(
        size, input1.ptr<__nv_bfloat16>(), input2.ptr<__nv_bfloat16>(),
        const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>()));
  }
}
}  // namespace kernel
