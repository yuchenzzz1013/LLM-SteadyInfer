#ifndef SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#include <tensor/tensor.h>
namespace kernel {
// CUDA SwiGLU: out = silu(in1) * in2 on raw bfloat16 (sigmoid/product in
// float, bf16-rounded result).
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);
}
#endif  // SWIGLU_KERNEL_CU_CUH
