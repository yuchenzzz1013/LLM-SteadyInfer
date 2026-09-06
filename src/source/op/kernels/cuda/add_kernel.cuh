#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ADD_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ADD_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
// CUDA element-wise add on raw bfloat16 (math in float, bf16-rounded result).
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif 
