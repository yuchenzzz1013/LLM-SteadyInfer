#ifndef SRC_SOURCE_OP_KERNELS_CPU_SWIGLU_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_SWIGLU_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}  // namespace kernel
#endif  
