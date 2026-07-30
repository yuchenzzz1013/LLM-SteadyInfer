#ifndef SRC_SOURCE_OP_KERNELS_CPU_SOFTMAX_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif  
