#ifndef SRC_SOURCE_OP_KERNELS_CPU_SCALE_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_SCALE_KERNEL_H
#include <tensor/tensor.h>
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}
#endif 
