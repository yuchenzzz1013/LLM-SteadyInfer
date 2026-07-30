#ifndef SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#include <tensor/tensor.h>
namespace kernel {
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);
}
#endif  // SWIGLU_KERNEL_CU_CUH
