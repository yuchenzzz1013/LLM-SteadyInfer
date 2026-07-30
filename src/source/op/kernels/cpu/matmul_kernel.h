#ifndef SRC_SOURCE_OP_KERNELS_CPU_MATMUL_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_MATMUL_KERNEL_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const CudaConfig* config = nullptr);
}  // namespace kernel
#endif
