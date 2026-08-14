#ifndef SRC_SOURCE_OP_KERNELS_CPU_RMSNORM_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);

// Batched RMSNorm over the last dim (OpenMP over rows).
void rmsnorm_kernel_cpu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t dim, void* stream = nullptr);
}  // namespace kernel
#endif 
