#ifndef SRC_SOURCE_OP_KERNELS_CPU_AWQ_MATMUL_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_AWQ_MATMUL_KERNEL_H
#include "../kernels_interface.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
// CPU AWQ W4A16 GEMM: packed int4 weights are dequantized on the fly inside
// the dot product. Compute dtype is FP32 on the CPU backend.
void awq_matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                           const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                           const tensor::Tensor& output, const AwqMatmulParams& params,
                           const CudaConfig* config = nullptr);
}  // namespace kernel
#endif
