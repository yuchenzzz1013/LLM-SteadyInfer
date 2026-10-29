#ifndef SRC_SOURCE_OP_KERNELS_CUDA_AWQ_MATMUL_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_AWQ_MATMUL_KERNEL_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
// CUDA AWQ W4A16 GEMM: int4 weights stay packed in memory and are dequantized
// into shared memory inside the GEMM (BF16 compute, FP32 accumulation).
void awq_matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                          const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                          const tensor::Tensor& output, const AwqMatmulParams& params,
                          const CudaConfig* config = nullptr);
}  // namespace kernel
#endif
