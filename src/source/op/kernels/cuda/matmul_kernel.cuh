#ifndef SRC_SOURCE_OP_KERNELS_CUDA_MATMUL_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_MATMUL_KERNEL_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
// CUDA BF16 GEMM: cuBLAS BF16 Tensor Core path (fp32 accumulation) with the
// custom-kernel fallback for pre-Ampere / cublas failures.
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);
}  // namespace kernel

#endif 
