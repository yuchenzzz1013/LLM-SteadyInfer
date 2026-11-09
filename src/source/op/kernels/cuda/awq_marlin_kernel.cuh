#ifndef SRC_SOURCE_OP_KERNELS_CUDA_AWQ_MARLIN_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_AWQ_MARLIN_KERNEL_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
// Marlin-style W4A16 GEMM. The checkpoint's qweight [K, N/8] is repacked once
// at load time into [K/16, N, 2] int32 (16 nibbles of one output column per
// 8-byte word pair), so the GEMM reads each thread's B fragment with one
// vectorized load and dequantizes it in registers.
constexpr int32_t kAwqMarlinBlockK = 128;

// Whether the shape can use the repacked layout (K must be a whole number of
// 128-wide k chunks and the group size a multiple of 16).
bool awq_marlin_supported(int32_t K, int32_t N, int32_t out_offset, int32_t group_size);

// Repack a staged device qweight ([K, N/8] int32, AWQ "gemm" nibble order) into
// the Marlin layout. Returns an empty tensor when the shape is unsupported.
tensor::Tensor awq_marlin_pack(const tensor::Tensor& qweight, int32_t K, int32_t N_total,
                               cudaStream_t stream);

void awq_marlin_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                          const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                          const tensor::Tensor& output, const AwqMatmulParams& params,
                          const CudaConfig* config);
}  // namespace kernel
#endif
