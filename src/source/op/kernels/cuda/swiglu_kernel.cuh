#ifndef SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_SWIGLU_KERNEL_CUH
#include <tensor/tensor.h>
namespace kernel {
// CUDA SwiGLU: out = silu(in1) * in2 on raw bfloat16 (sigmoid/product in
// float, bf16-rounded result).
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);

// Fused gate/up GEMM variant: `fused_in` is the [rows, 2 * ffn_dim] output of
// the stacked w1/w3 GEMM (gate in the first ffn_dim columns, up in the rest),
// `output` is the [rows, ffn_dim] activation. Same per-element arithmetic as
// swiglu_kernel_cu, one launch instead of a GEMM + a kernel.
void swiglu_kernel_cu_fused(const tensor::Tensor& fused_in, const tensor::Tensor& output,
                            int32_t rows, int32_t ffn_dim, void* stream);
}
#endif  // SWIGLU_KERNEL_CU_CUH
