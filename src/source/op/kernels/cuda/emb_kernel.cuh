#ifndef SRC_SOURCE_OP_KERNELS_CUDA_EMB_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_EMB_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
// CUDA embedding gather: raw bf16 element copy of the table rows.
void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}
#endif  // EMB_KERNEL_H
