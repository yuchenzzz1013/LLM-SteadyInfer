#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);

// Batched argmax over row-major logits [batch, row_stride]: each block handles
// one row and only the first `size` entries (so sampling can be restricted to
// the tokenizer vocab even when the logits row is wider).
void argmax_kernel_cu_batch(const float* input_ptr, size_t row_stride, size_t size,
                            int32_t batch, int32_t* out_tokens, void* stream);
}
#endif  
