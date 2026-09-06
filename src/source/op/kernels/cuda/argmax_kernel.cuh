#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
namespace kernel {
// CUDA argmax over BF16 logits (raw uint16_t bfloat16 bit patterns, widened
// to float before comparing — exact, since every bfloat16 value is exactly
// representable in float32).
size_t argmax_kernel_cu(const uint16_t* input_ptr, size_t size, void* stream);

// Batched argmax over row-major logits [batch, row_stride]: each block handles
// one row and only the first `size` entries (so sampling can be restricted to
// the tokenizer vocab even when the logits row is wider).
void argmax_kernel_cu_batch(const uint16_t* input_ptr, size_t row_stride, size_t size,
                            int32_t batch, int32_t* out_tokens, void* stream);
}
#endif
