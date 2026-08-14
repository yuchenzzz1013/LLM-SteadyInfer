#ifndef SRC_SOURCE_OP_KERNELS_CPU_ROPE_KERNEL_H
#define SRC_SOURCE_OP_KERNELS_CPU_ROPE_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                             float rope_theta = 1000000.0f);

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream);

// Batched RoPE: input_q [batch, dim], input_k [batch, kv_dim], input_pos [batch]
// (host int32). One OpenMP task per (batch, rotary-pair) — single pass for the
// whole batch instead of a per-sequence loop.
void rope_kernel_cpu_batch(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);
}  // namespace kernel
#endif
