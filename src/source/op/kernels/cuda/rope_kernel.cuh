#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ROPE_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ROPE_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

// Batched RoPE: q [batch, dim], k [batch, kv_dim], positions [batch] (CUDA tensor).
// One kernel launch for the whole batch instead of one per sequence.
void rope_kernel_cu_batch(int32_t dim, int32_t kv_dim, int32_t head_size,
                          const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                          const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                          const tensor::Tensor& cos_cache, void* stream);

// Scatter rows of src [batch, kv_dim] into an external KV cache tensor laid
// out head-dim-contiguous as [num_layers, num_slots, kv_dim, max_seq_len]:
//   cache[layer][slot][d][pos] = src[b][d]
// One launch per (layer, K/V).
void kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                   const tensor::Tensor& kv_offsets, const tensor::Tensor& positions,
                   int32_t kv_dim, int32_t num_slots, int32_t max_seq_len, int32_t layer_idx,
                   void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream,
                           float rope_theta = 1000000.0f);

}  // namespace kernel
#endif 
