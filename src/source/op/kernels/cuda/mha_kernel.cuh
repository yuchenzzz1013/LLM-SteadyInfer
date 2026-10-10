#ifndef SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
#include <algorithm>
namespace kernel {
// CUDA MHA kernels operate on raw bfloat16 q / KV caches / output / partials
// (see mha_kernel_bf16.cu); internal score arithmetic stays in fp32.
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                   CudaConfig* config);

// Flash-Decoding split count derived from the KV capacity. Must match the
// scratch sizing used by the callers of mha_kernel_cu_batch.
inline int flash_decoding_num_splits(int32_t max_seq_len) {
  constexpr int DECODE_TILE = 128;
  constexpr int MAX_SPLITS = 8;
  int splits = (max_seq_len + DECODE_TILE - 1) / DECODE_TILE;
  return std::max(1, std::min(MAX_SPLITS, splits));
}

// Split count of the paged split-KV decode kernel. One warp per (row, kv head,
// split) walks that split for the whole q-head group of its kv head, so the
// capacity-derived flash_decoding_num_splits() is multiplied by the group size
// and the warp count comes out at batch * head_num *
// flash_decoding_num_splits(max_seq_len) — the parallelism of a per-q-head
// split, with each K/V row loaded once per warp. Callers of
// paged_attention_cu_batch must size score_batch for this many splits: the
// partials of one (row, head) pair are strided by exactly this count.
inline int paged_decode_num_splits(int32_t max_seq_len, int32_t head_num, int32_t kv_head_num) {
  const int32_t group = (head_num + kv_head_num - 1) / kv_head_num;
  return flash_decoding_num_splits(max_seq_len) * (group > 0 ? group : 1);
}

// Element count of the partials buffer (score_batch) one attention call needs,
// in the model dtype. Two layouts share that buffer:
//   * bf16 rows of (head_size + 2) elements, (o | m | l) — the continuous
//     layout (mha_kernel_cu_batch);
//   * fp32 rows of (head_size + 4) floats, (acc | m | l), written by the
//     paged split-KV decode kernel (paged_attn_split_kernel_bf16), whose
//     `num_splits` is paged_decode_num_splits(). One fp32 row is 2 *
//     (head_size + 4) bf16 elements.
// `fp32_split` picks the fp32 split-KV sizing; a caller that does not know
// which path a step will take may size for it, since the larger footprint
// still satisfies the bf16 one.
inline int64_t flash_decoding_partials_elements(int32_t batch, int32_t head_num,
                                                int32_t num_splits, int32_t head_size,
                                                bool fp32_split) {
  const int64_t rows = static_cast<int64_t>(batch) * head_num * num_splits;
  return fp32_split ? 2 * rows * (head_size + 4) : rows * (head_size + 2);
}

// Batched decode / chunked-prefill MHA (Flash Decoding): one launch per layer
// for the whole batch. The KV range of each (batch, head) is split across
// num_splits blocks; partial (o, m, l) results are reduced by a second kernel.
//   positions   [batch] CUDA int32
//   kv_offsets  [batch] CUDA int32
//   query_batch [batch, dim] (dim = head_num * head_size)
//   score_batch scratch, must hold
//     batch * head_num * flash_decoding_num_splits(max_seq_len) * (head_size + 2)
//     elements of the model dtype for the partials (o | m | l) of every split.
//   mha_out     [batch, dim]
//   key/value_cache [num_layers, num_slots, kv_dim, max_seq_len] (head-dim
//   contiguous layout: cache[layer][slot][d][pos])
// All q / KV caches / output are raw bfloat16 (see mha_kernel_bf16.cu).
void mha_kernel_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                         int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                         int32_t head_size, const tensor::Tensor& positions,
                         const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                         tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                         const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                         CudaConfig* config);
}
#endif
