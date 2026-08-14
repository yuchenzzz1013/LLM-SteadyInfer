#ifndef SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
#include <algorithm>
namespace kernel {
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

// Batched decode / chunked-prefill MHA (Flash Decoding): one launch per layer
// for the whole batch. The KV range of each (batch, head) is split across
// num_splits blocks; partial (o, m, l) results are reduced by a second kernel.
//   positions   [batch] CUDA int32
//   kv_offsets  [batch] CUDA int32
//   query_batch [batch, dim] (dim = head_num * head_size)
//   score_batch scratch, must hold
//     batch * head_num * flash_decoding_num_splits(max_seq_len) * (head_size + 2)
//     floats for the partials (o | m | l) of every split.
//   mha_out     [batch, dim]
//   key/value_cache [num_layers, num_slots, kv_dim, max_seq_len] (head-dim
//   contiguous layout: cache[layer][slot][d][pos])
void mha_kernel_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                         int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                         int32_t head_size, const tensor::Tensor& positions,
                         const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                         tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                         const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                         CudaConfig* config);
}
#endif
