#ifndef SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_MHA_KERNEL_CUH
namespace kernel {
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                   CudaConfig* config);

// Batched decode MHA: one launch per layer for the whole batch.
//   positions   [batch] CUDA int32
//   kv_offsets  [batch] CUDA int32
//   query_batch [batch, dim] (dim = head_num * head_size)
//   score_batch [batch * head_num, max_seq_len] scratch
//   mha_out     [batch, dim]
//   key/value_cache [num_layers, num_slots, max_seq_len, kv_dim]
void mha_kernel_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                         int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                         int32_t head_size, const tensor::Tensor& positions,
                         const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                         tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                         const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                         CudaConfig* config);
}
#endif
