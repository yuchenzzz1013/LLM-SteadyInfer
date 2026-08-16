#ifndef SRC_SOURCE_OP_KERNELS_CUDA_PAGED_KERNELS_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_PAGED_KERNELS_CUH

#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

// Paged KV cache layout (vLLM-style):
//   [num_layers, num_blocks, block_size, kv_dim]
// element (layer, block, pos_in_block, d) at
//   layer * (num_blocks * block_size * kv_dim)
// + block * (block_size * kv_dim)
// + pos_in_block * kv_dim + d
// Position pos of sequence b maps to physical (block, offset) through the
// block table: block = block_table[b * table_stride + pos / block_size],
// offset = pos % block_size. The table is read with __ldg (read-only cache).

// Scatter one [kv_dim] row per batch element into the paged cache:
//   dst[layer][table[b][pos / block_size]][pos % block_size][d] = src[b][d]
// One block per batch row; positions are per-token (prefill rows and decode
// rows are both single-token rows after the scheduler flattens chunks).
void paged_kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                         const tensor::Tensor& block_table, const tensor::Tensor& positions,
                         int32_t kv_dim, int32_t num_blocks, int32_t block_size,
                         int32_t layer_idx, void* stream);

// Batched decode / chunked-prefill MHA over the paged cache (Flash Decoding):
// one split-KV launch pair per layer for the whole batch. Identical grid /
// thread mapping / partials format / GQA mapping as mha_kernel_cu_batch —
// only the cache addressing goes through block_table indirection — so A/B
// runs against the continuous layout are bit-identical in fp32.
//   positions    [batch] CUDA int32
//   block_table  [batch, table_stride] CUDA int32 (-1 = unused entry)
//   query_batch  [batch, dim] (dim = head_num * head_size)
//   score_batch  scratch: batch * head_num * flash_decoding_num_splits(
//                table_stride * block_size) * (head_size + 2) floats
//   mha_out      [batch, dim]
//   key/value_cache [num_layers, num_blocks, block_size, kv_dim]
void paged_attention_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& block_table, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              CudaConfig* config);

}  // namespace kernel
#endif
