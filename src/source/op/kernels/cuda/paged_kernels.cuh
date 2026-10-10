#ifndef SRC_SOURCE_OP_KERNELS_CUDA_PAGED_KERNELS_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_PAGED_KERNELS_CUH

#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

// Geometry the paged decode kernel (paged_attn_split_kernel_bf16) can serve:
// each lane owns 4 contiguous head dims and reads them as one 8-byte load, so
// head_size must be a multiple of 8; head_size <= 256 is covered in 128-dim
// chunks; the 4-position arithmetic group must not straddle a page (block_size
// a power of two, >= 4); and the cache needs a real block table. A geometry
// this rejects cannot run paged attention at all, so callers must size
// score_batch for the fp32 (acc | m | l) partials whenever it holds.
inline bool paged_decode_geometry_ok(int32_t head_size, int32_t table_stride, int32_t block_size) {
  return head_size >= 8 && head_size <= 256 && (head_size & 7) == 0 && table_stride > 0 &&
         block_size >= 4 && (block_size & (block_size - 1)) == 0;
}

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
// Raw bf16 element copy.
void paged_kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                         const tensor::Tensor& block_table, const tensor::Tensor& positions,
                         int32_t kv_dim, int32_t num_blocks, int32_t block_size,
                         int32_t layer_idx, void* stream);

// Batched decode / chunked-prefill MHA over the paged cache (Flash Decoding):
// one split-KV launch pair per layer for the whole batch. Identical grid /
// thread mapping / partials format / GQA mapping as mha_kernel_cu_batch —
// only the cache addressing goes through block_table indirection.
//   positions    [batch] CUDA int32
//   block_table  [batch, table_stride] CUDA int32 (-1 = unused entry)
//   query_batch  [batch, dim] (dim = head_num * head_size)
//   score_batch  scratch: batch * head_num * paged_decode_num_splits(
//                table_stride * block_size, head_num, kv_head_num) split
//                slots, each (head_size + 4) fp32 floats of (acc | m | l).
//                flash_decoding_partials_elements() (mha_kernel.cuh) returns
//                the element count in the model dtype — one fp32 row is two
//                bf16 elements. A buffer that is too small fails the call
//                instead of silently computing garbage.
//   mha_out      [batch, dim]
//   key/value_cache [num_layers, num_blocks, block_size, kv_dim]
void paged_attention_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& block_table, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              CudaConfig* config);

// ========== Mixed decode + prefill dispatch ==========
// One attention call for a batch whose first num_decode_rows rows are decode
// rows and whose remaining rows are prefill rows, grouped per sequence
// (seq_row_start[s] .. seq_row_start[s+1) = rows of prefill sequence s). The
// two row ranges run on the implementation that fits their load:
//   * decode rows   -> paged_attention_cu_batch (split-KV, one K/V read per
//                      row and kv head, shared by its whole q-head group);
//   * prefill rows  -> paged_prefill_attention_cu_batch (one KV read per
//                      (sequence, q-block) shared by all its rows and GQA
//                      heads).
// Both write disjoint row ranges of mha_out, so the split is invisible to the
// layers after attention. Table/positions/query/out all use the same row
// order as the caller's batch.
// seq_row_start == nullptr (or num_prefill_seqs == 0) means "no prefill rows":
// the whole batch is decode rows.
// Only the head_size == 128 geometry is split by row type (prefill_geometry_ok);
// other head sizes keep the whole batch on paged_attention_cu_batch, which
// handles every row.
void paged_attention_dispatch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, int32_t num_decode_rows,
                              const tensor::Tensor* seq_row_start, int32_t num_prefill_seqs,
                              const tensor::Tensor& positions, const tensor::Tensor& block_table,
                              const tensor::Tensor& query_batch, tensor::Tensor& score_batch,
                              const tensor::Tensor& mha_out, const tensor::Tensor& key_cache,
                              const tensor::Tensor& value_cache, CudaConfig* config);

}  // namespace kernel
#endif
