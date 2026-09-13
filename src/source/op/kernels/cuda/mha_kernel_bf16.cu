// CUDA attention kernels (all BF16): continuous-layout Flash Decoding
// (mha_kernel_cu / mha_kernel_cu_batch), paged KV scatter and the paged
// attention family (paged_attention_dispatch / paged_attention_cu_batch /
// paged_prefill_attention_cu_batch).
//
// A step's batch is split by row type: the scheduler puts the decode rows
// first, and paged_attention_dispatch sends [0, num_decode_rows) to the
// warp-per-(row, head) decode kernel (paged_attn_warp2_kernel_bf16) and
// [num_decode_rows, batch) — the prefill rows, grouped per sequence by
// seq_row_start — to paged_prefill_kernel_bf16, which stages each KV tile into
// smem once per CTA instead of re-reading the prefix once per (row, head).
// Both write disjoint row ranges of mha_out, so a mixed step runs one forward
// pass with the weights read once. Geometry the split kernels do not serve
// (head_size != 128, unusual page sizes, continuous layout) keeps the whole
// batch on the split-partials flash decoding path below.
//
// All q/k/v/output caches are raw bfloat16 (CUDA __nv_bfloat16, bit-compatible
// with the uint16_t host representation). Internal score / softmax / output
// arithmetic runs in float: the operands are exact bf16 values (products of
// two bf16 numbers are exact in float), so this matches cuBLAS BF16 semantics
// (bf16 operands, fp32 accumulation) rather than FP32 kernels — activation and
// cache data never round-trips through FP32 storage.
//
// Partials buffers (o | m | l) are stored as bf16 elements per split slot;
// the slot stride (head_size + 2 elements) matches the fp32 layout, so the
// scratch tensors only need the same element counts in the model dtype.
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <vector>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include "paged_kernels.cuh"
namespace kernel {

// Flash-softmax helper: scores live in fp32 smem (computed from bf16 inputs);
// the flash statistic is never stored in bf16.
__device__ void flash_softmax_tile_bf16(float* p_smem, int tile_len, float& m, float& l) {
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;

  float local_max = -FLT_MAX;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    local_max = fmaxf(local_max, p_smem[t]);
  }
  m = BlockReduce(temp).Reduce(local_max, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_val = m;
  }
  __syncthreads();
  m = shared_val;

  float local_l = 0.f;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    const float p = expf(p_smem[t] - m);
    p_smem[t] = p;
    local_l += p;
  }
  l = BlockReduce(temp).Sum(local_l);
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_val = l;
  }
  __syncthreads();
  l = shared_val;
}

// ========== Legacy single-sequence attention kernel ==========
// bf16 q / kv-cache rows / output; heads are read as bf16 and converted once
// into the fp32 smem query, KV elements convert on the fly per use.
constexpr int TILE_BF16 = 128;

__global__ void multi_head_attention_kernel_bf16(
    int32_t pos, int32_t seq_len, const __nv_bfloat16* query, const __nv_bfloat16* output,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache, int32_t kv_dim,
    int32_t kv_head_num, int32_t head_num, int32_t head_size, int64_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_mem[];
  float* s_query = s_mem;                 // [head_size]
  float* s_p = s_mem + head_size;         // [TILE_BF16]

  const float scale = 1.f / sqrtf(float(head_size));
  const __nv_bfloat16* query_head = query + head * head_size;
  const int seq = pos + 1;
  // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits.
  const int head_offset = (head * kv_head_num / head_num) * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = __bfloat162float(query_head[i]);
  }
  __syncthreads();

  float m_i = -FLT_MAX;
  float l_i = 0.f;
  // Per-thread accumulator for out[head_size] (thread d owns dimension d).
  float acc = 0.f;

  for (int tile_start = 0; tile_start < seq; tile_start += TILE_BF16) {
    const int tile_len = min(TILE_BF16, seq - tile_start);

    // Scores: s_t = sum_d q[d] * K[t][head_offset + d], strided over threads.
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const __nv_bfloat16* k_row =
          key_cache + layer_offset + static_cast<int64_t>(tile_start + t) * kv_dim + head_offset;
      float s_t = 0.f;
      for (int d = 0; d < head_size; ++d) {
        s_t += __bfloat162float(k_row[d]) * s_query[d];
      }
      s_p[t] = s_t * scale;
    }

    float m_j, l_j;
    flash_softmax_tile_bf16(s_p, tile_len, m_j, l_j);
    // First tile: m_i == -inf, so alpha must be 1 (not exp(+inf) = inf).
    const float alpha = (m_i == -FLT_MAX) ? 1.0f : expf(m_j - m_i);
    l_i = alpha * l_i + l_j;
    m_i = m_j;

    // acc = alpha * acc + sum_t p_t * V[t][head_offset + d]
    acc *= alpha;
    const int d = threadIdx.x;
    if (d < head_size) {
      const __nv_bfloat16* v_col =
          value_cache + layer_offset + static_cast<int64_t>(tile_start) * kv_dim + head_offset + d;
      for (int tt = 0; tt < tile_len; ++tt) {
        acc += s_p[tt] * __bfloat162float(v_col[static_cast<int64_t>(tt) * kv_dim]);
      }
    }
  }

  if (threadIdx.x < head_size) {
    const_cast<__nv_bfloat16*>(output)[head * head_size + threadIdx.x] =
        __float2bfloat16(acc / l_i);
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                        int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                        const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                        const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                        const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                        CudaConfig* config) {
  UNUSED(device_type);
  UNUSED(score_tensor);
  CHECK_LE(head_size, 256) << "Flash kernel requires head_size <= 256 (one output dim per thread).";
  CHECK(config != nullptr);
  // int64: large external caches (num_slots * max_seq_len) can overflow int32.
  int64_t layer_offset = static_cast<int64_t>(layer_index) * seq_len * kv_dim;
  const __nv_bfloat16* query = query_tensor.ptr<__nv_bfloat16>();
  const __nv_bfloat16* output = mha_out.ptr<__nv_bfloat16>();
  const __nv_bfloat16* key_cache = key_cache_tensor.ptr<__nv_bfloat16>();
  const __nv_bfloat16* value_cache = value_cache_tensor.ptr<__nv_bfloat16>();

  cudaStream_t stream = config->stream;
  int smem_bytes = (head_size + TILE_BF16) * sizeof(float);
  multi_head_attention_kernel_bf16<<<head_num, 256, smem_bytes, stream>>>(
      pos, seq_len, query, output, key_cache, value_cache, kv_dim, kv_head_num, head_num,
      head_size, layer_offset);
}

// ========== Flash Decoding (continuous KV layout) ==========
// bf16 query / caches / partials / output. Partials slots keep (head_size + 2)
// bf16 elements: [o | m | l], o rounded to bf16 per split, m/l kept in bf16
// (their bf16 precision is ample for the recombination weights).
__global__ void flash_decoding_kernel_bf16(const int32_t* positions, const int32_t* kv_offsets,
                                           int32_t num_slots, int32_t max_seq_len,
                                           int32_t layer_idx, int32_t dim,
                                           const __nv_bfloat16* query, __nv_bfloat16* partials,
                                           const __nv_bfloat16* key_cache,
                                           const __nv_bfloat16* value_cache, int32_t kv_dim,
                                           int32_t kv_head_num, int32_t head_num, int32_t head_size,
                                           int32_t num_splits) {
  const int pair = blockIdx.x / num_splits;  // pair = b * head_num + head
  const int split = blockIdx.x % num_splits;
  const int b = pair / head_num;
  const int head = pair % head_num;
  const int pos = positions[b];
  const int seq = pos + 1;

  const int tile_start = split * seq / num_splits;
  const int tile_end = (split + 1) * seq / num_splits;
  // The combine kernel reads every split's partial slot unconditionally, so
  // empty splits must record a neutral (o=0, m=-inf, l=0) partial.
  __nv_bfloat16* partial =
      partials + (static_cast<int64_t>(pair) * num_splits + split) * (head_size + 2);
  if (tile_start >= tile_end) {
    for (int dd = threadIdx.x; dd < head_size; dd += blockDim.x) {
      partial[dd] = __float2bfloat16(0.f);
    }
    if (threadIdx.x == 0) {
      partial[head_size] = __float2bfloat16(-FLT_MAX);  // -inf in bf16
      partial[head_size + 1] = __float2bfloat16(0.f);
    }
    return;  // empty split for short sequences
  }
  const int tile_len = tile_end - tile_start;

  extern __shared__ float s_mem[];
  float* s_query = s_mem;          // [head_size]
  float* s_p = s_mem + head_size;  // [max_tile] raw scores -> probs

  const float scale = 1.f / sqrtf(float(head_size));
  const int head_offset = (head * kv_head_num / head_num) * head_size;
  const int slot = kv_offsets[b];
  const int64_t cache_base =
      static_cast<int64_t>(layer_idx) * num_slots * kv_dim * max_seq_len +
      static_cast<int64_t>(slot) * kv_dim * max_seq_len + tile_start;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = __bfloat162float(q_head[i]);
  }
  __syncthreads();

  // Scores over this split's KV tile: s_t = sum_d q[d] * K[hd + d][t].
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    float s_t = 0.f;
    for (int d = 0; d < head_size; ++d) {
      s_t += s_query[d] *
             __bfloat162float(key_cache[cache_base +
                                       static_cast<int64_t>(head_offset + d) * max_seq_len + t]);
    }
    s_p[t] = s_t * scale;
  }

  float m_i, l_i;
  flash_softmax_tile_bf16(s_p, tile_len, m_i, l_i);

  // Partial output: acc_d = sum_t p_t * V[head_offset + d][t].
  float acc = 0.f;
  const int d = threadIdx.x;
  if (d < head_size) {
    const __nv_bfloat16* v_dim =
        value_cache + cache_base + static_cast<int64_t>(head_offset + d) * max_seq_len;
    for (int tt = 0; tt < tile_len; ++tt) {
      acc += s_p[tt] * __bfloat162float(v_dim[tt]);
    }
  }

  // partials[pair * num_splits + split] -> [o (head_size) | m | l]
  if (d < head_size) {
    partial[d] = __float2bfloat16(acc);
  }
  if (threadIdx.x == 0) {
    partial[head_size] = __float2bfloat16(m_i);
    partial[head_size + 1] = __float2bfloat16(l_i);
  }
}

__global__ void flash_decoding_combine_kernel_bf16(const __nv_bfloat16* partials,
                                                   __nv_bfloat16* output, int32_t dim,
                                                   int32_t head_num, int32_t head_size,
                                                   int32_t num_splits, int32_t batch) {
  const int pair = blockIdx.x;
  const int b = pair / head_num;
  const int head = pair % head_num;
  if (b >= batch) {
    return;
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float s_global_m;

  const __nv_bfloat16* partial =
      partials + static_cast<int64_t>(pair) * num_splits * (head_size + 2);

  // Global max over splits.
  float local_m = (threadIdx.x < num_splits)
                      ? __bfloat162float(partial[threadIdx.x * (head_size + 2) + head_size])
                      : -FLT_MAX;
  float global_m = BlockReduce(temp).Reduce(local_m, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) {
    s_global_m = global_m;
  }
  __syncthreads();
  global_m = s_global_m;

  // o = sum_s o_s * exp(m_s - m) / sum_s l_s * exp(m_s - m)
  float o = 0.f;
  float l = 0.f;
  const int d = threadIdx.x;
  for (int s = 0; s < num_splits; ++s) {
    const __nv_bfloat16* ps = partial + static_cast<int64_t>(s) * (head_size + 2);
    const float w = expf(__bfloat162float(ps[head_size]) - global_m);
    l += w * __bfloat162float(ps[head_size + 1]);
    if (d < head_size) {
      o += w * __bfloat162float(ps[d]);
    }
  }
  if (d < head_size) {
    output[static_cast<int64_t>(b) * dim + head * head_size + d] = __float2bfloat16(o / l);
  }
}

void mha_kernel_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                              int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              CudaConfig* config) {
  int32_t batch = query_batch.get_dim(0);
  int32_t dim = query_batch.get_dim(1);
  int32_t num_splits = flash_decoding_num_splits(max_seq_len);
  CHECK_LE(head_size, 256)
      << "Flash Decoding requires head_size <= 256 (one output dim per thread).";
  CHECK(config != nullptr);

  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  __nv_bfloat16* partials = const_cast<__nv_bfloat16*>(score_batch.ptr<__nv_bfloat16>());
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  const __nv_bfloat16* kcache = key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = value_cache.ptr<__nv_bfloat16>();

  cudaStream_t stream = config->stream;
  // s_p must hold the largest tile any split can own.
  int max_tile = (max_seq_len + num_splits - 1) / num_splits;
  int smem_bytes = (head_size + max_tile) * sizeof(float);
  int total_blocks = batch * head_num * num_splits;
  flash_decoding_kernel_bf16<<<total_blocks, 256, smem_bytes, stream>>>(
      positions.ptr<int32_t>(), kv_offsets.ptr<int32_t>(), num_slots, max_seq_len, layer_idx, dim,
      query, partials, kcache, vcache, kv_dim, kv_head_num, head_num, head_size, num_splits);
  flash_decoding_combine_kernel_bf16<<<batch * head_num, 256, 0, stream>>>(
      partials, output, dim, head_num, head_size, num_splits, batch);
}

// ========== Paged KV scatter ==========
// Raw bf16 element copy through the block table (2-byte elements).
__global__ void paged_kv_scatter_kernel_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                             const int32_t* block_table, int32_t table_stride,
                                             const int32_t* positions, int32_t kv_dim,
                                             int32_t num_blocks, int32_t block_size,
                                             int32_t layer_idx) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int pos = positions[b];
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int block_id = __ldg(table_row + (pos / block_size));
  // Offset within the page. NOT `pos - block_id * block_size`: block_id is the
  // *physical* page from the table, so that expression only equals pos %
  // block_size when the table happens to be the identity, and otherwise
  // cancels the page term out of the address entirely.
  const int off = pos % block_size;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const int64_t base = layer_base + static_cast<int64_t>(block_id) * (block_size * kv_dim) +
                       static_cast<int64_t>(off) * kv_dim;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(b) * kv_dim;
  for (int d = tid; d < kv_dim; d += blockDim.x) {
    dst[base + d] = src_row[d];
  }
}

void paged_kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                              const tensor::Tensor& block_table, const tensor::Tensor& positions,
                              int32_t kv_dim, int32_t num_blocks, int32_t block_size,
                              int32_t layer_idx, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  int32_t batch = src.get_dim(0);
  __nv_bfloat16* dst = const_cast<__nv_bfloat16*>(dst_cache.ptr<__nv_bfloat16>());
  const int32_t table_stride = block_table.get_dim(1);
  if (stream_) {
    paged_kv_scatter_kernel_bf16<<<batch, 256, 0, stream_>>>(
        src.ptr<__nv_bfloat16>(), dst, block_table.ptr<int32_t>(), table_stride,
        positions.ptr<int32_t>(), kv_dim, num_blocks, block_size, layer_idx);
  } else {
    paged_kv_scatter_kernel_bf16<<<batch, 256>>>(
        src.ptr<__nv_bfloat16>(), dst, block_table.ptr<int32_t>(), table_stride,
        positions.ptr<int32_t>(), kv_dim, num_blocks, block_size, layer_idx);
  }
}

// ========== Paged Flash Decoding ==========
// bf16 query / caches / partials / output with block-table addressing.
__global__ void paged_flash_decoding_kernel_bf16(const int32_t* positions,
                                                 const int32_t* block_table, int32_t table_stride,
                                                 int32_t num_blocks, int32_t block_size,
                                                 int32_t layer_idx, int32_t dim,
                                                 const __nv_bfloat16* query,
                                                 __nv_bfloat16* partials,
                                                 const __nv_bfloat16* key_cache,
                                                 const __nv_bfloat16* value_cache, int32_t kv_dim,
                                                 int32_t kv_head_num, int32_t head_num,
                                                 int32_t head_size, int32_t num_splits) {
  // 1-D grid of batch * head_num * num_splits blocks.
  const int pair = blockIdx.x / num_splits;  // pair = b * head_num + head
  const int split = blockIdx.x % num_splits;
  const int b = pair / head_num;
  const int head = pair % head_num;
  const int pos = positions[b];
  const int seq = pos + 1;

  const int tile_start = split * seq / num_splits;
  const int tile_end = (split + 1) * seq / num_splits;
  // Empty splits record a neutral (o=0, m=-inf, l=0) partial.
  __nv_bfloat16* partial =
      partials + (static_cast<int64_t>(pair) * num_splits + split) * (head_size + 2);
  if (tile_start >= tile_end) {
    for (int dd = threadIdx.x; dd < head_size; dd += blockDim.x) {
      partial[dd] = __float2bfloat16(0.f);
    }
    if (threadIdx.x == 0) {
      partial[head_size] = __float2bfloat16(-FLT_MAX);  // -inf in bf16
      partial[head_size + 1] = __float2bfloat16(0.f);
    }
    return;  // empty split for short sequences
  }
  const int tile_len = tile_end - tile_start;

  extern __shared__ float s_mem[];
  float* s_query = s_mem;          // [head_size]
  float* s_p = s_mem + head_size;  // [max_tile] raw scores -> probs

  const float scale = 1.f / sqrtf(float(head_size));
  const int head_offset = (head * kv_head_num / head_num) * head_size;
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = __bfloat162float(q_head[i]);
  }
  __syncthreads();

  // Scores over this split's KV tile, strided over threads; block_id advances
  // by blockDim / block_size per step.
  const int off_t = (tile_start + threadIdx.x) % block_size;
  int pos_t = tile_start + threadIdx.x;
  int block_t = -1;
  if (threadIdx.x < tile_len) {
    block_t = __ldg(table_row + (pos_t / block_size));
  }
  const __nv_bfloat16* k_base =
      key_cache + layer_base + static_cast<int64_t>(off_t) * kv_dim + head_offset;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    const __nv_bfloat16* k_row = k_base + static_cast<int64_t>(block_t) * (block_size * kv_dim);
    float s_t = 0.f;
    for (int d = 0; d < head_size; ++d) {
      s_t += s_query[d] * __bfloat162float(k_row[d]);
    }
    s_p[t] = s_t * scale;
    pos_t += blockDim.x;
    if (pos_t < tile_end) {
      block_t = __ldg(table_row + (pos_t / block_size));
    }
  }

  float m_i, l_i;
  flash_softmax_tile_bf16(s_p, tile_len, m_i, l_i);

  // Partial output: acc_d = sum_t p_t * V[pos][head_offset + d].
  float acc = 0.f;
  const int d = threadIdx.x;
  if (d < head_size) {
    int pos_v = tile_start;
    int block_v = __ldg(table_row + (pos_v / block_size));
    int off_v = pos_v % block_size;  // counter, wraps to the next page
    const __nv_bfloat16* v_base = value_cache + layer_base + head_offset + d;
    for (int tt = 0; tt < tile_len; ++tt) {
      acc += s_p[tt] * __bfloat162float(v_base[static_cast<int64_t>(block_v) *
                                                   (block_size * kv_dim) +
                                               static_cast<int64_t>(off_v) * kv_dim]);
      if (++off_v == block_size) {
        off_v = 0;
        ++pos_v;
        block_v = __ldg(table_row + (pos_v / block_size));
      }
    }
  }

  // partials[pair * num_splits + split] -> [o (head_size) | m | l]
  if (d < head_size) {
    partial[d] = __float2bfloat16(acc);
  }
  if (threadIdx.x == 0) {
    partial[head_size] = __float2bfloat16(m_i);
    partial[head_size + 1] = __float2bfloat16(l_i);
  }
}

// Combine pass — same recombination as flash_decoding_combine_kernel,
// operating on the bf16 partials only.
__global__ void paged_flash_decoding_combine_kernel_bf16(const __nv_bfloat16* partials,
                                                         __nv_bfloat16* output, int32_t dim,
                                                         int32_t head_num, int32_t head_size,
                                                         int32_t num_splits, int32_t batch) {
  const int pair = blockIdx.x;
  const int b = pair / head_num;
  const int head = pair % head_num;
  if (b >= batch) {
    return;
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float s_global_m;

  const __nv_bfloat16* partial =
      partials + static_cast<int64_t>(pair) * num_splits * (head_size + 2);

  // Global max over splits.
  float local_m = (threadIdx.x < num_splits)
                      ? __bfloat162float(partial[threadIdx.x * (head_size + 2) + head_size])
                      : -FLT_MAX;
  float global_m = BlockReduce(temp).Reduce(local_m, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) {
    s_global_m = global_m;
  }
  __syncthreads();
  global_m = s_global_m;

  float o = 0.f;
  float l = 0.f;
  const int d = threadIdx.x;
  for (int s = 0; s < num_splits; ++s) {
    const __nv_bfloat16* ps = partial + static_cast<int64_t>(s) * (head_size + 2);
    const float w = expf(__bfloat162float(ps[head_size]) - global_m);
    l += w * __bfloat162float(ps[head_size + 1]);
    if (d < head_size) {
      o += w * __bfloat162float(ps[d]);
    }
  }
  if (d < head_size) {
    output[static_cast<int64_t>(b) * dim + head * head_size + d] = __float2bfloat16(o / l);
  }
}

// ========== Paged decode attention, smem-tiled (not wired up) ==========
// Unreferenced experiment: the dispatcher below only ever launches warp2 or
// the prefill kernel, never this one. Kept for reference.
// One 128-thread block per (batch row, q head). The block walks the row's
// whole KV prefix in 64-position chunks; each chunk's K/V head-slice is
// staged in smem with coalesced bf16 loads (head-size row == 256B
// contiguous) instead of the split kernels' per-(position, dim) 2-byte
// gathers, and softmax + weighted V accumulation run in one pass, so no
// split partials are produced and no combine launch is needed.
// smem layout: s_k / s_v [64][kPagedTileStride] bf16 tiles (rows padded to
// 130 so a column walk stays bank-conflict-free), s_q (q * scale), s_p
// (chunk scores -> probs), s_base (per-position page base, shared by K/V).
constexpr int kPagedTilePos = 64;
constexpr int kPagedTileStride = 130;  // head_size (<= 128) + 2 pad elems

// ========== Paged decode attention, vectorized + grouped softmax ==========
// One warp (32 lanes) per (batch row, q head), 8 warps per 256-thread block.
// This is the decode kernel of record: softmax is warp-shuffle online (no
// __syncthreads anywhere), and each per-position critical path is short:
//   * each lane owns 4 CONTIGUOUS head dims (lane*4 .. lane*4+3) instead of
//     a lane-strided c*32+lane, so a whole K or V row costs ONE 8-byte load
//     per lane (2 loads per position instead of 8);
//   * the online softmax update runs once per GROUP of 4 positions: the four
//     score reductions are independent and pipeline together, and the
//     expf/rescale chain that serializes the walk is 4x less frequent;
//   * the block-table lookup is hoisted to the group: every supported block
//     size is a multiple of 4, so a 4-aligned group never crosses a page;
//   * the page index uses the block shift (block_size is a power of two),
//     not a hardcoded ">> 4".
// The per-position arithmetic here (lane-owned 4 dims, 4-position rescale
// groups, 5-shuffle warp score reduction, scalar tail for < 4 positions) is
// the reference that paged_prefill_kernel_bf16 below mirrors op for op, so a
// prefill row comes out bit-identical whichever of the two kernels runs it.
__global__ void paged_attn_warp2_kernel_bf16(
    const int32_t* positions, const int32_t* block_table, int32_t table_stride,
    int32_t num_blocks, int32_t block_size, int32_t block_shift, int32_t layer_idx,
    int32_t dim, const __nv_bfloat16* query, __nv_bfloat16* output,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache, int32_t kv_dim,
    int32_t kv_head_num, int32_t head_num, int32_t head_size) {
  constexpr int kVec = 4;  // head dims per lane; head_size == 128
  constexpr int kGroup = 4;
  const int warp_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int b = warp_id / head_num;
  const int head = warp_id - b * head_num;
  const int pos = positions[b];
  const float scale = 1.f / sqrtf(static_cast<float>(head_size));
  const int head_offset = (head * kv_head_num / head_num) * head_size;
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;
  const int64_t out_base = static_cast<int64_t>(b) * dim + head * head_size;
  const int d0 = lane * kVec;

  // Fold the softmax scale into q once; the dot then needs no extra multiply.
  float q[kVec];
  {
    const uint2 raw = *reinterpret_cast<const uint2*>(q_head + d0);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    q[0] = __bfloat162float(h[0].x) * scale;
    q[1] = __bfloat162float(h[0].y) * scale;
    q[2] = __bfloat162float(h[1].x) * scale;
    q[3] = __bfloat162float(h[1].y) * scale;
  }

  float m_i = -FLT_MAX;  // online flash stats (warp-uniform)
  float l_i = 0.f;
  float acc[kVec] = {0.f, 0.f, 0.f, 0.f};

  const int seq = pos + 1;
  int64_t row[kGroup];
  int gpos = 0;
  for (; gpos + kGroup <= seq; gpos += kGroup) {
    // kGroup divides block_size (both are powers of two, block_size >= 4 here)
    // and gpos is a multiple of kGroup, so a kGroup-wide run never straddles a
    // page: one page lookup per group, and the positions sit at a fixed kv_dim
    // stride inside the page.
    const int32_t pg = __ldg(table_row + (gpos >> block_shift));
    const int32_t off = gpos & (block_size - 1);
    const int64_t row0 = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                         head_offset + static_cast<int64_t>(off) * kv_dim + d0;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      row[j] = row0 + static_cast<int64_t>(j) * kv_dim;
    }
    float s[kGroup];
    float v_reg[kGroup][kVec];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      const uint2 kraw = *reinterpret_cast<const uint2*>(key_cache + row[j]);
      const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kraw);
      s[j] = q[0] * __bfloat162float(kh[0].x) + q[1] * __bfloat162float(kh[0].y) +
             q[2] * __bfloat162float(kh[1].x) + q[3] * __bfloat162float(kh[1].y);
      const uint2 vraw = *reinterpret_cast<const uint2*>(value_cache + row[j]);
      const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vraw);
      v_reg[j][0] = __bfloat162float(vh[0].x);
      v_reg[j][1] = __bfloat162float(vh[0].y);
      v_reg[j][2] = __bfloat162float(vh[1].x);
      v_reg[j][3] = __bfloat162float(vh[1].y);
    }
    // Independent across j, so the four reductions overlap.
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        s[j] += __shfl_xor_sync(0xffffffffu, s[j], off);
      }
    }
    float m_new = m_i;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      m_new = fmaxf(m_new, s[j]);
    }
    const float alpha = __expf(m_i - m_new);
    float l_new = l_i * alpha;
    float p[kGroup];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
      p[j] = __expf(s[j] - m_new);
      l_new += p[j];
    }
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
      float a = acc[c] * alpha;
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        a += p[j] * v_reg[j][c];
      }
      acc[c] = a;
    }
    m_i = m_new;
    l_i = l_new;
  }
  for (; gpos <= pos; ++gpos) {
    const int32_t pg = __ldg(table_row + (gpos >> block_shift));
    const int64_t row = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                        head_offset +
                        static_cast<int64_t>(gpos & (block_size - 1)) * kv_dim + d0;
    const uint2 kraw = *reinterpret_cast<const uint2*>(key_cache + row);
    const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kraw);
    float s = q[0] * __bfloat162float(kh[0].x) + q[1] * __bfloat162float(kh[0].y) +
              q[2] * __bfloat162float(kh[1].x) + q[3] * __bfloat162float(kh[1].y);
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, off);
    }
    const float m_new = fmaxf(m_i, s);
    const float alpha = __expf(m_i - m_new);
    const float p = __expf(s - m_new);
    l_i = l_i * alpha + p;
    const uint2 vraw = *reinterpret_cast<const uint2*>(value_cache + row);
    const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vraw);
    acc[0] = acc[0] * alpha + p * __bfloat162float(vh[0].x);
    acc[1] = acc[1] * alpha + p * __bfloat162float(vh[0].y);
    acc[2] = acc[2] * alpha + p * __bfloat162float(vh[1].x);
    acc[3] = acc[3] * alpha + p * __bfloat162float(vh[1].y);
    m_i = m_new;
  }

  const float inv_l = 1.f / l_i;
  __nv_bfloat162 o[2];
  o[0] = __floats2bfloat162_rn(acc[0] * inv_l, acc[1] * inv_l);
  o[1] = __floats2bfloat162_rn(acc[2] * inv_l, acc[3] * inv_l);
  *reinterpret_cast<uint2*>(output + out_base + d0) = *reinterpret_cast<const uint2*>(o);
}

// ========== Paged decode attention, KV-group shared smem (not wired up) ===
// Unreferenced experiment: the dispatcher below only ever launches warp2 or
// the prefill kernel, never this one. Kept for reference.
// One 128-thread block per (batch row, kv head group) with G=4 q heads (the
// engine's GQA ratio). The 4 warps handle one q head each; every K/V row of
// the group is staged into smem ONCE per page instead of being re-read by
// each head from global. At bs64 the per-layer KV working set (~105MB)
// thrashes L2, so the warp kernel's 4x redundant reads were DRAM-bound; this
// kernel reads each unique byte exactly once.
__global__ void paged_attn_group_kernel_bf16(
    const int32_t* positions, const int32_t* block_table, int32_t table_stride,
    int32_t num_blocks, int32_t block_size, int32_t layer_idx, int32_t dim,
    const __nv_bfloat16* query, __nv_bfloat16* output, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, int32_t kv_dim, int32_t kv_head_num, int32_t head_num,
    int32_t head_size) {
  constexpr int kColsPerLane = 4;      // head_size == 128
  constexpr int kMaxPage = 16;         // block_size <= 16
  const int G = head_num / kv_head_num;  // q heads sharing one kv head
  __shared__ __nv_bfloat16 s_k[kMaxPage][kPagedTileStride];
  __shared__ __nv_bfloat16 s_v[kMaxPage][kPagedTileStride];

  const int b = blockIdx.x / kv_head_num;
  const int g = blockIdx.x - b * kv_head_num;  // kv head of this group
  const int w = threadIdx.x >> 5;              // q head within group
  const int lane = threadIdx.x & 31;
  const int tid = threadIdx.x;
  const int head = g * G + w;
  const int pos = positions[b];
  const int seq = pos + 1;
  const float scale = 1.f / sqrtf(static_cast<float>(head_size));
  const int head_offset = g * head_size;
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;
  const int64_t out_base = static_cast<int64_t>(b) * dim + head * head_size;

  float q[kColsPerLane];
#pragma unroll
  for (int c = 0; c < kColsPerLane; ++c) {
    q[c] = __bfloat162float(q_head[c * 32 + lane]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.f;
  float acc[kColsPerLane] = {0.f, 0.f, 0.f, 0.f};

  const int n_pages = (seq + block_size - 1) / block_size;
  const int32_t* page_ids = table_row;
  for (int p = 0; p < n_pages; ++p) {
    const int plen = min(block_size, seq - p * block_size);
    const int32_t pg = __ldg(page_ids + p);
    const int64_t page_base = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                              head_offset;
    // Cooperative staged load of the group's K and V head slices: 256B
    // contiguous per (row), the group's cols are a prefix of each kv_dim row.
    for (int i = tid; i < plen * head_size; i += blockDim.x) {
      const int s = i / head_size;
      const int d = i - s * head_size;
      s_k[s][d] = key_cache[page_base + static_cast<int64_t>(s) * kv_dim + d];
    }
    for (int i = tid; i < plen * head_size; i += blockDim.x) {
      const int s = i / head_size;
      const int d = i - s * head_size;
      s_v[s][d] = value_cache[page_base + static_cast<int64_t>(s) * kv_dim + d];
    }
    __syncthreads();
    // Warp w: online flash update over this page's rows for q head g*G+w.
    for (int r = 0; r < plen; ++r) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < kColsPerLane; ++c) {
        s += q[c] * __bfloat162float(s_k[r][c * 32 + lane]);
      }
      s *= scale;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, off);
      }
      const float m_new = fmaxf(m_i, s);
      const float alpha = __expf(m_i - m_new);
      const float p = __expf(s - m_new);
      l_i = l_i * alpha + p;
#pragma unroll
      for (int c = 0; c < kColsPerLane; ++c) {
        acc[c] = acc[c] * alpha + p * __bfloat162float(s_v[r][c * 32 + lane]);
      }
      m_i = m_new;
    }
    __syncthreads();  // all warps done reading the staged page
  }

  const float inv_l = 1.f / l_i;
#pragma unroll
  for (int c = 0; c < kColsPerLane; ++c) {
    output[out_base + c * 32 + lane] = __float2bfloat16(acc[c] * inv_l);
  }
}

__global__ void paged_attn_decode_tiled_kernel_bf16(
    const int32_t* positions, const int32_t* block_table, int32_t table_stride,
    int32_t num_blocks, int32_t block_size, int32_t layer_idx, int32_t dim,
    const __nv_bfloat16* query, __nv_bfloat16* output, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, int32_t kv_dim, int32_t kv_head_num, int32_t head_num,
    int32_t head_size) {
  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ __nv_bfloat16 s_k[kPagedTilePos][kPagedTileStride];
  __shared__ __nv_bfloat16 s_v[kPagedTilePos][kPagedTileStride];
  __shared__ float s_q[128];
  __shared__ float s_p[kPagedTilePos];
  __shared__ int32_t s_base[kPagedTilePos];
  __shared__ union {
    BlockReduce::TempStorage temp;
    float bcast;
  } red;

  const int tid = threadIdx.x;
  const int b = blockIdx.x / head_num;
  const int head = blockIdx.x - b * head_num;
  const int pos = positions[b];
  const int seq = pos + 1;
  const float scale = 1.f / sqrtf(static_cast<float>(head_size));
  // GQA KV-head mapping (same as the split kernels).
  const int head_offset = (head * kv_head_num / head_num) * head_size;
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  // Query slice (scaled) in fp32 smem, loaded once per block.
  for (int d = tid; d < head_size; d += 128) {
    s_q[d] = __bfloat162float(q_head[d]) * scale;
  }

  float m_i = -FLT_MAX;  // running flash stats, replicated per thread
  float l_i = 0.f;
  float acc = 0.f;  // accumulator of output dim tid (thread = d)

  for (int c0 = 0; c0 < seq; c0 += kPagedTilePos) {
    const int tile_len = min(kPagedTilePos, seq - c0);

    // Resolve the page base of each chunk position once (shared by K and V).
    if (tid < tile_len) {
      const int32_t gpos = c0 + tid;
      const int32_t block_id = __ldg(table_row + gpos / block_size);
      const int32_t off = gpos % block_size;
      s_base[tid] = block_id * (block_size * kv_dim) + off * kv_dim + head_offset;
    }
    __syncthreads();

    // Stage the K chunk: coalesced bf16 row copies (row p is contiguous).
    for (int i = tid; i < tile_len * head_size; i += 128) {
      const int p = i / head_size;
      const int d = i - p * head_size;
      s_k[p][d] = key_cache[layer_base + s_base[p] + d];
    }
    __syncthreads();

    // Scores: thread p owns the dot of q with K row p (4-way ILP).
    if (tid < tile_len) {
      const int p = tid;
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      for (int d = 0; d < head_size; d += 4) {
        s0 += s_q[d] * __bfloat162float(s_k[p][d]);
        s1 += s_q[d + 1] * __bfloat162float(s_k[p][d + 1]);
        s2 += s_q[d + 2] * __bfloat162float(s_k[p][d + 2]);
        s3 += s_q[d + 3] * __bfloat162float(s_k[p][d + 3]);
      }
      s_p[p] = (s0 + s1) + (s2 + s3);
    }
    __syncthreads();

    // Online flash update: chunk max m_j, then probs and l_j.
    float local_m = -FLT_MAX;
    if (tid < tile_len) {
      local_m = s_p[tid];
    }
    float m_j = BlockReduce(red.temp).Reduce(local_m, cub::Max());
    __syncthreads();
    if (tid == 0) {
      red.bcast = m_j;
    }
    __syncthreads();
    m_j = red.bcast;
    const float m_old = m_i;
    m_i = fmaxf(m_i, m_j);
    // First chunk (m_old == -inf): exp(-inf - m) -> 0, so acc / l stay 0.
    const float alpha = (m_old == m_i) ? 1.f : __expf(m_old - m_i);
    acc *= alpha;
    l_i *= alpha;

    float local_l = 0.f;
    if (tid < tile_len) {
      const float pr = __expf(s_p[tid] - m_i);
      s_p[tid] = pr;
      local_l = pr;
    }
    __syncthreads();
    float l_j = BlockReduce(red.temp).Sum(local_l);
    __syncthreads();
    if (tid == 0) {
      red.bcast = l_j;
    }
    __syncthreads();
    l_i += red.bcast;

    // Stage the V chunk (same page bases as K).
    for (int i = tid; i < tile_len * head_size; i += 128) {
      const int p = i / head_size;
      const int d = i - p * head_size;
      s_v[p][d] = value_cache[layer_base + s_base[p] + d];
    }
    __syncthreads();

    // Weighted V accumulation: thread d owns output dim d; probs s_p[p] are
    // read at the same address by every lane (broadcast).
    if (tid < head_size) {
      const int d = tid;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      int p = 0;
      for (; p + 4 <= tile_len; p += 4) {
        a0 += s_p[p] * __bfloat162float(s_v[p][d]);
        a1 += s_p[p + 1] * __bfloat162float(s_v[p + 1][d]);
        a2 += s_p[p + 2] * __bfloat162float(s_v[p + 2][d]);
        a3 += s_p[p + 3] * __bfloat162float(s_v[p + 3][d]);
      }
      for (; p < tile_len; ++p) {
        a0 += s_p[p] * __bfloat162float(s_v[p][d]);
      }
      acc += (a0 + a1) + (a2 + a3);
    }
  }

  if (tid < head_size) {
    output[static_cast<int64_t>(b) * dim + head * head_size + tid] =
        __float2bfloat16(acc / l_i);
  }
}

// Paged kernels map a token position to its page with `pos >> block_shift`;
// the block_size CHECK in paged_attention_cu_batch already guarantees a power
// of two, so a shift is exact (and cheaper than the divide).
static int block_shift_for(int32_t block_size) {
  int shift = 0;
  while ((1 << shift) < block_size) ++shift;
  return shift;
}

void paged_attention_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                                   int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                                   int32_t head_size, const tensor::Tensor& positions,
                                   const tensor::Tensor& block_table,
                                   const tensor::Tensor& query_batch, tensor::Tensor& score_batch,
                                   const tensor::Tensor& mha_out,
                                   const tensor::Tensor& key_cache,
                                   const tensor::Tensor& value_cache, CudaConfig* config) {
  int32_t batch = query_batch.get_dim(0);
  int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Per-seq KV capacity derives from the block table width (continuous
  // layout: stride 1 x max_seq_len).
  const int32_t max_seq_len = table_stride * block_size;
  int32_t num_splits = flash_decoding_num_splits(max_seq_len);
  CHECK_LE(head_size, 256)
      << "Paged Flash Decoding requires head_size <= 256 (one output dim per thread).";
  CHECK(block_size > 0 && 256 % block_size == 0)
      << "block_size must be a positive divisor of 256 (8/16/32); got " << block_size;
  CHECK(config != nullptr);

  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  __nv_bfloat16* partials = const_cast<__nv_bfloat16*>(score_batch.ptr<__nv_bfloat16>());
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  const __nv_bfloat16* kcache = key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = value_cache.ptr<__nv_bfloat16>();

  cudaStream_t stream = config->stream;
  // Decode fast path: paged_attn_warp2_kernel_bf16 — one warp per (batch row,
  // q head), 4-deep pipelined K/V loads, warp-shuffle online softmax, no
  // block-wide syncs. At decode shapes it runs at the memory-traffic floor;
  // group-shared smem staging measured 2.5x slower (page loads serialize
  // behind __syncthreads) and the split-partials path below ~2.6x slower.
  // That leaves the split-partials path as the fallback for geometries warp2
  // does not serve: head_size != 128 (it owns exactly 4 head dims per lane),
  // the continuous block-table layout (table_stride == 0) and block_size < 4
  // (its group of 4 positions must stay inside one page).
  if (head_size == 128 && table_stride > 0 && block_size >= 4) {
    const int warps_per_block = 256 / 32;
    const int total_warps = batch * head_num;
    const int block_shift = block_shift_for(block_size);
    const unsigned grid = (total_warps + warps_per_block - 1) / warps_per_block;
    paged_attn_warp2_kernel_bf16<<<grid, 256, 0, stream>>>(
        positions.ptr<int32_t>(), block_table.ptr<int32_t>(), table_stride, num_blocks,
        block_size, block_shift, layer_idx, dim, query, output, kcache, vcache, kv_dim,
        kv_head_num, head_num, head_size);
    return;
  }
  // Fallback (any head_size): per (batch row, q head, split) flash decoding
  // into bf16 split partials, then a combine pass. Needs score_batch sized for
  // batch * head_num * num_splits splits of (head_size + 2) elements.
  // s_p must hold the largest tile any split can own.
  int max_tile = (max_seq_len + num_splits - 1) / num_splits;
  int smem_bytes = (head_size + max_tile) * sizeof(float);
  int total_blocks = batch * head_num * num_splits;
  paged_flash_decoding_kernel_bf16<<<total_blocks, 256, smem_bytes, stream>>>(
      positions.ptr<int32_t>(), block_table.ptr<int32_t>(), table_stride, num_blocks, block_size,
      layer_idx, dim, query, partials, kcache, vcache, kv_dim, kv_head_num, head_num, head_size,
      num_splits);
  paged_flash_decoding_combine_kernel_bf16<<<batch * head_num, 256, 0, stream>>>(
      partials, output, dim, head_num, head_size, num_splits, batch);
}

// ========== Paged prefill attention (FA2-style, one KV read per tile) =======
// One 256-thread block per (q block of B_q = 8 rows, prefill sequence, kv
// head): warp w owns row q_begin + w and all G = head_num / kv_head_num q
// heads that read that kv head. The K/V rows of a kv tile are staged into
// smem once per (CTA, tile) and consumed by every warp and every q head of
// the group, so a prefill chunk's KV traffic drops from the decode kernel's
// O(head_num * N^2 / 2) (each row re-reads its whole causal prefix once per q
// head) to O(kv_head_num * ceil(N / B_q) * N / 2) — for Qwen3-4B (H=32,
// kv_head=8) and B_q = 8 that is 32x less traffic per layer.
//
// The per-(row, head, position) arithmetic is deliberately the same sequence
// of fp32 ops as paged_attn_warp2_kernel_bf16: 4 contiguous head dims per
// lane, the softmax scale folded into q, groups of 4 positions sharing one
// rescale, a 5-step shfl_xor score reduction, and the same < 4-position tail.
// A prefill row therefore comes out bit-identical whether it ran here or on
// the decode kernel, which is what lets mixed steps keep the exact same
// sampled tokens (verify_tokens A/B).
template <int G>
__global__ void paged_prefill_kernel_bf16(
    const int32_t* __restrict__ positions, const int32_t* __restrict__ seq_row_start,
    const int32_t* __restrict__ block_table, int32_t table_stride, int32_t num_blocks,
    int32_t block_size, int32_t block_shift, int32_t layer_idx, int32_t dim,
    const __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ key_cache, const __nv_bfloat16* __restrict__ value_cache,
    int32_t kv_dim, int32_t kv_head_num, int32_t head_num, int32_t head_size) {
  constexpr int kVec = 4;       // head dims per lane (one uint2 load per row)
  constexpr int kGroup = 4;     // positions sharing one softmax rescale (as in warp2)
  constexpr int kTileKv = 64;   // KV rows staged per tile
  constexpr int kLoadVec = 8;   // bf16 per loader vector (16B, one uint4)
  constexpr int kChunks = 128 / kLoadVec;  // loader vectors per row (head_size == 128)
  constexpr int kPad = 136;     // 136 * 2B = 272B: rows stay 16B aligned
  __shared__ __align__(16) __nv_bfloat16 s_k[kTileKv][kPad];
  __shared__ __align__(16) __nv_bfloat16 s_v[kTileKv][kPad];
  __shared__ int s_max_pos;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = blockDim.x >> 5;

  const int seq = blockIdx.y;
  const int kv_head = blockIdx.z;
  const int row_start = seq_row_start[seq];
  const int row_end = seq_row_start[seq + 1];
  const int q_begin = row_start + blockIdx.x * nwarps;  // one q row per warp
  // grid.x is sized from the batch's total prefill rows, so blocks past a
  // short sequence's end exist and exit here (a few hundred ns each).
  if (q_begin >= row_end) return;
  const int row_stop = min(q_begin + nwarps, row_end);
  const int my_row = q_begin + warp;
  const bool active = my_row < row_stop;
  const int limit = active ? positions[my_row] + 1 : 0;  // positions [0, limit)

  // CTA-wide causal limit: no position beyond this needs to be staged. Rows
  // are normally consecutive chunk tokens, but nothing guarantees it (a
  // rewritten sequence can hold stale rows), so take a real max.
  if (tid == 0) {
    int m = 0;
    for (int r = q_begin; r < row_stop; ++r) m = max(m, positions[r] + 1);
    s_max_pos = m;
  }
  __syncthreads();
  const int tiles = (s_max_pos + kTileKv - 1) / kTileKv;

  const float scale = 1.f / sqrtf(static_cast<float>(head_size));
  // q heads served by this kv head; G may exceed the exact group size when
  // head_num is not a multiple of kv_head_num (head_num=36 / 8 -> 5,5,4,...).
  const int head_begin = (kv_head * head_num + kv_head_num - 1) / kv_head_num;
  const int head_end = ((kv_head + 1) * head_num + kv_head_num - 1) / kv_head_num;
  const int heads_here = head_end - head_begin;

  float q[G][kVec];
  if (active) {
#pragma unroll
    for (int i = 0; i < G; ++i) {
      if (i >= heads_here) break;
      const __nv_bfloat16* q_head =
          query + static_cast<int64_t>(my_row) * dim + (head_begin + i) * head_size + lane * kVec;
      const uint2 raw = *reinterpret_cast<const uint2*>(q_head);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      q[i][0] = __bfloat162float(h[0].x) * scale;
      q[i][1] = __bfloat162float(h[0].y) * scale;
      q[i][2] = __bfloat162float(h[1].x) * scale;
      q[i][3] = __bfloat162float(h[1].y) * scale;
    }
  }

  float m_i[G];
  float l_i[G];
  float acc[G][kVec];
#pragma unroll
  for (int i = 0; i < G; ++i) {
    m_i[i] = -FLT_MAX;
    l_i[i] = 0.f;
#pragma unroll
    for (int c = 0; c < kVec; ++c) acc[i][c] = 0.f;
  }

  // All rows of a prefill sequence share one block table row (the scheduler
  // writes it per batch row from the sequence's page table), so the CTA reads
  // the first row's copy.
  const int32_t* table_row = block_table + static_cast<int64_t>(row_start) * table_stride;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const int kv_head_off = (head_begin * kv_head_num / head_num) * head_size;

  for (int t = 0; t < tiles; ++t) {
    // tile_pos is the sequence position of this tile's first KV row; the tile
    // covers positions [tile_pos, tile_pos + kTileKv) of the sequence.
    const int tile_pos = t * kTileKv;
    const int tile_n = min(kTileKv, s_max_pos - tile_pos);
    // Cooperative load: thread -> (row, 16B chunk of the 128-dim head slice).
    // kChunks divides the CTA size, so chunk = tid % kChunks addresses like an
    // AoS store and row_off walks the tile in CTA-wide strides.
    const int chunk = tid % kChunks;
    const int row_off = tid / kChunks;
    const int ch = chunk * kLoadVec;
    for (int r = row_off; r < tile_n; r += blockDim.x / kChunks) {
      const int gpos = tile_pos + r;
      const int32_t pg = __ldg(table_row + (gpos >> block_shift));
      const int64_t src = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                          static_cast<int64_t>(gpos & (block_size - 1)) * kv_dim + kv_head_off +
                          ch;
      *reinterpret_cast<uint4*>(&s_k[r][ch]) =
          *reinterpret_cast<const uint4*>(key_cache + src);
      *reinterpret_cast<uint4*>(&s_v[r][ch]) =
          *reinterpret_cast<const uint4*>(value_cache + src);
    }
    __syncthreads();

    if (active) {
      const int p_end = min(tile_pos + kTileKv, limit);
      int p = tile_pos;
      // Complete groups of 4 positions: one rescale per group, exactly the
      // walk warp2 makes over the same prefix.
      for (; p + kGroup <= p_end; p += kGroup) {
        float kf[kGroup][kVec];
        float vf[kGroup][kVec];
        const int sp = p - tile_pos;
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          // The K/V slice of a kv head is shared by all G q heads, so it is
          // read once per (position, lane) and reused across the head loop.
          const uint2 kraw = *reinterpret_cast<const uint2*>(&s_k[sp + j][lane * kVec]);
          const uint2 vraw = *reinterpret_cast<const uint2*>(&s_v[sp + j][lane * kVec]);
          const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kraw);
          const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vraw);
          kf[j][0] = __bfloat162float(kh[0].x);
          kf[j][1] = __bfloat162float(kh[0].y);
          kf[j][2] = __bfloat162float(kh[1].x);
          kf[j][3] = __bfloat162float(kh[1].y);
          vf[j][0] = __bfloat162float(vh[0].x);
          vf[j][1] = __bfloat162float(vh[0].y);
          vf[j][2] = __bfloat162float(vh[1].x);
          vf[j][3] = __bfloat162float(vh[1].y);
        }
#pragma unroll
        for (int i = 0; i < G; ++i) {
          if (i >= heads_here) break;
          float s[kGroup];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            s[j] = q[i][0] * kf[j][0] + q[i][1] * kf[j][1] + q[i][2] * kf[j][2] +
                   q[i][3] * kf[j][3];
          }
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
#pragma unroll
            for (int off = 16; off; off >>= 1) {
              s[j] += __shfl_xor_sync(0xffffffffu, s[j], off);
            }
          }
          float m_new = m_i[i];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            m_new = fmaxf(m_new, s[j]);
          }
          const float alpha = __expf(m_i[i] - m_new);
          float l_new = l_i[i] * alpha;
          float pr[kGroup];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            pr[j] = __expf(s[j] - m_new);
            l_new += pr[j];
          }
#pragma unroll
          for (int c = 0; c < kVec; ++c) {
            float a = acc[i][c] * alpha;
#pragma unroll
            for (int j = 0; j < kGroup; ++j) {
              a += pr[j] * vf[j][c];
            }
            acc[i][c] = a;
          }
          m_i[i] = m_new;
          l_i[i] = l_new;
        }
      }
      // Fewer than 4 positions left in this row's prefix (warp2's tail).
      for (; p < p_end; ++p) {
        const int sp = p - tile_pos;
        const uint2 kraw = *reinterpret_cast<const uint2*>(&s_k[sp][lane * kVec]);
        const uint2 vraw = *reinterpret_cast<const uint2*>(&s_v[sp][lane * kVec]);
        const __nv_bfloat162* kh = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        const __nv_bfloat162* vh = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        const float kf0 = __bfloat162float(kh[0].x);
        const float kf1 = __bfloat162float(kh[0].y);
        const float kf2 = __bfloat162float(kh[1].x);
        const float kf3 = __bfloat162float(kh[1].y);
        const float vf0 = __bfloat162float(vh[0].x);
        const float vf1 = __bfloat162float(vh[0].y);
        const float vf2 = __bfloat162float(vh[1].x);
        const float vf3 = __bfloat162float(vh[1].y);
#pragma unroll
        for (int i = 0; i < G; ++i) {
          if (i >= heads_here) break;
          float s = q[i][0] * kf0 + q[i][1] * kf1 + q[i][2] * kf2 + q[i][3] * kf3;
#pragma unroll
          for (int off = 16; off; off >>= 1) {
            s += __shfl_xor_sync(0xffffffffu, s, off);
          }
          const float m_new = fmaxf(m_i[i], s);
          const float alpha = __expf(m_i[i] - m_new);
          const float pr = __expf(s - m_new);
          l_i[i] = l_i[i] * alpha + pr;
          acc[i][0] = acc[i][0] * alpha + pr * vf0;
          acc[i][1] = acc[i][1] * alpha + pr * vf1;
          acc[i][2] = acc[i][2] * alpha + pr * vf2;
          acc[i][3] = acc[i][3] * alpha + pr * vf3;
          m_i[i] = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (!active) return;
#pragma unroll
  for (int i = 0; i < G; ++i) {
    if (i >= heads_here) break;
    const float inv_l = 1.f / l_i[i];
    const int64_t out_base =
        static_cast<int64_t>(my_row) * dim + (head_begin + i) * head_size + lane * kVec;
    __nv_bfloat162 o[2];
    o[0] = __floats2bfloat162_rn(acc[i][0] * inv_l, acc[i][1] * inv_l);
    o[1] = __floats2bfloat162_rn(acc[i][2] * inv_l, acc[i][3] * inv_l);
    *reinterpret_cast<uint2*>(output + out_base) = *reinterpret_cast<const uint2*>(o);
  }
}

// ========== Mixed decode + prefill dispatch (see paged_kernels.cuh) ==========
namespace {
// Zero-copy view of rows [row_offset, row_offset + rows) of a [batch, width]
// tensor (width == 0: a flat [batch] tensor). Rows are contiguous at a fixed
// stride, so the view needs no data movement.
tensor::Tensor row_range_view(const tensor::Tensor& src, int32_t row_offset, int32_t rows,
                              int32_t width) {
  const size_t elem = base::DataTypeSize(src.data_type());
  uint8_t* base = const_cast<uint8_t*>(src.ptr<uint8_t>());
  if (width > 0) {
    base += static_cast<int64_t>(row_offset) * width * elem;
    tensor::Tensor view(src.data_type(), std::vector<int32_t>{rows, width}, false, nullptr, base);
    view.set_device_type(src.device_type());
    return view;
  }
  base += static_cast<int64_t>(row_offset) * elem;
  tensor::Tensor view(src.data_type(), std::vector<int32_t>{rows}, false, nullptr, base);
  view.set_device_type(src.device_type());
  return view;
}

// Geometry paged_prefill_kernel_bf16 can serve: the kernel reproduces warp2's
// arithmetic, so it inherits warp2's head_size == 128 (4 head dims per lane),
// needs a paged block table (table_stride > 0) and at least 4 positions per
// page (kGroup positions per rescale must not straddle pages). Its CTA maps one
// q row per warp and one kv head per block, with one kernel instantiation per
// q-heads-per-kv-head group, so an uneven GQA split is fine as long as the
// ceil ratio fits in the 1..8 range of instantiations.
static bool prefill_geometry_ok(int32_t head_num, int32_t kv_head_num, int32_t head_size,
                                int32_t table_stride, int32_t block_size) {
  if (head_num <= 0 || kv_head_num <= 0) return false;
  const int32_t heads_per_kv = (head_num + kv_head_num - 1) / kv_head_num;
  return head_size == 128 && table_stride > 0 && block_size >= 4 &&
         (block_size & (block_size - 1)) == 0 && heads_per_kv <= 8;
}
}  // namespace

// Prefill-row attention: rows [row_begin, batch) are prefill rows of
// num_prefill_seqs sequences, seq_row_start[s] .. seq_row_start[s+1) being the
// rows of sequence s. Each row is one prompt token with its own position, and
// every row attends to the full causal prefix [0, pos] — the KV of earlier
// chunks (and of a prefix-cache hit) is already in the cache by the time
// attention runs, so no chunk bookkeeping is needed here.
//
// This wrapper picks the kernel's G = q heads per kv head instantiation and
// sizes the grid. Callers must have checked prefill_geometry_ok() first:
// paged_attention_dispatch owns that decision because the only correct thing
// to do with an unservable geometry is to leave the prefill rows on the decode
// kernel, which needs the caller's partials buffer.
void paged_prefill_attention_cu_batch(int32_t row_begin, int32_t head_num, int32_t layer_idx,
                                      int32_t num_blocks, int32_t block_size, int32_t kv_dim,
                                      int32_t kv_head_num, int32_t head_size,
                                      const tensor::Tensor* seq_row_start,
                                      int32_t num_prefill_seqs, const tensor::Tensor& positions,
                                      const tensor::Tensor& block_table,
                                      const tensor::Tensor& query_batch,
                                      const tensor::Tensor& mha_out,
                                      const tensor::Tensor& key_cache,
                                      const tensor::Tensor& value_cache, CudaConfig* config) {
  const int32_t batch = positions.get_dim(0);
  const int32_t rows = batch - row_begin;
  if (rows <= 0 || num_prefill_seqs <= 0 || seq_row_start == nullptr) return;
  CHECK(config != nullptr);
  const int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Rounded up: with an uneven GQA split (head_num not a multiple of
  // kv_head_num) the per-kv-head groups hold ceil or floor of the ratio.
  const int32_t heads_per_kv = (head_num + kv_head_num - 1) / kv_head_num;
  CHECK(prefill_geometry_ok(head_num, kv_head_num, head_size, table_stride, block_size))
      << "paged_prefill_kernel_bf16 cannot serve head_size=" << head_size
      << " table_stride=" << table_stride << " block_size=" << block_size
      << " kv_head_num=" << kv_head_num << "; the prefill rows must stay on the decode kernel.";

  const int32_t block_shift = block_shift_for(block_size);
  constexpr int kWarpsPerBlock = 8;  // one q row per warp, 8 rows per block
  // grid.x is bounded by the batch's total prefill rows, which is >= the
  // longest prefill sequence in the batch: blocks past a sequence's end exit
  // immediately (2 loads) instead of needing a host-side max_q_len.
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock, num_prefill_seqs, kv_head_num);
  const dim3 block(32 * kWarpsPerBlock);
  const int32_t* pos = positions.ptr<int32_t>();
  const int32_t* row_map = seq_row_start->ptr<int32_t>();
  const int32_t* table = block_table.ptr<int32_t>();
  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  const __nv_bfloat16* kcache = key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = value_cache.ptr<__nv_bfloat16>();
  cudaStream_t stream = config->stream;
  switch (heads_per_kv) {
    case 1:
      paged_prefill_kernel_bf16<1><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 2:
      paged_prefill_kernel_bf16<2><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 3:
      paged_prefill_kernel_bf16<3><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 4:
      paged_prefill_kernel_bf16<4><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 5:
      paged_prefill_kernel_bf16<5><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 6:
      paged_prefill_kernel_bf16<6><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    case 7:
      paged_prefill_kernel_bf16<7><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
    default:
      paged_prefill_kernel_bf16<8><<<grid, block, 0, stream>>>(
          pos, row_map, table, table_stride, num_blocks, block_size, block_shift, layer_idx, dim,
          query, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
      break;
  }
}

void paged_attention_dispatch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, int32_t num_decode_rows,
                              const tensor::Tensor* seq_row_start, int32_t num_prefill_seqs,
                              const tensor::Tensor& positions, const tensor::Tensor& block_table,
                              const tensor::Tensor& query_batch, tensor::Tensor& score_batch,
                              const tensor::Tensor& mha_out, const tensor::Tensor& key_cache,
                              const tensor::Tensor& value_cache, CudaConfig* config) {
  const int32_t batch = query_batch.get_dim(0);
  CHECK(config != nullptr);
  const int32_t decode_rows = std::max(0, std::min(num_decode_rows, batch));
  const bool has_prefill = decode_rows < batch && seq_row_start != nullptr &&
                           num_prefill_seqs > 0;
  const int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Only a geometry both kernels serve can be split by row type: the decode
  // rows go to warp2 and the prefill rows to the prefill kernel, each writing
  // its own row range of mha_out. Anything else (head_size != 128, an
  // unsupported page size, no prefill rows at all) keeps the whole batch on
  // paged_attention_cu_batch, which dispatches per kernel internally.
  if (!has_prefill ||
      !prefill_geometry_ok(head_num, kv_head_num, head_size, table_stride, block_size)) {
    paged_attention_cu_batch(head_num, layer_idx, num_blocks, block_size, kv_dim, kv_head_num,
                             head_size, positions, block_table, query_batch, score_batch, mha_out,
                             key_cache, value_cache, config);
    return;
  }
  if (decode_rows > 0) {
    paged_attention_cu_batch(head_num, layer_idx, num_blocks, block_size, kv_dim, kv_head_num,
                             head_size, row_range_view(positions, 0, decode_rows, 0),
                             row_range_view(block_table, 0, decode_rows, table_stride),
                             row_range_view(query_batch, 0, decode_rows, dim), score_batch,
                             row_range_view(mha_out, 0, decode_rows, dim), key_cache, value_cache,
                             config);
  }
  paged_prefill_attention_cu_batch(decode_rows, head_num, layer_idx, num_blocks, block_size,
                                   kv_dim, kv_head_num, head_size, seq_row_start, num_prefill_seqs,
                                   positions, block_table, query_batch, mha_out, key_cache,
                                   value_cache, config);
}

}  // namespace kernel
