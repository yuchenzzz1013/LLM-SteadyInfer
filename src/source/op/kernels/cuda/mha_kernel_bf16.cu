// CUDA attention kernels (all BF16): continuous-layout Flash Decoding
// (mha_kernel_cu / mha_kernel_cu_batch), paged KV scatter and paged Flash
// Decoding (paged_kv_scatter_cu / paged_attention_cu_batch).
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
  const int off = pos - block_id * block_size;  // pos % block_size
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
    int off_v = pos_v - block_v * block_size;  // pos_v % block_size
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

}  // namespace kernel
