#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
namespace kernel {
// The flash kernels use one accumulator per thread for the output dims, so
// head_size must not exceed the 256-thread block (real models: <= 128).

// ========== Flash Attention helpers ==========
// Flash-softmax over one KV tile: p_smem[0..tile_len) holds RAW scores (the
// caller fills it with a strided loop, so tile_len may exceed blockDim).
// This block-reduces the max, rewrites p_smem in place as exp(s - m), and
// block-reduces the sum. Returns (m, l) broadcast to every thread.
__device__ void flash_softmax_tile(float* p_smem, int tile_len, float& m, float& l) {
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

// ========== Prefill: Flash Attention V2 style (online softmax + tiling) ==========
// One block per head; the KV sequence [0, pos] is processed in tiles of
// TILE positions. Scores never round-trip through global memory: each tile
// updates (acc, m, l) with the online-softmax recurrence
//   alpha = exp(m_j - m_i);  acc = alpha*acc + sum_t p_t * v_t
//   l = alpha*l + l_j;       m = m_j
// K/V reads use float4 vectorized, per-row contiguous accesses.
// Cache layout (internal, single-sequence path): [num_layers, seq, kv_dim].
constexpr int TILE = 128;

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim,
                                            int32_t kv_head_num, int32_t head_num,
                                            int32_t head_size, int64_t layer_offset) {
  UNUSED(score_ptr);  // Flash: no global score storage, see partials path
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_mem[];
  float* s_query = s_mem;                 // [head_size]
  float* s_p = s_mem + head_size;         // [TILE]

  const float scale = 1.f / sqrtf(float(head_size));
  const float* query_head = query + head * head_size;
  const int seq = pos + 1;
  // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits.
  const int head_offset = (head * kv_head_num / head_num) * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = query_head[i];
  }
  __syncthreads();

  const bool vec_ok = (head_size % 4 == 0) && (head_offset % 4 == 0);
  float m_i = -FLT_MAX;
  float l_i = 0.f;
  // Per-thread accumulator for out[head_size] (thread d owns dimension d).
  float acc = 0.f;

  for (int tile_start = 0; tile_start < seq; tile_start += TILE) {
    const int tile_len = min(TILE, seq - tile_start);

    // Scores: s_t = sum_d q[d] * K[t][head_offset + d], strided over threads.
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const float* k_row = key_cache + layer_offset +
                           static_cast<int64_t>(tile_start + t) * kv_dim + head_offset;
      float s_t = 0.f;
      if (vec_ok) {
        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        const float4* q4 = reinterpret_cast<const float4*>(s_query);
        for (int d = 0; d < head_size / 4; ++d) {
          float4 kv4 = k4[d];
          float4 qv4 = q4[d];
          s_t += kv4.x * qv4.x + kv4.y * qv4.y + kv4.z * qv4.z + kv4.w * qv4.w;
        }
      } else {
        for (int d = 0; d < head_size; ++d) {
          s_t += k_row[d] * s_query[d];
        }
      }
      s_p[t] = s_t * scale;
    }

    float m_j, l_j;
    flash_softmax_tile(s_p, tile_len, m_j, l_j);
    // First tile: m_i == -inf, so alpha must be 1 (not exp(+inf) = inf —
    // that would poison acc and l_i with inf * 0 = NaN).
    const float alpha = (m_i == -FLT_MAX) ? 1.0f : expf(m_j - m_i);
    l_i = alpha * l_i + l_j;
    m_i = m_j;

    // acc = alpha * acc + sum_t p_t * V[t][head_offset + d]
    // Thread d owns output dimension d and walks the tile's V column:
    // per-iteration the warp reads consecutive d -> coalesced global loads.
    acc *= alpha;
    const int d = threadIdx.x;
    if (d < head_size) {
      const float* v_col = value_cache + layer_offset +
                           static_cast<int64_t>(tile_start) * kv_dim + head_offset + d;
      for (int tt = 0; tt < tile_len; ++tt) {
        acc += s_p[tt] * v_col[static_cast<int64_t>(tt) * kv_dim];
      }
    }
  }

  if (threadIdx.x < head_size) {
    output[head * head_size + threadIdx.x] = acc / l_i;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                   CudaConfig* config) {
  UNUSED(device_type);
  CHECK_LE(head_size, 256) << "Flash kernel requires head_size <= 256 (one output dim per thread).";
  // int64: large external caches (num_slots * max_seq_len) can overflow int32.
  int64_t layer_offset = static_cast<int64_t>(layer_index) * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  int smem_bytes = (head_size + TILE) * sizeof(float);
  multi_head_attention_kernel<<<head_num, 256, smem_bytes, stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_head_num, head_num,
      head_size, layer_offset);
}

// ========== Decode / chunked prefill: Flash Decoding (split-KV + reduce) ==========
// The KV range [0, pos] of each (batch, head) is split across num_splits
// blocks; each block runs an online-softmax pass over its tile and writes
// partial (o, m, l) to the partials buffer. A second kernel reduces the
// splits with the standard Flash-Decoding recombination.
//
// Cache layout (external): [num_layers, num_slots, kv_dim, max_seq_len] —
// head-dim contiguous, so per-dimension reads across the KV range are
// coalesced: K[base + (head_offset + d) * max_seq_len + t] is contiguous in t.

// (num_splits helper lives in mha_kernel.cuh — shared with the scratch sizing
// in the model forward paths. The largest tile per split is
// ceil(max_seq_len / num_splits), which sizes the smem score buffer.)

__global__ void flash_decoding_kernel(const int32_t* positions, const int32_t* kv_offsets,
                                      int32_t num_slots, int32_t max_seq_len, int32_t layer_idx,
                                      int32_t dim, const float* query, float* partials,
                                      const float* key_cache, const float* value_cache,
                                      int32_t kv_dim, int32_t kv_head_num, int32_t head_num,
                                      int32_t head_size, int32_t num_splits) {
  // 1-D grid of batch * head_num * num_splits blocks (a 2-D grid.y would
  // overflow 65535 for large prefill batches x many heads).
  const int pair = blockIdx.x / num_splits;  // pair = b * head_num + head
  const int split = blockIdx.x % num_splits;
  const int b = pair / head_num;
  const int head = pair % head_num;
  const int pos = positions[b];
  const int seq = pos + 1;

  const int tile_start = split * seq / num_splits;
  const int tile_end = (split + 1) * seq / num_splits;
  // The combine kernel reads every split's partial slot unconditionally, so
  // empty splits must record a neutral (o=0, m=-inf, l=0) partial instead of
  // leaving the buffer uninitialized/stale.
  float* partial = partials + (static_cast<int64_t>(pair) * num_splits + split) * (head_size + 2);
  if (tile_start >= tile_end) {
    for (int dd = threadIdx.x; dd < head_size; dd += blockDim.x) {
      partial[dd] = 0.f;
    }
    if (threadIdx.x == 0) {
      partial[head_size] = -FLT_MAX;
      partial[head_size + 1] = 0.f;
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
  const float* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = q_head[i];
  }
  __syncthreads();

  // Scores over this split's KV tile: s_t = sum_d q[d] * K[hd + d][t].
  // Strided over threads (a tile can exceed blockDim): K[base +
  // (head_offset + d) * max_seq_len + t] is contiguous in t across the warp
  // -> coalesced global reads.
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    float s_t = 0.f;
    for (int d = 0; d < head_size; ++d) {
      s_t += s_query[d] *
             key_cache[cache_base + static_cast<int64_t>(head_offset + d) * max_seq_len + t];
    }
    s_p[t] = s_t * scale;
  }

  float m_i, l_i;
  flash_softmax_tile(s_p, tile_len, m_i, l_i);

  // Partial output: acc_d = sum_t p_t * V[head_offset + d][t].
  // Thread d walks t: contiguous in t -> coalesced reads.
  float acc = 0.f;
  const int d = threadIdx.x;
  if (d < head_size) {
    const float* v_dim =
        value_cache + cache_base + static_cast<int64_t>(head_offset + d) * max_seq_len;
    for (int tt = 0; tt < tile_len; ++tt) {
      acc += s_p[tt] * v_dim[tt];
    }
  }

  // partials[pair * num_splits + split] -> [o (head_size) | m | l] (declared above)
  if (d < head_size) {
    partial[d] = acc;
  }
  if (threadIdx.x == 0) {
    partial[head_size] = m_i;
    partial[head_size + 1] = l_i;
  }
}

__global__ void flash_decoding_combine_kernel(const float* partials, float* output, int32_t dim,
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

  const float* partial = partials + static_cast<int64_t>(pair) * num_splits * (head_size + 2);

  // Global max over splits.
  float local_m = (threadIdx.x < num_splits) ? partial[threadIdx.x * (head_size + 2) + head_size]
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
    const float* ps = partial + static_cast<int64_t>(s) * (head_size + 2);
    const float w = expf(ps[head_size] - global_m);
    l += w * ps[head_size + 1];
    if (d < head_size) {
      o += w * ps[d];
    }
  }
  if (d < head_size) {
    output[static_cast<int64_t>(b) * dim + head * head_size + d] = o / l;
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

  float* query = const_cast<float*>(query_batch.ptr<float>());
  float* partials = const_cast<float*>(score_batch.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());
  float* kcache = const_cast<float*>(key_cache.ptr<float>());
  float* vcache = const_cast<float*>(value_cache.ptr<float>());

  cudaStream_t stream = config->stream;
  // s_p must hold the largest tile any split can own.
  int max_tile = (max_seq_len + num_splits - 1) / num_splits;
  int smem_bytes = (head_size + max_tile) * sizeof(float);
  int total_blocks = batch * head_num * num_splits;
  flash_decoding_kernel<<<total_blocks, 256, smem_bytes, stream>>>(
      positions.ptr<int32_t>(), kv_offsets.ptr<int32_t>(), num_slots, max_seq_len, layer_idx, dim,
      query, partials, kcache, vcache, kv_dim, kv_head_num, head_num, head_size, num_splits);
  flash_decoding_combine_kernel<<<batch * head_num, 256, 0, stream>>>(
      partials, output, dim, head_num, head_size, num_splits, batch);
}

}  // namespace kernel
