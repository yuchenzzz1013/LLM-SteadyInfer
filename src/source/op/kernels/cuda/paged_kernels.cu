#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <glog/logging.h>
#include "mha_kernel.cuh"
#include "paged_kernels.cuh"

namespace kernel {

// ========== Flash softmax helper (per-TU copy of the one in mha_kernel.cu) ==========
// Flash-softmax over one KV tile: p_smem[0..tile_len) holds RAW scores (the
// caller fills it with a strided loop, so tile_len may exceed blockDim).
// This block-reduces the max, rewrites p_smem in place as exp(s - m), and
// block-reduces the sum. Returns (m, l) broadcast to every thread.
__device__ void paged_flash_softmax_tile(float* p_smem, int tile_len, float& m, float& l) {
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

// ========== Paged KV scatter ==========
// One [kv_dim] row per batch element, written through the block table:
//   dst[layer][table[b][pos / block_size]][pos % block_size][d] = src[b][d]
// d-contiguous writes inside a page; the per-page write is coalesced when
// consecutive threads cover consecutive d.
__global__ void paged_kv_scatter_kernel(const float* src, float* dst, const int32_t* block_table,
                                        int32_t table_stride, const int32_t* positions,
                                        int32_t kv_dim, int32_t num_blocks, int32_t block_size,
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
  const float* src_row = src + static_cast<int64_t>(b) * kv_dim;
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
  float* dst = const_cast<float*>(dst_cache.ptr<float>());
  const int32_t table_stride = block_table.get_dim(1);
  if (stream_) {
    paged_kv_scatter_kernel<<<batch, 256, 0, stream_>>>(
        src.ptr<float>(), dst, block_table.ptr<int32_t>(), table_stride, positions.ptr<int32_t>(),
        kv_dim, num_blocks, block_size, layer_idx);
  } else {
    paged_kv_scatter_kernel<<<batch, 256>>>(
        src.ptr<float>(), dst, block_table.ptr<int32_t>(), table_stride, positions.ptr<int32_t>(),
        kv_dim, num_blocks, block_size, layer_idx);
  }
}

// ========== Paged Flash Decoding (split-KV + reduce) ==========
// Same grid / thread mapping / GQA mapping / partials format as the
// continuous flash_decoding_kernel (mha_kernel.cu) — only the cache
// addressing goes through the block table — so fp32 results are bit-identical
// to the continuous-layout path for the same inputs.
__global__ void paged_flash_decoding_kernel(const int32_t* positions, const int32_t* block_table,
                                            int32_t table_stride, int32_t num_blocks,
                                            int32_t block_size, int32_t layer_idx, int32_t dim,
                                            const float* query, float* partials,
                                            const float* key_cache, const float* value_cache,
                                            int32_t kv_dim, int32_t kv_head_num,
                                            int32_t head_num, int32_t head_size,
                                            int32_t num_splits) {
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
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const float* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = q_head[i];
  }
  __syncthreads();

  // Scores over this split's KV tile: s_t = sum_d q[d] * K[pos][head_offset + d].
  // Strided over threads like the continuous kernel. blockDim (256) is a
  // multiple of block_size (8/16/32, checked on the host), so each thread's
  // in-page offset is constant across its iterations and only the block id
  // advances (by blockDim / block_size per step).
  const int off_t = (tile_start + threadIdx.x) % block_size;
  int pos_t = tile_start + threadIdx.x;
  int block_t = -1;
  if (threadIdx.x < tile_len) {
    block_t = __ldg(table_row + (pos_t / block_size));
  }
  const float* k_base = key_cache + layer_base + static_cast<int64_t>(off_t) * kv_dim + head_offset;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    const float* k_row = k_base + static_cast<int64_t>(block_t) * (block_size * kv_dim);
    float s_t = 0.f;
    for (int d = 0; d < head_size; ++d) {
      s_t += s_query[d] * k_row[d];
    }
    s_p[t] = s_t * scale;
    // Advance to the next position this thread will process (guarded so a
    // thread whose stride exits the tile never reads the table out of range).
    pos_t += blockDim.x;
    if (pos_t < tile_end) {
      block_t = __ldg(table_row + (pos_t / block_size));
    }
  }

  float m_i, l_i;
  paged_flash_softmax_tile(s_p, tile_len, m_i, l_i);

  // Partial output: acc_d = sum_t p_t * V[pos][head_offset + d].
  // Thread d walks the tile's positions in order; the block id is updated
  // only at page boundaries (one __ldg per block_size positions).
  float acc = 0.f;
  const int d = threadIdx.x;
  if (d < head_size) {
    int pos_v = tile_start;
    int block_v = __ldg(table_row + (pos_v / block_size));
    int off_v = pos_v - block_v * block_size;  // pos_v % block_size
    const float* v_base = value_cache + layer_base + head_offset + d;
    for (int tt = 0; tt < tile_len; ++tt) {
      acc += s_p[tt] * v_base[static_cast<int64_t>(block_v) * (block_size * kv_dim) +
                             static_cast<int64_t>(off_v) * kv_dim];
      if (++off_v == block_size) {
        off_v = 0;
        ++pos_v;
        block_v = __ldg(table_row + (pos_v / block_size));
      }
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

// Reduce pass — identical recombination to flash_decoding_combine_kernel
// (operates on partials only, never touches the cache).
__global__ void paged_flash_decoding_combine_kernel(const float* partials, float* output,
                                                    int32_t dim, int32_t head_num,
                                                    int32_t head_size, int32_t num_splits,
                                                    int32_t batch) {
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

void paged_attention_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& block_table, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              CudaConfig* config) {
  int32_t batch = query_batch.get_dim(0);
  int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Per-seq KV capacity derives from the block table width (continuous
  // layout: stride 1 x max_seq_len). num_splits must stay capacity-derived so
  // grid/smem/partials shapes are stable across steps (CUDA-graph safe).
  const int32_t max_seq_len = table_stride * block_size;
  int32_t num_splits = flash_decoding_num_splits(max_seq_len);
  CHECK_LE(head_size, 256)
      << "Paged Flash Decoding requires head_size <= 256 (one output dim per thread).";
  CHECK(block_size > 0 && 256 % block_size == 0)
      << "block_size must be a positive divisor of 256 (8/16/32); got " << block_size;

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
  paged_flash_decoding_kernel<<<total_blocks, 256, smem_bytes, stream>>>(
      positions.ptr<int32_t>(), block_table.ptr<int32_t>(), table_stride, num_blocks, block_size,
      layer_idx, dim, query, partials, kcache, vcache, kv_dim, kv_head_num, head_num, head_size,
      num_splits);
  paged_flash_decoding_combine_kernel<<<batch * head_num, 256, 0, stream>>>(
      partials, output, dim, head_num, head_size, num_splits, batch);
}

}  // namespace kernel
