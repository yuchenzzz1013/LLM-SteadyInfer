#include "rope_kernel.cuh"
namespace kernel {

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx >= total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                              float rope_theta) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(rope_theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream,
                           float rope_theta) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                            const_cast<float*>(sin_cache.ptr<float>()),
                                            const_cast<float*>(cos_cache.ptr<float>()),
                                            rope_theta);
  } else {
    sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                 const_cast<float*>(cos_cache.ptr<float>()),
                                 rope_theta);
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  int threads = 128;
  int blocks = (dim + threads - 1) / threads;

  int32_t batch = 1;
  if (input_q.dims_size() > 1) {
    batch = input_q.get_dim(0);
  }

  if (batch == 1) {
    const int32_t pos = *input_pos.ptr<int32_t>(0);
    if (stream_) {
      rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
          sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
      rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                               input_k.ptr<float>(), sin_cache.ptr<float>(),
                                               cos_cache.ptr<float>());
    }
  } else {
    const int32_t* pos_ptr = input_pos.ptr<int32_t>();
    const float* input_q_base = input_q.ptr<float>();
    const float* input_k_base = input_k.ptr<float>();
    for (int b = 0; b < batch; ++b) {
      const int32_t pos = pos_ptr[b];
      const float* q_b = input_q_base + b * dim;
      const float* k_b = input_k_base + b * kv_dim;
      if (stream_) {
        rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
            pos, dim, kv_dim, head_size, q_b, k_b,
            sin_cache.ptr<float>(), cos_cache.ptr<float>());
      } else {
        rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, q_b, k_b,
                                                  sin_cache.ptr<float>(), cos_cache.ptr<float>());
      }
    }
  }
}

__global__ void rope_kernel_cu_fp32_batch(const int32_t* positions, int32_t dim, int32_t kv_dim,
                                          int32_t head_size, const float* input_q,
                                          const float* input_k, const float* sin_cache,
                                          const float* cos_cache) {
  int pair_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (pair_idx >= total_pairs) {
    return;
  }
  int b = blockIdx.y;
  int pos = positions[b];

  int head_idx = pair_idx / head_pair_count;
  int head_dim = pair_idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  const float* q_b = input_q + b * dim;
  const float* k_b = input_k + b * kv_dim;
  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? q_b : k_b);
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

void rope_kernel_cu_batch(int32_t dim, int32_t kv_dim, int32_t head_size,
                          const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                          const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                          const tensor::Tensor& cos_cache, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  int32_t batch = input_q.get_dim(0);
  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  int threads = 128;
  int blocks_x = (total_pairs + threads - 1) / threads;
  dim3 grid(blocks_x, batch);
  if (stream_) {
    rope_kernel_cu_fp32_batch<<<grid, threads, 0, stream_>>>(
        input_pos.ptr<int32_t>(), dim, kv_dim, head_size, input_q.ptr<float>(),
        input_k.ptr<float>(), sin_cache.ptr<float>(), cos_cache.ptr<float>());
  } else {
    rope_kernel_cu_fp32_batch<<<grid, threads>>>(
        input_pos.ptr<int32_t>(), dim, kv_dim, head_size, input_q.ptr<float>(),
        input_k.ptr<float>(), sin_cache.ptr<float>(), cos_cache.ptr<float>());
  }
}

// Scatter one [kv_dim] row per batch element into the head-dim-contiguous
// cache layout [num_layers, num_slots, kv_dim, max_seq_len]:
//   dst[layer][slot][d][pos] = src[b][d]
// The write is strided (stride max_seq_len between consecutive d) — inherent
// to the layout; decode-side reads win back far more bandwidth than this
// kv_dim-element scatter costs.
__global__ void kv_scatter_kernel(const float* src, float* dst, const int32_t* kv_offsets,
                                  const int32_t* positions, int32_t kv_dim, int32_t num_slots,
                                  int32_t max_seq_len, int32_t layer_idx) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int slot = kv_offsets[b];
  int pos = positions[b];
  int64_t base = static_cast<int64_t>(layer_idx) * num_slots * kv_dim * max_seq_len +
                 static_cast<int64_t>(slot) * kv_dim * max_seq_len + pos;
  const float* src_row = src + static_cast<int64_t>(b) * kv_dim;
  for (int d = tid; d < kv_dim; d += blockDim.x) {
    dst[base + static_cast<int64_t>(d) * max_seq_len] = src_row[d];
  }
}

void kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                   const tensor::Tensor& kv_offsets, const tensor::Tensor& positions,
                   int32_t kv_dim, int32_t num_slots, int32_t max_seq_len, int32_t layer_idx,
                   void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  int32_t batch = src.get_dim(0);
  float* dst = const_cast<float*>(dst_cache.ptr<float>());
  if (stream_) {
    kv_scatter_kernel<<<batch, 256, 0, stream_>>>(
        src.ptr<float>(), dst, kv_offsets.ptr<int32_t>(), positions.ptr<int32_t>(), kv_dim,
        num_slots, max_seq_len, layer_idx);
  } else {
    kv_scatter_kernel<<<batch, 256>>>(src.ptr<float>(), dst, kv_offsets.ptr<int32_t>(),
                                      positions.ptr<int32_t>(), kv_dim, num_slots, max_seq_len,
                                      layer_idx);
  }
}
}  // namespace kernel