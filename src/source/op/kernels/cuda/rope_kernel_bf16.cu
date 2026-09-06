// CUDA RoPE kernels (single-position + batched), the continuous-layout KV
// scatter and the sin/cos cache fill — all BF16. q/k/caches and the sin/cos
// tables are raw bfloat16 (the tables store bf16-rounded fill values and are
// widened back exactly on load: every bfloat16 is representable in fp32).
// Rotation math runs in float on the bf16 operands; results are rounded back
// to bf16.
#include <cuda_bf16.h>
#include "rope_kernel.cuh"
namespace kernel {

__global__ void rope_kernel_single(int pos, int dim, int kv_dim, int head_size,
                                   __nv_bfloat16* input_q, __nv_bfloat16* input_k,
                                   const __nv_bfloat16* sin_cache,
                                   const __nv_bfloat16* cos_cache) {
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

  float fci = __bfloat162float(sin_cache[pos * head_size + head_dim * 2]);
  float fcr = __bfloat162float(cos_cache[pos * head_size + head_dim * 2]);

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    __nv_bfloat16* vec = (v == 0) ? input_q : input_k;
    float v0 = __bfloat162float(vec[v0_idx]);
    float v1 = __bfloat162float(vec[v1_idx]);
    vec[v0_idx] = __float2bfloat16(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2bfloat16(fcr * v1 + fci * v0);
  }
}

__global__ void rope_kernel_batch(const int32_t* positions, int32_t dim, int32_t kv_dim,
                                  int32_t head_size, const __nv_bfloat16* input_q,
                                  const __nv_bfloat16* input_k, const __nv_bfloat16* sin_cache,
                                  const __nv_bfloat16* cos_cache) {
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

  float fci = __bfloat162float(sin_cache[pos * head_size + head_dim * 2]);
  float fcr = __bfloat162float(cos_cache[pos * head_size + head_dim * 2]);

  int rotn = i < kv_dim ? 2 : 1;

  __nv_bfloat16* q_b = const_cast<__nv_bfloat16*>(input_q) + static_cast<int64_t>(b) * dim;
  __nv_bfloat16* k_b = const_cast<__nv_bfloat16*>(input_k) + static_cast<int64_t>(b) * kv_dim;
  for (int v = 0; v < rotn; v++) {
    __nv_bfloat16* vec = (v == 0) ? q_b : k_b;
    float v0 = __bfloat162float(vec[v0_idx]);
    float v1 = __bfloat162float(vec[v1_idx]);
    vec[v0_idx] = __float2bfloat16(fcr * v0 - fci * v1);
    vec[v1_idx] = __float2bfloat16(fcr * v1 + fci * v0);
  }
}

// Fill the per-position sin/cos caches (BF16 storage, fp32 fill math):
//   cache[pos * head_size + d] = sin / cos(pos * rope_theta^(-d / head_size))
// Values are rounded to bf16 on store; RoPE kernels widen them back to fp32
// on load (exact — bf16 values are a subset of fp32).
__global__ void sin_cos_calc_bf16(int head_size, int max_seq_len, __nv_bfloat16* sin_cache,
                                  __nv_bfloat16* cos_cache, float rope_theta) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq =
        1.0f / pow(rope_theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    sin_cache[pos * head_size + head_dim] = __float2bfloat16(fci);
    cos_cache[pos * head_size + head_dim] = __float2bfloat16(fcr);
  }
}

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream,
                           float rope_theta) {
  CHECK_EQ(sin_cache.is_empty(), false);
  CHECK_EQ(cos_cache.is_empty(), false);
  int threads = head_size;
  if (stream) {
    sin_cos_calc_bf16<<<1, threads, 0, stream>>>(
        head_size, max_seq_len, const_cast<__nv_bfloat16*>(sin_cache.ptr<__nv_bfloat16>()),
        const_cast<__nv_bfloat16*>(cos_cache.ptr<__nv_bfloat16>()), rope_theta);
  } else {
    sin_cos_calc_bf16<<<1, threads>>>(head_size, max_seq_len,
                                      const_cast<__nv_bfloat16*>(sin_cache.ptr<__nv_bfloat16>()),
                                      const_cast<__nv_bfloat16*>(cos_cache.ptr<__nv_bfloat16>()),
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
  const __nv_bfloat16* sin_ptr = sin_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* cos_ptr = cos_cache.ptr<__nv_bfloat16>();

  if (batch == 1) {
    const int32_t pos = *input_pos.ptr<int32_t>(0);
    if (stream_) {
      rope_kernel_single<<<blocks, threads, 0, stream_>>>(
          pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>()),
          const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>()), sin_ptr, cos_ptr);
    } else {
      rope_kernel_single<<<blocks, threads>>>(
          pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(input_q.ptr<__nv_bfloat16>()),
          const_cast<__nv_bfloat16*>(input_k.ptr<__nv_bfloat16>()), sin_ptr, cos_ptr);
    }
  } else {
    const int32_t* pos_ptr = input_pos.ptr<int32_t>();
    const __nv_bfloat16* input_q_base = input_q.ptr<__nv_bfloat16>();
    const __nv_bfloat16* input_k_base = input_k.ptr<__nv_bfloat16>();
    for (int b = 0; b < batch; ++b) {
      const int32_t pos = pos_ptr[b];
      const __nv_bfloat16* q_b = input_q_base + static_cast<int64_t>(b) * dim;
      const __nv_bfloat16* k_b = input_k_base + static_cast<int64_t>(b) * kv_dim;
      if (stream_) {
        rope_kernel_single<<<blocks, threads, 0, stream_>>>(
            pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(q_b),
            const_cast<__nv_bfloat16*>(k_b), sin_ptr, cos_ptr);
      } else {
        rope_kernel_single<<<blocks, threads>>>(
            pos, dim, kv_dim, head_size, const_cast<__nv_bfloat16*>(q_b),
            const_cast<__nv_bfloat16*>(k_b), sin_ptr, cos_ptr);
      }
    }
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
    rope_kernel_batch<<<grid, threads, 0, stream_>>>(
        input_pos.ptr<int32_t>(), dim, kv_dim, head_size, input_q.ptr<__nv_bfloat16>(),
        input_k.ptr<__nv_bfloat16>(), sin_cache.ptr<__nv_bfloat16>(),
        cos_cache.ptr<__nv_bfloat16>());
  } else {
    rope_kernel_batch<<<grid, threads>>>(
        input_pos.ptr<int32_t>(), dim, kv_dim, head_size, input_q.ptr<__nv_bfloat16>(),
        input_k.ptr<__nv_bfloat16>(), sin_cache.ptr<__nv_bfloat16>(),
        cos_cache.ptr<__nv_bfloat16>());
  }
}

// Continuous-layout KV scatter: raw 2-byte element copy into
// dst[layer][slot][d][pos] = src[b][d].
__global__ void kv_scatter_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                  const int32_t* kv_offsets, const int32_t* positions,
                                  int32_t kv_dim, int32_t num_slots, int32_t max_seq_len,
                                  int32_t layer_idx) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int slot = kv_offsets[b];
  int pos = positions[b];
  int64_t base = static_cast<int64_t>(layer_idx) * num_slots * kv_dim * max_seq_len +
                 static_cast<int64_t>(slot) * kv_dim * max_seq_len + pos;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(b) * kv_dim;
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
  __nv_bfloat16* dst = const_cast<__nv_bfloat16*>(dst_cache.ptr<__nv_bfloat16>());
  if (stream_) {
    kv_scatter_kernel<<<batch, 256, 0, stream_>>>(
        src.ptr<__nv_bfloat16>(), dst, kv_offsets.ptr<int32_t>(), positions.ptr<int32_t>(),
        kv_dim, num_slots, max_seq_len, layer_idx);
  } else {
    kv_scatter_kernel<<<batch, 256>>>(
        src.ptr<__nv_bfloat16>(), dst, kv_offsets.ptr<int32_t>(), positions.ptr<int32_t>(),
        kv_dim, num_slots, max_seq_len, layer_idx);
  }
}
}  // namespace kernel
