#include "rope_kernel.h"
namespace kernel {
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache,
                             float rope_theta) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(rope_theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim ++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec =
            const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                      : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}

void rope_kernel_cpu_batch(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream) {
  UNUSED(stream);
  CHECK(input_q.dims_size() == 2 && input_k.dims_size() == 2);
  const int32_t batch = input_q.get_dim(0);
  CHECK_EQ(input_k.get_dim(0), batch);
  CHECK_EQ(input_pos.size(), static_cast<size_t>(batch));

  const int32_t* pos_ptr = input_pos.ptr<int32_t>();
  const float* sin_ptr = sin_cache.ptr<float>();
  const float* cos_ptr = cos_cache.ptr<float>();
  float* q_ptr = const_cast<float*>(input_q.ptr<float>());
  float* k_ptr = const_cast<float*>(input_k.ptr<float>());

  const int32_t head_pair_count = head_size / 2;
  const int32_t num_heads = dim / head_size;
  const int32_t total_pairs = num_heads * head_pair_count;

  // Same math as rope_kernel_cu_fp32_batch, one OpenMP task per rotary pair.
#pragma omp parallel for collapse(2) schedule(static)
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
      const int32_t pos = pos_ptr[b];
      const int32_t head_idx = pair_idx / head_pair_count;
      const int32_t head_dim = pair_idx % head_pair_count;

      const int32_t i = head_idx * head_size;
      const int32_t v0_idx = i + head_dim;
      const int32_t v1_idx = i + head_dim + head_size / 2;

      const float fci = sin_ptr[pos * head_size + head_dim * 2];
      const float fcr = cos_ptr[pos * head_size + head_dim * 2];

      const int32_t rotn = i < kv_dim ? 2 : 1;  // 2 = q & k, 1 = q only
      float* q_b = q_ptr + static_cast<int64_t>(b) * dim;
      float* k_b = k_ptr + static_cast<int64_t>(b) * kv_dim;
      for (int32_t v = 0; v < rotn; v++) {
        float* vec = (v == 0 ? q_b : k_b);
        const float v0 = vec[v0_idx];
        const float v1 = vec[v1_idx];
        vec[v0_idx] = fcr * v0 - fci * v1;
        vec[v1_idx] = fcr * v1 + fci * v0;
      }
    }
  }
}
}  // namespace kernel