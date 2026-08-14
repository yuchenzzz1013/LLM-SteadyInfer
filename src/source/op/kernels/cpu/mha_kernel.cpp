#include "../cpu/mha_kernel.h"
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <vector>
#include "../kernels_interface.h"
namespace kernel {
// Per-thread scratch for the batched CPU attention (avoids a heap allocation
// per (batch, head) iteration inside the OpenMP region).
static thread_local std::vector<float> tls_attn_out;  // [head_size]
static thread_local std::vector<float> tls_attn_p;    // [max_seq_len] softmax probs

void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_head_num, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  // int64: large external caches (num_slots * max_seq_len) can overflow int32.
  int64_t layer_offset = static_cast<int64_t>(layer_index) * seq_len * kv_dim;
  float scale = 1.f / std::sqrt(static_cast<float>(head_size));

  std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type == base::DeviceType::kDeviceCPU) {
      allocator = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
      allocator = base::CUDADeviceAllocatorFactory::get_instance();
    }
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);


    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                               query_head_addr);
    query_mat.set_device_type(device_type);

    // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits
    // (e.g. 36 heads / 8 KV heads) correctly and stays within kv_dim.
    int32_t head_offset = (h * kv_head_num / head_num) * head_size;
    for (int32_t t = 0; t <= pos; t++) {
      int32_t cache_offset = t * kv_dim + head_offset;
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));

      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    int32_t cache_offset = head_offset;
    float* value_head_addr =
        const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
    tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                value_head_addr);
    get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                      head_size, kv_dim, config ? config->stream : nullptr);
  }
}

// Batched CPU MHA for continuous batching (decode and chunked prefill).
// One OpenMP task per (batch, head); the KV cache uses the head-dim-contiguous
// external layout [num_layers, num_slots, kv_dim, max_seq_len], so the
// per-dimension KV walks are contiguous and simd-friendly:
//   K[layer][slot][head_offset + d][t] -> contiguous in t for fixed d.
// Two passes over [0, pos] (online-max softmax): pass 1 finds the row max,
// pass 2 accumulates exp(s - max) * V into the output. No per-sequence score
// buffer and no per-token GEMM dispatch — the whole batch is one call.
void mha_kernel_cpu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                          int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                          int32_t head_size, const tensor::Tensor& positions,
                          const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                          tensor::Tensor& mha_out_batch, const tensor::Tensor& key_cache,
                          const tensor::Tensor& value_cache) {
  CHECK(positions.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(query_batch.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(key_cache.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(mha_out_batch.device_type() == base::DeviceType::kDeviceCPU);

  const int32_t batch = query_batch.get_dim(0);
  const int32_t dim = query_batch.get_dim(1);
  CHECK_EQ(positions.size(), static_cast<size_t>(batch));
  CHECK_EQ(kv_offsets.size(), static_cast<size_t>(batch));
  CHECK_EQ(mha_out_batch.get_dim(0), batch);

  const int32_t* pos_ptr = positions.ptr<int32_t>();
  const int32_t* slot_ptr = kv_offsets.ptr<int32_t>();
  const float* q_ptr = query_batch.ptr<float>();
  float* out_ptr = const_cast<float*>(mha_out_batch.ptr<float>());
  const float* kcache = key_cache.ptr<float>();
  const float* vcache = value_cache.ptr<float>();

  const float scale = 1.f / std::sqrt(static_cast<float>(head_size));
  const int64_t slot_stride = static_cast<int64_t>(kv_dim) * max_seq_len;
  const int64_t layer_stride = static_cast<int64_t>(num_slots) * slot_stride;

#pragma omp parallel for collapse(2) schedule(static)
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t h = 0; h < head_num; ++h) {
      const int32_t pos = pos_ptr[b];
      const int32_t slot = slot_ptr[b];
      const int32_t seq = pos + 1;
      // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits.
      const int32_t head_offset = (h * kv_head_num / head_num) * head_size;
      const int64_t base = static_cast<int64_t>(layer_idx) * layer_stride +
                           static_cast<int64_t>(slot) * slot_stride +
                           static_cast<int64_t>(head_offset) * max_seq_len;
      const float* q_head = q_ptr + static_cast<int64_t>(b) * dim + h * head_size;

      // Pass 1: max score for numerical stability (online softmax).
      float max_s = -std::numeric_limits<float>::infinity();
      for (int32_t t = 0; t < seq; ++t) {
        float s = 0.f;
        for (int32_t d = 0; d < head_size; ++d) {
          s += q_head[d] * kcache[base + static_cast<int64_t>(d) * max_seq_len + t];
        }
        s *= scale;
        if (s > max_s) max_s = s;
      }

      // Pass 2: probs exp(s - max) (reuse the dot walk; K reads are the
      // cheap half of the memory traffic).
      if (tls_attn_p.size() < static_cast<size_t>(seq)) {
        tls_attn_p.resize(seq);
      }
      float* p_t = tls_attn_p.data();
      float sum_p = 0.f;
      for (int32_t t = 0; t < seq; ++t) {
        float s = 0.f;
        for (int32_t d = 0; d < head_size; ++d) {
          s += q_head[d] * kcache[base + static_cast<int64_t>(d) * max_seq_len + t];
        }
        p_t[t] = std::exp(s * scale - max_s);
        sum_p += p_t[t];
      }

      // Pass 3: out[d] = sum_t p[t] * V[d][t]. V[d][t] is contiguous in t for
      // fixed d (head-dim-contiguous layout) -> simd-friendly walk.
      if (tls_attn_out.size() < static_cast<size_t>(head_size)) {
        tls_attn_out.resize(head_size);
      }
      float* acc = tls_attn_out.data();
      for (int32_t d = 0; d < head_size; ++d) {
        const float* v_dim = vcache + base + static_cast<int64_t>(d) * max_seq_len;
        float a = 0.f;
#pragma omp simd reduction(+ : a)
        for (int32_t t = 0; t < seq; ++t) {
          a += p_t[t] * v_dim[t];
        }
        acc[d] = a;
      }

      float* out_head = out_ptr + static_cast<int64_t>(b) * dim + h * head_size;
      for (int32_t d = 0; d < head_size; ++d) {
        out_head[d] = acc[d] / sum_p;
      }
    }
  }
}
}  // namespace kernel