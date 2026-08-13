#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}


__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim,
                                            int32_t kv_head_num, int32_t head_num,
                                            int32_t head_size, int64_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  // 预加载query到共享内存
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits
  // (e.g. 36 heads / 8 KV heads) correctly and stays within kv_dim, unlike
  // head / (head_num / kv_head_num).
  int head_offset = (head * kv_head_num / head_num) * head_size;
  // 计算自注意力分数
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;

    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }

    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 使用自注意力分数对value矩阵加权
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                   const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                   CudaConfig* config) {
  UNUSED(device_type);
  // int64: large external caches (num_slots * max_seq_len) can overflow int32.
  int64_t layer_offset = static_cast<int64_t>(layer_index) * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());

  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_head_num, head_num,
      head_size, layer_offset);
}

__global__ void multi_head_attention_kernel_batch(
    const int32_t* positions, const int32_t* kv_offsets, int32_t num_slots, int32_t max_seq_len,
    int32_t layer_idx, int32_t dim, float* query, float* score, float* output, float* key_cache,
    float* value_cache, int32_t kv_dim, int32_t kv_head_num, int32_t head_num, int32_t head_size) {
  int head = blockIdx.x;
  int b = blockIdx.y;
  if (head >= head_num) {
    return;
  }
  int pos = positions[b];
  int slot = kv_offsets[b];

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + (size_t)b * dim + head * head_size;

  // Preload this head's query vector into shared memory.
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score + (size_t)(b * head_num + head) * max_seq_len;
  // Same GQA mapping as the single-sequence kernel (see above).
  int head_offset = (head * kv_head_num / head_num) * head_size;
  int64_t cache_base = static_cast<int64_t>(layer_idx) * num_slots * max_seq_len * kv_dim +
                       static_cast<int64_t>(slot) * max_seq_len * kv_dim;

  // Attention scores q·k^T for t in [0, pos] (implicit causal mask).
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + cache_base + t * kv_dim + head_offset;

    float score_val = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);

      score_val += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
                   key_val.w * query_val.w;
    }

    score_head[t] = score_val * scale;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + (size_t)b * dim + head * head_size;
  // Weighted sum of values with the attention scores.
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + cache_base + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
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
  float* query = const_cast<float*>(query_batch.ptr<float>());
  float* score = const_cast<float*>(score_batch.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());
  float* kcache = const_cast<float*>(key_cache.ptr<float>());
  float* vcache = const_cast<float*>(value_cache.ptr<float>());

  dim3 grid(head_num, batch);
  cudaStream_t stream = config->stream;
  multi_head_attention_kernel_batch<<<grid, thread_num, head_size * sizeof(float), stream>>>(
      positions.ptr<int32_t>(), kv_offsets.ptr<int32_t>(), num_slots, max_seq_len, layer_idx, dim,
      query, score, output, kcache, vcache, kv_dim, kv_head_num, head_num, head_size);
}

}  // namespace kernel