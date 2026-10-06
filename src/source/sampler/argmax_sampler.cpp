#include "sampler/argmax_sampler.h"
#include <algorithm>
#include <cstring>
#include "base/bf16_utils.h"
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
ArgmaxSampler::ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {
}

ArgmaxSampler::~ArgmaxSampler() {
  if (cuda_idx_dev_) {
    cudaFree(cuda_idx_dev_);
    cuda_idx_dev_ = nullptr;
  }
  if (cuda_idx_host_) {
    cudaFreeHost(cuda_idx_host_);
    cuda_idx_host_ = nullptr;
  }
  if (cuda_done_) {
    cudaEventDestroy(cuda_done_);
    cuda_done_ = nullptr;
  }
}

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  CHECK(device_type_ == base::DeviceType::kDeviceCPU)
      << "FP32 argmax sampling only exists on the CPU device; CUDA models keep "
         "logits in BF16 and sample through sample_bf16().";
  size_t next = std::distance(logits, std::max_element(logits, logits + size));
  return next;
}

void ArgmaxSampler::sample_batch(const float* logits, size_t row_stride, size_t size,
                                 int32_t batch, int32_t* out_tokens, void* stream) {
  CHECK(device_type_ == base::DeviceType::kDeviceCPU)
      << "FP32 argmax sampling only exists on the CPU device; CUDA models keep "
         "logits in BF16 and sample through sample_batch_bf16().";
  for (int32_t b = 0; b < batch; ++b) {
    const float* row = logits + b * row_stride;
    out_tokens[b] = static_cast<int32_t>(
        std::distance(row, std::max_element(row, row + size)));
  }
}

size_t ArgmaxSampler::sample_bf16(const uint16_t* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    // CPU models run FP32 (weights converted at load time), so this is never
    // reached in practice; still handle it by widening each element exactly.
    if (size == 0) {
      return 0;
    }
    size_t max_index = 0;
    float max_value = base::bf16_to_fp32(logits[0]);
    for (size_t i = 1; i < size; ++i) {
      const float v = base::bf16_to_fp32(logits[i]);
      if (v > max_value) {
        max_value = v;
        max_index = i;
      }
    }
    return max_index;
  }
  int32_t token = 0;
  // A single row: the row stride is irrelevant (block 0 reads from the start).
  sample_bf16_on_stream(logits, size, size, 1, &token, stream);
  return static_cast<size_t>(token);
}

void ArgmaxSampler::sample_batch_bf16(const uint16_t* logits, size_t row_stride, size_t size,
                                      int32_t batch, int32_t* out_tokens, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    for (int32_t b = 0; b < batch; ++b) {
      const uint16_t* row = logits + static_cast<size_t>(b) * row_stride;
      if (size == 0) {
        out_tokens[b] = 0;
        continue;
      }
      size_t max_index = 0;
      float max_value = base::bf16_to_fp32(row[0]);
      for (size_t i = 1; i < size; ++i) {
        const float v = base::bf16_to_fp32(row[i]);
        if (v > max_value) {
          max_value = v;
          max_index = i;
        }
      }
      out_tokens[b] = static_cast<int32_t>(max_index);
    }
    return;
  }
  sample_bf16_on_stream(logits, row_stride, size, batch, out_tokens, stream);
}

void ArgmaxSampler::sample_bf16_on_stream(const uint16_t* logits, size_t row_stride, size_t size,
                                          int32_t batch, int32_t* out_tokens, void* stream) {
  if (batch <= 0) {
    return;
  }
  ensure_cuda_staging(batch);

  // The argmax kernel runs on the caller's stream — the same stream the LM
  // head wrote the logits on, so no cross-stream ordering (and no full-stream
  // drain) is needed before it. Previously this ran on the legacy default
  // stream and the caller had to cudaStreamSynchronize the model stream
  // first.
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
  kernel::argmax_kernel_launch(logits, row_stride, size, batch, cuda_idx_dev_, stream_);

  const size_t bytes = static_cast<size_t>(batch) * sizeof(int32_t);
  // Pinned D2H: one DMA straight into page-locked memory. With a pageable
  // target the driver would first copy through its own bounce buffer with the
  // calling thread blocked.
  CHECK_EQ(cudaMemcpyAsync(cuda_idx_host_, cuda_idx_dev_, bytes, cudaMemcpyDeviceToHost, stream_),
           cudaSuccess);
  // Wait for the sampling only — the event completes once every operation
  // enqueued on the stream before it (the forward pass, the argmax, this
  // copy) has finished, without draining unrelated later work.
  CHECK_EQ(cudaEventRecord(cuda_done_, stream_), cudaSuccess);
  CHECK_EQ(cudaEventSynchronize(cuda_done_), cudaSuccess);
  std::memcpy(out_tokens, cuda_idx_host_, bytes);
}

void ArgmaxSampler::ensure_cuda_staging(int32_t batch) {
  if (static_cast<size_t>(batch) <= cuda_capacity_) {
    return;
  }
  if (cuda_idx_dev_) {
    cudaFree(cuda_idx_dev_);
    cuda_idx_dev_ = nullptr;
  }
  if (cuda_idx_host_) {
    cudaFreeHost(cuda_idx_host_);
    cuda_idx_host_ = nullptr;
  }
  const size_t bytes = static_cast<size_t>(batch) * sizeof(int32_t);
  const cudaError_t dev_err = cudaMalloc(reinterpret_cast<void**>(&cuda_idx_dev_), bytes);
  CHECK_EQ(dev_err, cudaSuccess) << "argmax staging: cudaMalloc(" << bytes
                                 << ") failed: " << cudaGetErrorString(dev_err);
  const cudaError_t host_err = cudaHostAlloc(reinterpret_cast<void**>(&cuda_idx_host_), bytes,
                                             cudaHostAllocDefault);
  CHECK_EQ(host_err, cudaSuccess) << "argmax staging: cudaHostAlloc(" << bytes
                                  << ") failed: " << cudaGetErrorString(host_err);
  if (cuda_done_ == nullptr) {
    const cudaError_t ev_err = cudaEventCreateWithFlags(&cuda_done_, cudaEventDisableTiming);
    CHECK_EQ(ev_err, cudaSuccess)
        << "argmax staging: cudaEventCreate failed: " << cudaGetErrorString(ev_err);
  }
  cuda_capacity_ = static_cast<size_t>(batch);
}
}  // namespace sampler
