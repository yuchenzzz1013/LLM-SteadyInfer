#ifndef SRC_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H
#define SRC_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H
#include <cstdint>
#include <cuda_runtime_api.h>
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type);
  ~ArgmaxSampler() override;

  size_t sample(const float* logits, size_t size, void* stream) override;

  void sample_batch(const float* logits, size_t row_stride, size_t size, int32_t batch,
                    int32_t* out_tokens, void* stream) override;

  size_t sample_bf16(const uint16_t* logits, size_t size, void* stream) override;

  void sample_batch_bf16(const uint16_t* logits, size_t row_stride, size_t size, int32_t batch,
                         int32_t* out_tokens, void* stream) override;

 private:
  // CUDA sample path: enqueue the argmax on `stream` (the model stream that
  // produced the logits), DMA the indices back into page-locked host memory,
  // and wait on that transfer's event only. Fills `out_tokens` with `batch`
  // tokens. Grows the persistent staging buffers on demand.
  void sample_bf16_on_stream(const uint16_t* logits, size_t row_stride, size_t size, int32_t batch,
                             int32_t* out_tokens, void* stream);
  void ensure_cuda_staging(int32_t batch);

  // Persistent CUDA staging: device argmax output, its page-locked host
  // target, and the completion event. Reused across steps so the hot path
  // neither allocates nor page-locks; a sampler samples one call at a time
  // (single scheduler thread per model).
  int32_t* cuda_idx_dev_ = nullptr;   // device: batch x int32 argmax indices
  int32_t* cuda_idx_host_ = nullptr;  // pinned host: D2H target
  size_t cuda_capacity_ = 0;          // elements the two buffers hold
  cudaEvent_t cuda_done_ = nullptr;   // set once the D2H transfer has landed
};
}  // namespace sampler
#endif
