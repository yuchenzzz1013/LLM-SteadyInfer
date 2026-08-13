#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
  }
}

void ArgmaxSampler::sample_batch(const float* logits, size_t row_stride, size_t size,
                                 int32_t batch, int32_t* out_tokens, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    for (int32_t b = 0; b < batch; ++b) {
      const float* row = logits + b * row_stride;
      out_tokens[b] = static_cast<int32_t>(
          std::distance(row, std::max_element(row, row + size)));
    }
  } else {
    kernel::argmax_kernel_cu_batch(logits, row_stride, size, batch, out_tokens, stream);
  }
}
}  // namespace sampler