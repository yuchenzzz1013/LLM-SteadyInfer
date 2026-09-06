#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "base/bf16_utils.h"
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
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
  } else {
    return kernel::argmax_kernel_cu(logits, size, stream);
  }
}

void ArgmaxSampler::sample_batch_bf16(const uint16_t* logits, size_t row_stride, size_t size,
                                      int32_t batch, int32_t* out_tokens, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    for (int32_t b = 0; b < batch; ++b) {
      const uint16_t* row = logits + static_cast<size_t>(b) * row_stride;
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
  } else {
    kernel::argmax_kernel_cu_batch(logits, row_stride, size, batch, out_tokens, stream);
  }
}
}  // namespace sampler