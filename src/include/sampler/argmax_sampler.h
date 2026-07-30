#ifndef SRC_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H
#define SRC_INCLUDE_SAMPLER_ARGMAX_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

  size_t sample(const float* logits, size_t size, void* stream) override;
};
}  // namespace sampler
#endif 
