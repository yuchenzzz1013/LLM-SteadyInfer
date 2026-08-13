#ifndef SRC_INCLUDE_SAMPLER_SAMPLER_H
#define SRC_INCLUDE_SAMPLER_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
class Sampler {
 public:
  explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

  virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

  // Row-major logits [batch, row_stride]; each row's argmax is computed over
  // the first `size` entries (size <= row_stride). One kernel + one copy.
  virtual void sample_batch(const float* logits, size_t row_stride, size_t size, int32_t batch,
                            int32_t* out_tokens, void* stream = nullptr) = 0;

 protected:
  base::DeviceType device_type_;
};
}  // namespace sampler
#endif
