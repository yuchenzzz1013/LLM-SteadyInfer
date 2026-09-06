#ifndef SRC_INCLUDE_BASE_BF16_UTILS_H_
#define SRC_INCLUDE_BASE_BF16_UTILS_H_
// BF16 (bfloat16) conversion helpers for the host side.
//
// The CPU representation of a BF16 value is a raw IEEE-754 bfloat16 bit
// pattern stored in a uint16_t (the same layout HF safetensors "BF16" uses,
// little-endian). CUDA kernels use __nv_bfloat16 from <cuda_bf16.h> instead;
// the two representations are bit-identical, so a uint16_t buffer can be
// memcpy'd onto the device and read back as __nv_bfloat16 without any
// conversion.
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace base {

using bf16_t = uint16_t;

// fp32 -> bf16 with round-to-nearest-even on the discarded low 16 bits
// (numerically equivalent to __float2bfloat16 in <cuda_bf16.h>).
inline bf16_t fp32_to_bf16(float value) {
  uint32_t u = 0;
  std::memcpy(&u, &value, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;  // round bit guard for ties-to-even
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// bf16 -> fp32 (exact: zero-extend the mantissa).
inline float bf16_to_fp32(bf16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline void fp32_to_bf16_batch(const float* src, bf16_t* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = fp32_to_bf16(src[i]);
  }
}

inline void bf16_to_fp32_batch(const bf16_t* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = bf16_to_fp32(src[i]);
  }
}

}  // namespace base
#endif  // SRC_INCLUDE_BASE_BF16_UTILS_H_
