#include "rmsnorm_kernel.h"
#include <cmath>

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  const int32_t dim = static_cast<int32_t>(input.size());

  const float eps = 1e-6f;

  float sum_sq = 0.f;
#pragma omp simd reduction(+ : sum_sq)
  for (int32_t i = 0; i < dim; ++i) {
    sum_sq += in_ptr[i] * in_ptr[i];
  }
  const float rsqrt = 1.f / std::sqrt(sum_sq / dim + eps);

#pragma omp simd
  for (int32_t i = 0; i < dim; ++i) {
    out_ptr[i] = wei_ptr[i] * (rsqrt * in_ptr[i]);
  }
}

// Batched RMSNorm over the last dim: input [rows, dim] (or any leading shape
// flattened), one OpenMP task per row. Used by the continuous-batching CPU
// path — the single-tensor kernel above normalizes the whole buffer as ONE
// row, which would be wrong for [batch, dim].
void rmsnorm_kernel_cpu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t dim, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);
  CHECK_EQ(input.size() % dim, 0);
  CHECK_EQ(output.size(), input.size());

  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  const int32_t rows = static_cast<int32_t>(input.size() / dim);
  const float eps = 1e-6f;

#pragma omp parallel for schedule(static)
  for (int32_t r = 0; r < rows; ++r) {
    const float* in_row = in_ptr + static_cast<int64_t>(r) * dim;
    float* out_row = out_ptr + static_cast<int64_t>(r) * dim;
    float sum_sq = 0.f;
#pragma omp simd reduction(+ : sum_sq)
    for (int32_t i = 0; i < dim; ++i) {
      sum_sq += in_row[i] * in_row[i];
    }
    const float rsqrt = 1.f / std::sqrt(sum_sq / dim + eps);
#pragma omp simd
    for (int32_t i = 0; i < dim; ++i) {
      out_row[i] = wei_ptr[i] * (rsqrt * in_row[i]);
    }
  }
}
}  // namespace kernel