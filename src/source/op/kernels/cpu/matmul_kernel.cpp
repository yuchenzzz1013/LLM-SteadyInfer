#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "base/base.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale,
                       const CudaConfig* config) {
  UNUSED(config);
  CHECK(input.is_empty() == false);
  CHECK(weight.is_empty() == false);
  CHECK(output.is_empty() == false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  float* output_ptr = const_cast<float*>(output.ptr<float>());

  int32_t in_dim0 = 1;
  int32_t in_dim1 = 1;
  if (input.dims_size() == 2) {
    in_dim0 = input.get_dim(0);  // batch
    in_dim1 = input.get_dim(1);  // M
  } else if (input.dims_size() == 1) {
    in_dim0 = input.get_dim(0);
  } else {
    LOG(FATAL) << "The input tensor has a wrong dim size.";
  }

  CHECK_EQ(weight.dims_size(), 2);
  const int32_t K = weight.get_dim(0);  // output dim per sample
  const int32_t M = weight.get_dim(1);  // input dim per sample

  // The layer flattens a [batch, M] input into 1D, so a single "row" input
  // (dims_size() == 1) is ambiguous: its size is batch * M. Derive the batch
  // from the weight's input dim — this is what makes the whole batch share
  // ONE GEMM call instead of looping per sequence.
  int32_t batch = 1;
  if (input.dims_size() == 2) {
    CHECK_EQ(in_dim1, M);
    batch = in_dim0;
  } else {
    CHECK_EQ(in_dim0 % M, 0) << "input size " << in_dim0 << " is not divisible by M " << M;
    batch = in_dim0 / M;
  }
  CHECK_EQ(output.size(), static_cast<size_t>(batch) * K);

  // Batched GEMM: out[b, k] = scale * sum_m in[b, m] * W[k, m].
  // OpenMP over (b, k) rows — CPU batch GEMM without a BLAS dependency
  // (linking cblas/MKL is optional here; the row loop parallelizes cleanly
  // and the simd inner loop uses the compiler's vector units).
  const float* in_base = input_ptr;
  float* out_base = output_ptr;
#pragma omp parallel for collapse(2) schedule(static)
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t k = 0; k < K; ++k) {
      const float* in_row = in_base + static_cast<int64_t>(b) * M;
      const float* w_row = weight_ptr + static_cast<int64_t>(k) * M;
      float sum = 0.f;
#pragma omp simd
      for (int32_t m = 0; m < M; ++m) {
        sum += in_row[m] * w_row[m];
      }
      out_base[static_cast<int64_t>(b) * K + k] = sum * scale;
    }
  }
}
}  // namespace kernel