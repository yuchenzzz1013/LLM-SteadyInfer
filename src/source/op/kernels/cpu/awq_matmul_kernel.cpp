#include "awq_matmul_kernel.h"
#include "base/base.h"
namespace kernel {
namespace {
// AWQ's "gemm" packing stores eight 4-bit codes per int32; output column i of
// the packed group sits at bit 4 * ((i >> 1) + 4 * (i & 1)). Zero points use
// the same interleaving.
inline int awq_unpack(uint32_t packed, int i) {
  return static_cast<int>((packed >> (4 * (i >> 1) + 16 * (i & 1))) & 0xFu);
}
}  // namespace

void awq_matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                           const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                           const tensor::Tensor& output, const AwqMatmulParams& params,
                           const CudaConfig* config) {
  UNUSED(config);
  CHECK(!input.is_empty() && !qweight.is_empty() && !scales.is_empty() && !output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(qweight.device_type() == base::DeviceType::kDeviceCPU);
  CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

  const int32_t K = qweight.get_dim(0);           // input dim per sample
  const int32_t packed_total = qweight.get_dim(1);  // packed output columns
  const int32_t N = params.out_features;
  const int32_t total_out = scales.get_dim(1);
  const int32_t group_size = params.group_size;
  const bool zero_point = params.zero_point && !qzeros.is_empty();
  CHECK_GT(K, 0);
  CHECK_GT(N, 0);
  CHECK_EQ(N % 8, 0);
  CHECK_EQ(params.out_offset % 8, 0);
  CHECK_EQ(input.size() % K, 0);

  const int32_t batch = static_cast<int32_t>(input.size() / K);
  const int32_t c0 = params.out_offset / 8;
  CHECK_LE(c0 + N / 8, packed_total);
  CHECK_LE(params.out_offset + N, total_out);
  CHECK_EQ(scales.get_dim(0), K / group_size);

  const float* in_ptr = input.ptr<float>();
  const float* scales_ptr = scales.ptr<float>();
  const int32_t* qw_ptr = qweight.ptr<int32_t>();
  const int32_t* qz_ptr = zero_point ? qzeros.ptr<int32_t>() : nullptr;
  float* out_ptr = const_cast<float*>(output.ptr<float>());

  // One packed column (eight output rows) per inner block: the int32 and its
  // zero point are decoded once and reused by the eight accumulators.
#pragma omp parallel for collapse(2) schedule(static)
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t c = 0; c < N / 8; ++c) {
      const int32_t n0 = c * 8;
      const int32_t pc = c0 + c;
      float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const float* in_row = in_ptr + static_cast<int64_t>(b) * K;
      int32_t g = -1;
      int32_t zero[8];
      float scale[8];
      for (int32_t k = 0; k < K; ++k) {
        if (k / group_size != g) {
          g = k / group_size;
          const uint32_t zp =
              zero_point ? static_cast<uint32_t>(qz_ptr[static_cast<int64_t>(g) * packed_total + pc])
                         : 0u;
          for (int i = 0; i < 8; ++i) {
            zero[i] = zero_point ? awq_unpack(zp, i) : 8;
            scale[i] = scales_ptr[static_cast<int64_t>(g) * total_out + params.out_offset + n0 + i];
          }
        }
        const uint32_t packed =
            static_cast<uint32_t>(qw_ptr[static_cast<int64_t>(k) * packed_total + pc]);
        const float x = in_row[k];
        for (int i = 0; i < 8; ++i) {
          sum[i] += static_cast<float>(awq_unpack(packed, i) - zero[i]) * scale[i] * x;
        }
      }
      for (int i = 0; i < 8; ++i) {
        out_ptr[static_cast<int64_t>(b) * N + n0 + i] = sum[i];
      }
    }
  }
}
}  // namespace kernel
