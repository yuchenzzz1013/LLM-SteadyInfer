// CUDA RMSNorm (BF16): bf16 in/weight/out, math in float with bf16-rounded
// operands and products (exact in float), fp32 accumulation — the same
// semantics as the cuBLAS BF16 path and the fp32 CPU reference, so the
// BF16/FP32 numerical gap stays within bf16 rounding error.
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {

template <int32_t BLOCK_DIM>
__global__ void row_rmsnorm_bf16(const __nv_bfloat16* in, const __nv_bfloat16* wei,
                                 __nv_bfloat16* out, int size, float eps) {
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < size; i += BLOCK_DIM) {
    const float x = __bfloat162float(in[i]);
    sum += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += BLOCK_DIM) {
    const float x = __bfloat162float(in[i]);
    const float w = __bfloat162float(wei[i]);
    out[i] = __float2bfloat16(scale * x * w);
  }
}

// Multi-row form (dim1 = last dim): one block per row of `size` elements.
__global__ void row_rmsnorm_bf16_dim(const __nv_bfloat16* in, const __nv_bfloat16* wei,
                                     __nv_bfloat16* out, int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }
  const __nv_bfloat16* block_in = in + static_cast<int64_t>(bid) * size;
  __nv_bfloat16* block_out = out + static_cast<int64_t>(bid) * size;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    const float x = __bfloat162float(block_in[i]);
    sum += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  for (int i = tid; i < size; i += blockDim.x) {
    const float x = __bfloat162float(block_in[i]);
    const float w = __bfloat162float(wei[i]);
    block_out[i] = __float2bfloat16(scale * x * w);
  }
}

// Head-dim specialization: one warp per 128-element row, 4 contiguous dims per
// lane (one 8-byte load for the row, one for the weight), and a 5-step
// shfl_xor reduction instead of a cub BlockReduce. The generic kernel above
// costs a 128-thread block, a smem round trip and two __syncthreads per row,
// which is pure overhead when a row is only 128 elements: Qwen3's q_norm runs
// on [batch * head_num, 128] (1024 rows at batch 32, 16384 at batch 512) and
// k_norm on [batch * kv_head_num, 128] every layer, next to the 2560-wide
// hidden norms that keep the generic path. Element math is unchanged
// (scale * x * w with bf16 operands in fp32, one bf16 rounding at the end);
// only the order of the fp32 square sum differs.
__global__ void row_rmsnorm_bf16_dim128_warp(const __nv_bfloat16* __restrict__ in,
                                             const __nv_bfloat16* __restrict__ wei,
                                             __nv_bfloat16* __restrict__ out, int32_t rows,
                                             float eps) {
  constexpr int kVec = 4;  // 32 lanes * 4 dims = one 128-element row
  const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (row >= rows) {
    return;
  }
  const int d0 = (threadIdx.x & 31) * kVec;
  const __nv_bfloat16* rin = in + static_cast<int64_t>(row) * 128 + d0;
  const uint2 raw = *reinterpret_cast<const uint2*>(rin);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  const float x[kVec] = {__bfloat162float(h[0].x), __bfloat162float(h[0].y),
                         __bfloat162float(h[1].x), __bfloat162float(h[1].y)};
  float sum = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, off);
  }
  const float scale = rsqrtf(sum / 128.f + eps);
  const uint2 wraw = *reinterpret_cast<const uint2*>(wei + d0);
  const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wraw);
  __nv_bfloat162 o[2];
  o[0] = __floats2bfloat162_rn(scale * x[0] * __bfloat162float(wh[0].x),
                               scale * x[1] * __bfloat162float(wh[0].y));
  o[1] = __floats2bfloat162_rn(scale * x[2] * __bfloat162float(wh[1].x),
                               scale * x[3] * __bfloat162float(wh[1].y));
  *reinterpret_cast<uint2*>(out + static_cast<int64_t>(row) * 128 + d0) =
      *reinterpret_cast<const uint2*>(o);
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t size = static_cast<int32_t>(input.size());
  const __nv_bfloat16* in_ptr = input.ptr<__nv_bfloat16>();
  const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
  __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_bf16<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_bf16<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  // Row length is the layer dim (as in the CPU kernel), not the tensor's last
  // dim: the two agree for the [rows, dim] shapes the layer checks for, but
  // only the former stays correct for any rank.
  CHECK_GT(dim, 0);
  const int32_t total_size = static_cast<int32_t>(input.size());
  CHECK_EQ(total_size % dim, 0);
  CHECK_EQ(weight.size(), static_cast<int64_t>(dim))
      << "RMSNorm weight must cover exactly one row of `dim` elements.";
  const int32_t size = dim;
  const int32_t dim_size = total_size / dim;

  const __nv_bfloat16* in_ptr = input.ptr<__nv_bfloat16>();
  const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
  __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
  if (size == 128) {
    // Per-head Q/K norm (see row_rmsnorm_bf16_dim128_warp): 8 rows per block.
    constexpr int threads_num = 256;
    const int blocks = (dim_size + 7) / 8;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_bf16_dim128_warp<<<blocks, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                                        dim_size, eps);
    } else {
      row_rmsnorm_bf16_dim128_warp<<<blocks, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                            eps);
    }
    return;
  }
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_bf16_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr,
                                                                dim_size, size, eps);
  } else {
    row_rmsnorm_bf16_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}  // namespace kernel
