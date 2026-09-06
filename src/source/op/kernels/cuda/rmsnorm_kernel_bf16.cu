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
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  const __nv_bfloat16* in_ptr = input.ptr<__nv_bfloat16>();
  const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
  __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
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
