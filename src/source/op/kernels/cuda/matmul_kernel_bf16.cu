// CUDA BF16 GEMM (cuBLAS Tensor Core path + custom fallback).
//
// Matmul semantics: weight [K, M] row-major x input [batch, M] -> [batch, K].
// cuBLAS BF16 runs on the Tensor Cores with fp32 accumulation
// (CUBLAS_COMPUTE_32F on CUDA_R_16BF operands — products of two bf16 values
// are exact in fp32, so this is the "BF16 math, fp32 accumulate" convention
// used across the migration; identical to the other BF16 kernels here).
// The custom kernel is the pre-Ampere / error fallback: it computes the same
// products in float but stores bf16, i.e. a software simulation of BF16 math.
#include <tensor/tensor.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {

template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_bf16_custom(const __nv_bfloat16* input,
                                             const __nv_bfloat16* weight, __nv_bfloat16* output,
                                             float scale, int M, int K, int batch) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int p = blockIdx.x;  // output row (== weight row)
  if (p >= K) {
    return;
  }
  const __nv_bfloat16* weight_row = weight + static_cast<int64_t>(p) * M;

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  for (int b = 0; b < batch; ++b) {
    const __nv_bfloat16* input_row = input + static_cast<int64_t>(b) * M;
    sdata[tid] = 0;
    for (int i = tid; i < M; i += blockDim.x) {
      sdata[tid] += __bfloat162float(input_row[i]) * __bfloat162float(weight_row[i]);
    }
    __syncthreads();

    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[static_cast<int64_t>(b) * K + p] = __float2bfloat16(part_sum * scale);
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row (output dim per sample)
  const int32_t M = weight.get_dim(1);  // col (input dim per sample)

  int32_t input_size = input.size();
  int32_t output_size = output.size();
  int32_t batch = input_size / M;

  CHECK_EQ(input_size % M, 0);
  CHECK_EQ(output_size % K, 0);
  CHECK_EQ(batch, output_size / K);

  const __nv_bfloat16* input_ptr = input.ptr<__nv_bfloat16>();
  const __nv_bfloat16* weight_ptr = weight.ptr<__nv_bfloat16>();
  __nv_bfloat16* output_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());

  // Tensor-Core fast path (Ampere+): BF16 GEMM via cublasGemmEx. Same
  // transposed views as before the BF16 migration: C^T = W * A^T.
  if (config) {
    static int cc_major = -1;
    if (cc_major < 0) {
      int dev = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    }
    if (cc_major >= 8) {
      const float alpha = scale;
      const float beta = 0.f;
      cublasStatus_t st = cublasGemmEx(
          config->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, K, batch, M, &alpha, weight_ptr,
          CUDA_R_16BF, M, input_ptr, CUDA_R_16BF, M, &beta, output_ptr, CUDA_R_16BF, K,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      if (st == CUBLAS_STATUS_SUCCESS) {
        return;
      }
      LOG(WARNING) << "[MATMUL-BF16] cublasGemmEx BF16 failed (" << int(st)
                   << "); falling back to the custom kernel.";
    }
  }

  if (config && config->stream) {
    matmul_kernel_bf16_custom<128><<<K, 128, 0, config->stream>>>(
        input_ptr, weight_ptr, output_ptr, scale, M, K, batch);
  } else {
    matmul_kernel_bf16_custom<128><<<K, 128>>>(input_ptr, weight_ptr, output_ptr, scale, M, K,
                                                  batch);
  }
}
}  // namespace kernel
