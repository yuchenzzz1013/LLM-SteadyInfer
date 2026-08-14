#include <tensor/tensor.h>
#include <cuda_runtime_api.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
// One thread block per output row (weight row), looping over the batch.
// Previously one block was launched per (batch element, output row), which
// re-read the whole weight matrix once per batch element. For the LM head
// (batch=64, vocab=151936) that was ~9.7M tiny blocks and ~97GB of weight
// traffic per decode step; now the weight is streamed exactly once.
template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K, int batch) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int p = blockIdx.x;  // output row (== weight row)
  if (p >= K) {
    return;
  }
  const float* weight_row = weight + p * M;

  constexpr int pack_size = 4;
  // Only use the vectorized path when rows are 16B aligned (M % 4 == 0).
  const int pack_num = (M % pack_size == 0) ? M / pack_size : 0;
  const int pack_off = pack_size * pack_num;

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  for (int b = 0; b < batch; ++b) {
    const float* input_row = input + b * M;
    sdata[tid] = 0;

    if (pack_num > 0) {
      const float4* input_float4_ptr = reinterpret_cast<const float4*>(input_row);
      const float4* weight_float4_ptr = reinterpret_cast<const float4*>(weight_row);
#pragma unroll
      for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 input_float4 = *(input_float4_ptr + i);
        float4 weight_float4 = *(weight_float4_ptr + i);
        float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                         input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
        sdata[tid] += part_sum;
      }
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input_row[i] * weight_row[i];
    }

    __syncthreads();

    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[b * K + p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K, int batch) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int p = blockIdx.x;  // output row (== weight row)
  if (p >= K) {
    return;
  }
  const int8_t* weight_row = weight + p * M;
  const float* scale_row = scales + (p * M) / group_size;

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  for (int b = 0; b < batch; ++b) {
    const float* input_row = input + b * M;
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      sdata[tid] += input_row[i] * scale_row[i / group_size] *
                    static_cast<float>(weight_row[i]);
    }
    __syncthreads();

    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[b * K + p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row (output dim per sample)
  const int32_t M = weight.get_dim(1);  // col (input dim per sample)

  // Compute batch from input size: input can be [M] or [batch, M]
  int32_t input_size = input.size();
  int32_t output_size = output.size();
  int32_t batch = input_size / M;

  CHECK_EQ(input_size % M, 0);
  CHECK_EQ(output_size % K, 0);
  CHECK_EQ(batch, output_size / K);

  // Tensor-Core fast path (Ampere+): FP32 GEMM through TF32 via cublasGemmEx.
  // Storage is row-major [batch, M] x [K, M] -> [batch, K]; reinterpreted as
  // column-major that becomes C^T = W * A^T:
  //   A' = weight [K, M] (col-major view = [M, K], lda = M), op = transpose
  //   B' = input  [batch, M] (col-major view = [M, batch], ldb = M), op = N
  //   C' = output [batch, K] (col-major view = [K, batch], ldc = K)
  // The TF32 compute type runs on the tensor cores; the hand-written kernel
  // below stays as the fallback for pre-Ampere devices and cublas errors.
  if (config && input.data_type() == base::DataType::kDataTypeFp32 &&
      weight.data_type() == base::DataType::kDataTypeFp32) {
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
          config->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, K, batch, M, &alpha,
          weight.ptr<float>(), CUDA_R_32F, M, input.ptr<float>(), CUDA_R_32F, M, &beta,
          const_cast<float*>(output.ptr<float>()), CUDA_R_32F, K, CUBLAS_COMPUTE_32F_FAST_TF32,
          CUBLAS_GEMM_DEFAULT);
      if (st == CUBLAS_STATUS_SUCCESS) {
        return;
      }
      LOG(WARNING) << "[MATMUL] cublasGemmEx TF32 failed (" << int(st)
                   << "); falling back to the custom kernel.";
    }
  }

  if (config && config->stream) {
    matmul_kernel_cu_fp32<128><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), M, K,
        batch);
  } else {
    matmul_kernel_cu_fp32<128><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                           const_cast<float*>(output.ptr<float>()), M, K, batch);
  }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col

  int32_t input_size = input.size();
  int32_t output_size = output.size();
  int32_t batch = input_size / M;

  CHECK_EQ(input_size % M, 0);
  CHECK_EQ(output_size % K, 0);
  CHECK_EQ(batch, output_size / K);

  int32_t total_rows = batch * K;
  UNUSED(total_rows);
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  if (config->stream) {
    matmul_kernel_cu_fp32int8<128><<<K, 128, 0, config->stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K, batch);
  } else {
    matmul_kernel_cu_fp32int8<128><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                               scale.ptr<float>(), group_size,
                                               const_cast<float*>(output.ptr<float>()), M, K,
                                               batch);
  }
}
}  // namespace kernel