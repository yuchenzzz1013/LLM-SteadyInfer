// CUDA embedding gather (BF16): copies embedding rows as raw bf16 elements.
#include <base/cuda_check.h>
#include <cuda_bf16.h>
#include "emb_kernel.cuh"
namespace kernel {
__global__ void emb_kernel_bf16_kernel(int32_t vocab_size, int32_t token_num,
                                          int32_t weight_dim, const int32_t* input_ptr,
                                          const __nv_bfloat16* weight_ptr,
                                          __nv_bfloat16* output_ptr) {
  int32_t token_idx = blockIdx.x;
  if (token_idx >= token_num) {
    return;
  }
  int32_t token = input_ptr[token_idx];
  if (token >= vocab_size) {
    return;
  }

  __nv_bfloat16* output_ptr_start = output_ptr + static_cast<int64_t>(token_idx) * weight_dim;
  const __nv_bfloat16* weight_ptr_start = weight_ptr + static_cast<int64_t>(token) * weight_dim;

  for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
    output_ptr_start[i] = weight_ptr_start[i];
  }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, int32_t vocab_size, void* stream) {
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);
  CHECK(weight.device_type() == output.device_type());
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  constexpr int32_t thread_num = 128;
  int32_t grid_size = input_num;
  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);

  // Host token ids (prefill / mixed path): upload them into a pooled device
  // buffer on the caller's stream. The previous clone() + to_cuda() pair
  // allocated a host copy, copied host-to-host, issued the H2D on the legacy
  // default stream and freed the copy — three host-side operations per step
  // for a few KB of ids, with the calling thread blocked through the upload.
  // The source must stay alive until the stream drains, the same contract as
  // forward_batch's other H2D staging copies (the device buffer is released to
  // the pool immediately, but any reuse is stream-ordered after this kernel).
  const int32_t* in_ptr = input.ptr<int32_t>();
  if (input.device_type() != base::DeviceType::kDeviceCUDA) {
    tensor::Tensor tokens_cu(base::DataType::kDataTypeInt32, input_num, true,
                             base::CUDADeviceAllocatorFactory::get_instance());
    CHECK(tokens_cu.ptr<int32_t>() != nullptr)
        << "Failed to stage the embedding input tokens on the device";
    const size_t bytes = static_cast<size_t>(input_num) * sizeof(int32_t);
    if (stream_) {
      CHECK_EQ(cudaMemcpyAsync(tokens_cu.ptr<int32_t>(), input.ptr<int32_t>(), bytes,
                               cudaMemcpyHostToDevice, stream_),
               cudaSuccess);
    } else {
      CHECK_EQ(cudaMemcpy(tokens_cu.ptr<int32_t>(), input.ptr<int32_t>(), bytes,
                          cudaMemcpyHostToDevice),
               cudaSuccess);
    }
    in_ptr = tokens_cu.ptr<int32_t>();
  }
  const __nv_bfloat16* wei_ptr = weight.ptr<__nv_bfloat16>();
  __nv_bfloat16* out_ptr = const_cast<__nv_bfloat16*>(output.ptr<__nv_bfloat16>());
  if (stream_) {
    emb_kernel_bf16_kernel<<<grid_size, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    CUDA_KERNEL_CHECK();
  } else {
    emb_kernel_bf16_kernel<<<grid_size, thread_num>>>(vocab_size, input_num, weight_dim,
                                                          in_ptr, wei_ptr, out_ptr);
    CUDA_KERNEL_CHECK();
  }
}
}  // namespace kernel
