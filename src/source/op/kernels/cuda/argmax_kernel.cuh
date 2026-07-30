#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
}
#endif  
