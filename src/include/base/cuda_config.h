#ifndef SRC_INCLUDE_BASE_CUDA_CONFIG_H
#define SRC_INCLUDE_BASE_CUDA_CONFIG_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
