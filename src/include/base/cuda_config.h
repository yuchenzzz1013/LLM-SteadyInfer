#ifndef SRC_INCLUDE_BASE_CUDA_CONFIG_H
#define SRC_INCLUDE_BASE_CUDA_CONFIG_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;

  // Lazily-created cuBLAS handle bound to `stream`, used by the Tensor-Core
  // GEMM path (cublasGemmEx with TF32 / FP16 compute types). Created on first
  // use so pure-CPU and graph-free paths never pay for it.
  cublasHandle_t cublas_handle() const {
    if (cublas_handle_ == nullptr) {
      cublasCreate(&cublas_handle_);
      if (stream) {
        cublasSetStream(cublas_handle_, stream);
      }
    }
    return cublas_handle_;
  }

  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
    if (cublas_handle_) {
      cublasDestroy(cublas_handle_);
    }
  }

 private:
  mutable cublasHandle_t cublas_handle_ = nullptr;
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
