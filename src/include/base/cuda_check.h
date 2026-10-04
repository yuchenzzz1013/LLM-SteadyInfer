#ifndef SRC_INCLUDE_BASE_CUDA_CHECK_H_
#define SRC_INCLUDE_BASE_CUDA_CHECK_H_
#include <cuda_runtime_api.h>
#include <glog/logging.h>

// Kernel-launch error check for development builds.
//
// A launch with an invalid configuration (grid/block out of range, dynamic
// smem above the kernel's limit, too many registers for the requested block)
// fails immediately but leaves no trace unless someone asks: the failure is
// stored as the thread's last error and the next cudaGetLastError() returns
// it. Without this check a bad launch looks exactly like success — the
// kernels downstream read stale partials and the model returns wrong tokens
// instead of an error.
//
// CUDA_KERNEL_CHECK() must be called right after a <<< >>> launch (before any
// other CUDA call, which would consume or overwrite the error). It is
// compiled out when NDEBUG is set, so release builds keep zero launch-path
// overhead; debug / RelWithDebInfo-without-NDEBUG builds abort at the failing
// launch with the driver's error string.
#ifdef NDEBUG
#define CUDA_KERNEL_CHECK() \
  do {                      \
  } while (0)
#else
#define CUDA_KERNEL_CHECK()                                                          \
  do {                                                                               \
    const cudaError_t cuda_kernel_err__ = cudaGetLastError();                        \
    CHECK_EQ(cuda_kernel_err__, cudaSuccess)                                         \
        << "CUDA kernel launch failed: " << cudaGetErrorString(cuda_kernel_err__);   \
  } while (0)
#endif

#endif  // SRC_INCLUDE_BASE_CUDA_CHECK_H_
