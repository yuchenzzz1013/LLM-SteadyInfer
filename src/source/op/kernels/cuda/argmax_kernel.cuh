#ifndef SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#define SRC_SOURCE_OP_KERNELS_CUDA_ARGMAX_KERNEL_CUH
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
namespace kernel {
// Batched argmax over BF16 logits (raw uint16_t bfloat16 bit patterns, widened
// to float before comparing — exact, since every bfloat16 value is exactly
// representable in float32).
//
// Row-major input [batch, row_stride]: each row's argmax is computed over the
// first `size` entries (size <= row_stride), so sampling can be restricted to
// the tokenizer vocab even when the logits row is wider. Ties pick the lowest
// index; NaN entries never win (as with a plain float compare).
//
// One block per row, vectorized loads — see the kernel for the parallel
// decomposition. `out_idx` is a device buffer of `batch` int32s owned by the
// caller; this only enqueues on `stream` (no allocation, no copy, no sync), so
// the caller controls the ordering and the wait.
void argmax_kernel_launch(const uint16_t* logits, size_t row_stride, size_t size, int32_t batch,
                          int32_t* out_idx, cudaStream_t stream);
}  // namespace kernel
#endif
