#ifndef SRC_INCLUDE_MODEL_QKV_SPLIT_H_
#define SRC_INCLUDE_MODEL_QKV_SPLIT_H_
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace kernel {
// De-interleave the fused-QKV GEMM output (row layout [q | k | v], row width
// dim + 2 * kv_dim) into contiguous q/k/v buffers, all raw bf16:
//   dst_q[b][i]        = src[b][i]                 for i in [0, dim)
//   dst_k[b][j]        = src[b][dim + j]           for j in [0, kv_dim)
//   dst_v[b][k]        = src[b][dim + kv_dim + k]  for k in [0, kv_dim)
// One block per batch row, 16 bytes (8 bf16 values) per thread step, so the
// whole row move is a single flat coalesced pass per row.
// Requires bf16 data with dim and kv_dim both multiples of 8 and 16B-aligned
// base pointers; model::split_fused_qkv_output falls back to the strided
// device-to-device copies otherwise.
void split_fused_qkv_bf16_cu(const void* fused_src, void* dst_q, void* dst_k, void* dst_v,
                             int32_t batch, int32_t dim, int32_t kv_dim, cudaStream_t stream);
}  // namespace kernel
#endif  // SRC_INCLUDE_MODEL_QKV_SPLIT_H_
