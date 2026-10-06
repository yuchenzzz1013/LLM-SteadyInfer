#include <base/cuda_check.h>
#include "argmax_kernel.cuh"
#include <cfloat>
#include <cstdint>

namespace kernel {
__forceinline__ __device__ void warp_reduce_argmax(float& val, size_t& ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX) continue;
    if (tmp_val > val) {
      val = tmp_val;
      ptr = tmp_ptr;
    } else if (tmp_val == val && tmp_ptr < ptr) {
      ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}

// Widen the upper 16 bits of the bit pattern: bfloat16 is the top half of
// IEEE float32, so the result is the exact float value of the bfloat16.
__forceinline__ __device__ float bf16_bits_to_fp32(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = static_cast<uint32_t>(bits) << 16;
  return conv.f;
}

// Keep the (value, lowest index) maximum seen so far. A NaN value compares
// false both ways, so it is skipped — same as the plain `v > max` compare this
// replaced. `best_idx` seeds to SIZE_MAX so the first real value always wins.
__forceinline__ __device__ void argmax_update(uint16_t bits, size_t idx, float& best,
                                              size_t& best_idx) {
  const float v = bf16_bits_to_fp32(bits);
  if (v > best || (v == best && idx < best_idx)) {
    best = v;
    best_idx = idx;
  }
}

constexpr int kArgmaxThreads = 1024;
constexpr int kArgmaxVecElems = 8;  // one uint4 = 8 x bf16

// One block per row. Each thread streams a strided slice of the row through
// 8-wide vector loads, then the block reduces (value, index) pairs with the
// same deterministic tie-break as before (lowest index wins).
//
// The previous shape — 512 threads, one scalar 2-byte load per iteration,
// ~297 serial iterations per thread — was latency-bound at ~170us per step
// regardless of batch size: the row's 300 KB were read by a single block
// issuing 64 B per warp instruction, with the dependent max chain exposing
// roughly one DRAM round trip per iteration. Vectorized loads plus 1024
// threads cut the iteration count ~16x and the instruction count 8x.
__global__ void argmax_kernel_bf16(const uint16_t* __restrict__ logits, size_t row_stride,
                                   size_t size, int32_t* __restrict__ out_idx) {
  __shared__ size_t shared_max_ptr[32];
  __shared__ float shared_max_value[32];
  const uint16_t* row = logits + static_cast<size_t>(blockIdx.x) * row_stride;

  float best = -FLT_MAX;
  size_t best_idx = SIZE_MAX;
  size_t done = 0;  // elements covered by the vector loop

  // Vector path: 8 bf16 per thread per iteration. Needs a 16B-aligned row
  // start (row_stride is the config vocab, usually a multiple of 8 — but not
  // guaranteed, so misaligned rows fall through to the scalar loop).
  if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0) {
    const uint4* row_vec = reinterpret_cast<const uint4*>(row);
    const size_t groups = size / kArgmaxVecElems;  // full 8-element groups
    for (size_t g = threadIdx.x; g < groups; g += blockDim.x) {
      const uint4 u = row_vec[g];
      const size_t base = g * kArgmaxVecElems;
      const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        argmax_update(static_cast<uint16_t>(words[k] & 0xFFFFu), base + 2 * k, best, best_idx);
        argmax_update(static_cast<uint16_t>(words[k] >> 16), base + 2 * k + 1, best, best_idx);
      }
    }
    done = groups * kArgmaxVecElems;
  }

  // Scalar tail (or the whole row when the vector path did not apply).
  for (size_t i = done + threadIdx.x; i < size; i += blockDim.x) {
    argmax_update(row[i], i, best, best_idx);
  }

  block_reduce_argmax(best, best_idx, shared_max_value, shared_max_ptr);
  __syncthreads();
  if (threadIdx.x == 0) {
    // SIZE_MAX = the row held no comparable value (size == 0 or all-NaN
    // logits); token 0 is the only sane fallback.
    out_idx[blockIdx.x] = (best_idx == SIZE_MAX) ? 0 : static_cast<int32_t>(best_idx);
  }
}

void argmax_kernel_launch(const uint16_t* logits, size_t row_stride, size_t size, int32_t batch,
                          int32_t* out_idx, cudaStream_t stream) {
  if (batch <= 0) {
    return;
  }
  argmax_kernel_bf16<<<batch, kArgmaxThreads, 0, stream>>>(logits, row_stride, size, out_idx);
  CUDA_KERNEL_CHECK();
}
}  // namespace kernel
