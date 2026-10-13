// ============================================================================
// Device-side NLL reduction for the PPL gate (driven by ppl_benchmark.cpp).
//
// The LM head writes [rows, vocab] logits on the device at every decode step.
// For the PPL harness that is one step per position, so pulling logits back to
// the host would move rows*vocab*2 bytes per step (152K vocab, 32 rows, 4K
// positions: ~40 GB over a single pass) and force a stream sync every step.
// The reduction runs on the device instead; only the resulting [rows, 1] fp32
// NLL per step crosses back (and that is itself accumulated in a device buffer,
// so the pass crosses the bus once at the end).
//
// NLL(row) = logsumexp_v(logits[row][v]) - logits[row][target]
//          = row_max + log(sum_v exp(logits[row][v] - row_max)) - z_target
//
// The max is subtracted before the exp-sum (the standard trick): logits are
// computed in bf16 and a raw exp-sum would overflow for large |z|.
// ============================================================================
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

// Block-wide (256-thread) reduction of one value per thread into thread-0's
// value, smem-sized to the block. The shared buffer is re-seeded per call, so
// callers must __syncthreads() after reading the result before the next call
// reuses the buffer (the two passes below do).
__device__ __forceinline__ float block_reduce(float v, float* smem, bool is_max) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = is_max ? fmaxf(smem[tid], smem[tid + s]) : (smem[tid] + smem[tid + s]);
    }
    __syncthreads();
  }
  return smem[0];
}

}  // namespace

// logits: [rows, vocab] bf16, row-major.
// targets: [rows] int32 sampled with stride target_stride (the caller offsets
//   it to the current position, so the target of row r sits at
//   targets[r * target_stride]); out: [rows] fp32 with stride out_stride.
// One block per row. Two passes over the row (max, then exp-sum); the row is
// 2*vocab bytes, which for the served vocabularies stays resident in L2 across
// the two passes.
__global__ void ppl_nll_kernel_bf16(const __nv_bfloat16* __restrict__ logits,
                                    const int32_t* __restrict__ targets,
                                    float* __restrict__ out, int32_t vocab,
                                    int32_t target_stride, int32_t out_stride) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* row_logits = logits + static_cast<int64_t>(row) * vocab;
  __shared__ float smem[256];

  float m = -CUDART_INF_F;
  for (int v = tid; v < vocab; v += blockDim.x) {
    m = fmaxf(m, __bfloat162float(row_logits[v]));
  }
  const float row_max = block_reduce(m, smem, /*is_max=*/true);
  __syncthreads();  // every thread has the max before smem is reused below

  float s = 0.f;
  for (int v = tid; v < vocab; v += blockDim.x) {
    s += __expf(__bfloat162float(row_logits[v]) - row_max);
  }
  const float sum = block_reduce(s, smem, /*is_max=*/false);

  if (tid == 0) {
    const int32_t tgt = targets[static_cast<int64_t>(row) * target_stride];
    const float z_target = __bfloat162float(row_logits[tgt]);
    out[static_cast<int64_t>(row) * out_stride] = row_max + logf(sum) - z_target;
  }
}

// Host-side launcher. `stream` is the model's CUDA stream (void* so this
// header-free entry point can be declared from a .cpp); null means the default
// stream. No error check here: the harness reads the result back, and a launch
// failure surfaces as a cudaMemcpy/MemcpyAsync error at that point.
void ppl_nll_cu(const void* logits_bf16, const int32_t* targets, float* out, int32_t vocab,
                int32_t rows, int32_t target_stride, int32_t out_stride, void* stream) {
  if (rows <= 0 || vocab <= 0) {
    return;
  }
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  ppl_nll_kernel_bf16<<<rows, 256, 0, s>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits_bf16), targets, out, vocab,
      target_stride, out_stride);
}
