// 逐位置 KLD / Same Top-1 / 两个臂的 NLL 归约(与 wq_eval_kernel.cu 同一数值形式)。
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf2f(const __nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float block_max(float v, float* smem) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  const float out = smem[0];
  __syncthreads();
  return out;
}

__device__ __forceinline__ double block_sum(double v, double* smem) {
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  const double out = smem[0];
  __syncthreads();
  return out;
}

}  // namespace

__global__ void kv_eval_kernel_bf16(const __nv_bfloat16* __restrict__ ref_logits,
                                    const __nv_bfloat16* __restrict__ cand_logits,
                                    const int32_t* __restrict__ targets, int32_t vocab,
                                    float* __restrict__ out_kl, int32_t* __restrict__ same_top1,
                                    float* __restrict__ nll_ref,
                                    float* __restrict__ nll_cand) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* rrow = ref_logits + static_cast<int64_t>(row) * vocab;
  const __nv_bfloat16* crow = cand_logits + static_cast<int64_t>(row) * vocab;

  __shared__ float sf[256];
  __shared__ double sd[256];
  __shared__ int si[2];

  float mr_t = -CUDART_INF_F;
  float mc_t = -CUDART_INF_F;
  for (int v = tid; v < vocab; v += blockDim.x) {
    const float r = bf2f(rrow[v]);
    const float c = bf2f(crow[v]);
    mr_t = fmaxf(mr_t, r);
    mc_t = fmaxf(mc_t, c);
  }
  const float mr = block_max(mr_t, sf);
  const float mc = block_max(mc_t, sf);

  if (tid == 0) {
    si[0] = vocab;
    si[1] = vocab;
  }
  __syncthreads();

  double a_t = 0.0, b_t = 0.0, c_t = 0.0;
  for (int v = tid; v < vocab; v += blockDim.x) {
    const float r = bf2f(rrow[v]);
    const float c = bf2f(crow[v]);
    const float er = __expf(r - mr);
    a_t += static_cast<double>(er);
    b_t += static_cast<double>(er) * static_cast<double>(r - c);
    c_t += static_cast<double>(__expf(c - mc));
    if (r == mr) atomicMin(&si[0], v);
    if (c == mc) atomicMin(&si[1], v);
  }
  __syncthreads();

  const double A = block_sum(a_t, sd);
  const double B = block_sum(b_t, sd);
  const double C = block_sum(c_t, sd);
  const int arg_ref = si[0];
  const int arg_cand = si[1];

  if (tid == 0) {
    const double log_a = log(A);
    const double log_c = log(C);
    double kl = B / A - (static_cast<double>(mr) - static_cast<double>(mc)) + (log_c - log_a);
    if (kl < 0.0) kl = 0.0;
    out_kl[row] = static_cast<float>(kl);
    same_top1[row] = (arg_ref == arg_cand) ? 1 : 0;

    const int32_t tgt = targets[row];
    nll_ref[row] = static_cast<float>(log_a + static_cast<double>(mr) - bf2f(rrow[tgt]));
    nll_cand[row] = static_cast<float>(log_c + static_cast<double>(mc) - bf2f(crow[tgt]));
  }
}

void kv_eval_cu(const void* ref_logits_bf16, const void* cand_logits_bf16,
                const int32_t* targets, int32_t vocab, int32_t rows, float* out_kl,
                int32_t* same_top1, float* nll_ref, float* nll_cand, void* stream) {
  if (rows <= 0 || vocab <= 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  kv_eval_kernel_bf16<<<rows, 256, 0, s>>>(
      reinterpret_cast<const __nv_bfloat16*>(ref_logits_bf16),
      reinterpret_cast<const __nv_bfloat16*>(cand_logits_bf16), targets, vocab, out_kl,
      same_top1, nll_ref, nll_cand);
}
