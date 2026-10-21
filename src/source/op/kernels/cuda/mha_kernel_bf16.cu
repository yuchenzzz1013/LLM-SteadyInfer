// CUDA attention kernels (all BF16): continuous-layout Flash Decoding
// (mha_kernel_cu / mha_kernel_cu_batch), paged KV scatter and the paged
// attention family (paged_attention_dispatch / paged_attention_cu_batch /
// paged_prefill_attention_cu_batch).
//
// A step's batch is split by row type: the scheduler puts the decode rows
// first, and paged_attention_dispatch sends [0, num_decode_rows) to the
// split-KV decode kernel (paged_attn_split_kernel_bf16, one warp per row x
// kv head x split, K/V reused by every q head of the group) and
// [num_decode_rows, batch) — the prefill rows, grouped per sequence by
// seq_row_start — to paged_prefill_kernel_bf16, which stages each KV tile into
// smem once per CTA instead of re-reading the prefix once per (row, head).
// Both write disjoint row ranges of mha_out, so a mixed step runs one forward
// pass with the weights read once. A geometry the prefill kernel does not
// serve (head_size != 128, unusual page sizes) or a continuous layout keeps
// the whole batch on paged_attention_cu_batch / mha_kernel_cu_batch, which
// handle every supported head size.
//
// S4 KV quantization (K int4 grouped with a BF16 residual window + V int4
// asymmetric — see KvQuant) is a compile-time arm split, not a runtime flag
// inside one kernel: both paged kernels are templated on QUANT and the int4
// reads are guarded with `if constexpr`, so the QUANT false instantiations are
// the pre-quantization kernels — same registers, same instructions (the write
// path is split too: the residual scatter and the group quantizer are their
// own kernels). This matters because ptxas sizes a kernel's register file for
// its worst path: with quantization folded into the bf16 kernel as a runtime
// branch, the decode instantiation went from 71 to 94 registers per thread
// (~28 -> ~21 resident warps per SM on A100) and the whole offline benchmark
// lost ~7% throughput with quantization off — the switch was numerically
// transparent but not performance transparent.
//
// All q/k/v/output caches are raw bfloat16 (CUDA __nv_bfloat16, bit-compatible
// with the uint16_t host representation). Internal score / softmax / output
// arithmetic runs in float: the operands are exact bf16 values (products of
// two bf16 numbers are exact in float), so this matches cuBLAS BF16 semantics
// (bf16 operands, fp32 accumulation) rather than FP32 kernels — activation and
// cache data never round-trips through FP32 storage.
//
// Partials buffers hold one slot per (row, head, split). Two layouts share
// them: bf16 (o | m | l) rows of head_size + 2 elements (the continuous split
// path), and fp32 (acc | m | l) rows of head_size + 4 floats written by the
// paged split-KV decode kernel — 2x the bf16 element count, since a bf16
// partial would round the split's running accumulator before the combine.
// Callers size with flash_decoding_partials_elements (mha_kernel.cuh).
#include <base/cuda_config.h>
#include <base/cuda_check.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cstdlib>
#include <map>
#include <vector>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include "paged_kernels.cuh"
#include "model/qkv_split.h"
namespace kernel {

// ========== KV pool dtype gate ==========
// The KV wrappers below cannot tell a mis-typed pool from a correct one:
// Tensor::ptr<T>() is an unchecked reinterpret_cast, so a pool that is int8
// while the kernels read it as bf16 (or the reverse) computes garbage with no
// error anywhere. KV-cache quantization turns that into a live hazard — the
// pool dtype becomes runtime state, and every read site has to be updated by
// hand — so each wrapper states which dtypes its kernels can actually address
// and everything else fails here, at the entry, with the pool named.
//
// S4 is a package: the int4 K/V pools, their scale/zero pools and the BF16 K
// residual pool must arrive together. A missing member is unreadable (a
// dequantized row would be zero or garbage, and a residual position would read
// the wrong array), and a bf16 pool with metadata means the caller thinks
// quantization is on while the writer wrote full-precision rows. All of those
// are wiring bugs, so they are rejected on the argument set rather than at the
// first read.
//
// The K and V pools must have the same dtype: S4 quantizes both halves, and a
// mixed pair has no reader in this build.
static bool check_kv_pools(const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                           const tensor::Tensor& key_scale, const tensor::Tensor& key_zero,
                           const tensor::Tensor& value_scale, const tensor::Tensor& value_zero,
                           const tensor::Tensor& key_residual, const char* what) {
  const base::DataType dtype = key_cache.data_type();
  CHECK(dtype == value_cache.data_type())
      << what << ": K pool dtype " << static_cast<int>(dtype) << " != V pool dtype "
      << static_cast<int>(value_cache.data_type()) << "; S4 quantizes both halves together";
  CHECK(dtype == base::DataType::kDataTypeBF16 || dtype == base::DataType::kDataTypeInt8)
      << what << " has dtype " << static_cast<int>(dtype)
      << ", but the KV kernels address bf16 and int8-storage (int4-packed) pools only";
  if (dtype == base::DataType::kDataTypeInt8) {
    CHECK(!key_scale.is_empty() && key_scale.data_type() == base::DataType::kDataTypeFp32 &&
          !key_zero.is_empty() && key_zero.data_type() == base::DataType::kDataTypeFp32 &&
          !value_scale.is_empty() &&
          value_scale.data_type() == base::DataType::kDataTypeFp32 &&
          !value_zero.is_empty() && value_zero.data_type() == base::DataType::kDataTypeFp32)
        << what << ": int4 pools are missing a scale/zero pool (K scale empty="
        << key_scale.is_empty() << ", K zero empty=" << key_zero.is_empty()
        << ", V scale empty=" << value_scale.is_empty()
        << ", V zero empty=" << value_zero.is_empty() << ")";
    CHECK(!key_residual.is_empty() &&
          key_residual.data_type() == base::DataType::kDataTypeBF16)
        << what << ": int4 K pool is missing its BF16 residual pool (empty="
        << key_residual.is_empty() << ", dtype=" << static_cast<int>(key_residual.data_type())
        << ")";
    return true;
  }
  CHECK(key_scale.is_empty() && key_zero.is_empty() && value_scale.is_empty() &&
        value_zero.is_empty() && key_residual.is_empty())
      << what
      << ": bf16 pools were passed together with S4 scale/zero/residual tensors; the writer "
         "and the readers disagree on whether quantization is on";
  return false;
}

// ========== 4-bit element helpers ==========
// Both pools pack two values per byte: head dim d lives in byte d/2, the low
// nibble for even d. `off` is always a LOGICAL ELEMENT offset (one element is
// one head dim); the halving happens here, and every call site passes a
// 4-element-aligned offset so the 2-byte accesses stay aligned.
//
// S4 codes are UNSIGNED [0, 15]: the asymmetry lives in the zero point, so a
// nibble must not be sign-extended. (The removed symmetric arm stored -8..7;
// reading these codes that way folds every value >= 8 by -16.)
__device__ __forceinline__ float i4_nibble_to_float(uint32_t n) {
  return static_cast<float>(n & 0xfu);
}

// Dequantize one 4-dim chunk: x = (q - zero) * scale.
__device__ __forceinline__ void load_i4_chunk4(float* dst, const int8_t* cache, float scale,
                                               float zero, int64_t off) {
  const uint32_t raw = *reinterpret_cast<const uint16_t*>(cache + (off >> 1));
  dst[0] = (i4_nibble_to_float(raw & 0xfu) - zero) * scale;
  dst[1] = (i4_nibble_to_float((raw >> 4) & 0xfu) - zero) * scale;
  dst[2] = (i4_nibble_to_float((raw >> 8) & 0xfu) - zero) * scale;
  dst[3] = (i4_nibble_to_float((raw >> 12) & 0xfu) - zero) * scale;
}

// Same chunk from a prefill smem tile, where the packed row starts at the tile
// row base (so only the element offset inside the row is halved).
__device__ __forceinline__ void load_i4_tile_chunk4(float* dst, const int8_t* tile_row,
                                                    float scale, float zero, int32_t elem_off) {
  const uint32_t raw = *reinterpret_cast<const uint16_t*>(tile_row + (elem_off >> 1));
  dst[0] = (i4_nibble_to_float(raw & 0xfu) - zero) * scale;
  dst[1] = (i4_nibble_to_float((raw >> 4) & 0xfu) - zero) * scale;
  dst[2] = (i4_nibble_to_float((raw >> 8) & 0xfu) - zero) * scale;
  dst[3] = (i4_nibble_to_float((raw >> 12) & 0xfu) - zero) * scale;
}

__device__ __forceinline__ void load_bf16_chunk4(float* dst, const __nv_bfloat16* cache,
                                                 int64_t off) {
  const uint2 raw = *reinterpret_cast<const uint2*>(cache + off);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  dst[0] = __bfloat162float(h[0].x);
  dst[1] = __bfloat162float(h[0].y);
  dst[2] = __bfloat162float(h[1].x);
  dst[3] = __bfloat162float(h[1].y);
}

// Raw (undequantized) 4-code chunk: the value-side reader uses this and folds
// the affine transform into its weight accumulator, so the per-value subtract
// and multiply are not paid here.
__device__ __forceinline__ void load_i4_raw_chunk4(float* dst, const int8_t* cache, int64_t off) {
  const uint32_t raw = *reinterpret_cast<const uint16_t*>(cache + (off >> 1));
  dst[0] = static_cast<float>(raw & 0xfu);
  dst[1] = static_cast<float>((raw >> 4) & 0xfu);
  dst[2] = static_cast<float>((raw >> 8) & 0xfu);
  dst[3] = static_cast<float>((raw >> 12) & 0xfu);
}

// Stage one 4-element chunk of an int4 row into the prefill tile: the tile
// keeps the packed bytes, the dequantization happens on the way out (see
// paged_prefill_kernel_bf16). `tile_dst` is the chunk's byte address in the
// tile; `off` is the SOURCE element offset into the packed pool.
// Raw codes from a packed smem tile (the value-side reader folds the affine
// into its weight, exactly like load_i4_raw_chunk4 on the global path).
__device__ __forceinline__ void load_i4_tile_raw_chunk4(float* dst, const int8_t* tile_row,
                                                        int32_t elem_off) {
  const uint32_t raw = *reinterpret_cast<const uint16_t*>(tile_row + (elem_off >> 1));
  dst[0] = static_cast<float>(raw & 0xfu);
  dst[1] = static_cast<float>((raw >> 4) & 0xfu);
  dst[2] = static_cast<float>((raw >> 8) & 0xfu);
  dst[3] = static_cast<float>((raw >> 12) & 0xfu);
}

__device__ __forceinline__ void stage_i4_row_chunk(int8_t* tile_dst, const int8_t* cache,
                                                   int64_t off) {
  *reinterpret_cast<uint16_t*>(tile_dst) =
      *reinterpret_cast<const uint16_t*>(cache + (off >> 1));
}

// ========== Dynamic shared-memory opt-in ==========
// The flash-decoding kernels size their K/V tile scratch from max_seq_len /
// num_splits, so beyond ~97K context (head_size 128, 8 splits) the requested
// dynamic smem exceeds the 48KB default. A launch above a kernel's granted
// dynamic-smem limit fails with cudaErrorInvalidValue; that failure used to be
// silent, leaving the split partials stale and attention returning garbage.
// Raise each kernel's limit to the device maximum once, then fail loudly if
// the requested size is beyond what the device can ever grant.
static int max_dynamic_smem_for(const void* kernel, const char* name) {
  static std::map<const void*, int> granted_limits;
  auto it = granted_limits.find(kernel);
  if (it != granted_limits.end()) {
    return it->second;
  }
  int device = 0;
  int max_optin = 0;
  int granted = 0;
  cudaFuncAttributes attrs{};
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
  if (err == cudaSuccess) {
    err = cudaFuncGetAttributes(&attrs, kernel);
  }
  if (err == cudaSuccess) {
    // Static __shared__ bytes count against the opt-in limit, so the dynamic
    // ceiling is the device limit minus them (these kernels carry 48B of cub
    // temp storage). Asking for the full device limit fails with
    // cudaErrorInvalidValue, which is why the ceiling is derived, not assumed.
    const int ceiling = max_optin - static_cast<int>(attrs.sharedSizeBytes);
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ceiling);
    if (err == cudaSuccess) {
      granted = ceiling;
    }
  }
  if (err != cudaSuccess) {
    // Do not leave the failed driver call as the thread's last error: the
    // debug-build launch check would otherwise blame the next kernel launch.
    cudaGetLastError();
    LOG(ERROR) << "[MHA] failed to opt " << name << " into dynamic shared memory: "
               << cudaGetErrorString(err) << " (device=" << device
               << ", max_optin=" << max_optin << "); launches above 48KB will fail.";
  }
  granted_limits.emplace(kernel, granted);
  return granted;
}

static void check_dynamic_smem_fits(const void* kernel, const char* name, int smem_bytes,
                                    int32_t max_seq_len) {
  if (smem_bytes <= 48 * 1024) {
    return;
  }
  const int limit = max_dynamic_smem_for(kernel, name);
  CHECK_GE(limit, smem_bytes)
      << name << " needs " << (smem_bytes >> 10) << "KB of dynamic shared memory (max_seq_len="
      << max_seq_len << "), above the device's opt-in limit of " << (limit >> 10) << "KB.";
}

// Flash-softmax helper: scores live in fp32 smem (computed from bf16 inputs);
// the flash statistic is never stored in bf16.
__device__ void flash_softmax_tile_bf16(float* p_smem, int tile_len, float& m, float& l) {
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;

  float local_max = -FLT_MAX;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    local_max = fmaxf(local_max, p_smem[t]);
  }
  m = BlockReduce(temp).Reduce(local_max, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_val = m;
  }
  __syncthreads();
  m = shared_val;

  float local_l = 0.f;
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    const float p = expf(p_smem[t] - m);
    p_smem[t] = p;
    local_l += p;
  }
  l = BlockReduce(temp).Sum(local_l);
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_val = l;
  }
  __syncthreads();
  l = shared_val;
}

// ========== Legacy single-sequence attention kernel ==========
// bf16 q / kv-cache rows / output; heads are read as bf16 and converted once
// into the fp32 smem query, KV elements convert on the fly per use.
constexpr int TILE_BF16 = 128;

__global__ void multi_head_attention_kernel_bf16(
    int32_t pos, int32_t seq_len, const __nv_bfloat16* query, const __nv_bfloat16* output,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache, int32_t kv_dim,
    int32_t kv_head_num, int32_t head_num, int32_t head_size, int64_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_mem[];
  float* s_query = s_mem;                 // [head_size]
  float* s_p = s_mem + head_size;         // [TILE_BF16]

  const float scale = 1.f / sqrtf(float(head_size));
  const __nv_bfloat16* query_head = query + head * head_size;
  const int seq = pos + 1;
  // GQA KV head mapping: h * kv_head_num / head_num handles uneven splits.
  const int head_offset = (head * kv_head_num / head_num) * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = __bfloat162float(query_head[i]);
  }
  __syncthreads();

  float m_i = -FLT_MAX;
  float l_i = 0.f;
  // Per-thread accumulator for out[head_size] (thread d owns dimension d).
  float acc = 0.f;

  for (int tile_start = 0; tile_start < seq; tile_start += TILE_BF16) {
    const int tile_len = min(TILE_BF16, seq - tile_start);

    // Scores: s_t = sum_d q[d] * K[t][head_offset + d], strided over threads.
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
      const __nv_bfloat16* k_row =
          key_cache + layer_offset + static_cast<int64_t>(tile_start + t) * kv_dim + head_offset;
      float s_t = 0.f;
      for (int d = 0; d < head_size; ++d) {
        s_t += __bfloat162float(k_row[d]) * s_query[d];
      }
      s_p[t] = s_t * scale;
    }

    float m_j, l_j;
    flash_softmax_tile_bf16(s_p, tile_len, m_j, l_j);
    // First tile: m_i == -inf, so alpha must be 1 (not exp(+inf) = inf).
    const float alpha = (m_i == -FLT_MAX) ? 1.0f : expf(m_j - m_i);
    l_i = alpha * l_i + l_j;
    m_i = m_j;

    // acc = alpha * acc + sum_t p_t * V[t][head_offset + d]
    acc *= alpha;
    const int d = threadIdx.x;
    if (d < head_size) {
      const __nv_bfloat16* v_col =
          value_cache + layer_offset + static_cast<int64_t>(tile_start) * kv_dim + head_offset + d;
      for (int tt = 0; tt < tile_len; ++tt) {
        acc += s_p[tt] * __bfloat162float(v_col[static_cast<int64_t>(tt) * kv_dim]);
      }
    }
  }

  if (threadIdx.x < head_size) {
    const_cast<__nv_bfloat16*>(output)[head * head_size + threadIdx.x] =
        __float2bfloat16(acc / l_i);
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                        int32_t kv_dim, int32_t kv_head_num, int32_t head_size,
                        const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                        const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor,
                        const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                        CudaConfig* config) {
  UNUSED(device_type);
  UNUSED(score_tensor);
  CHECK_LE(head_size, 256) << "Flash kernel requires head_size <= 256 (one output dim per thread).";
  CHECK(config != nullptr);
  // int64: large external caches (num_slots * max_seq_len) can overflow int32.
  int64_t layer_offset = static_cast<int64_t>(layer_index) * seq_len * kv_dim;
  const __nv_bfloat16* query = query_tensor.ptr<__nv_bfloat16>();
  const __nv_bfloat16* output = mha_out.ptr<__nv_bfloat16>();
  const __nv_bfloat16* key_cache = key_cache_tensor.ptr<__nv_bfloat16>();
  const __nv_bfloat16* value_cache = value_cache_tensor.ptr<__nv_bfloat16>();

  cudaStream_t stream = config->stream;
  int smem_bytes = (head_size + TILE_BF16) * sizeof(float);
  multi_head_attention_kernel_bf16<<<head_num, 256, smem_bytes, stream>>>(
      pos, seq_len, query, output, key_cache, value_cache, kv_dim, kv_head_num, head_num,
      head_size, layer_offset);
  CUDA_KERNEL_CHECK();
}

// ========== Flash Decoding (continuous KV layout) ==========
// bf16 query / caches / partials / output. Partials slots keep (head_size + 2)
// bf16 elements: [o | m | l], with o rounded to bf16 per split and m/l kept in
// bf16 as well — that halves the scratch for the continuous layout. The
// tradeoff is a per-token rounding residual against the fp32-register paged
// paths: measured, it does not change the first ~10 generated tokens but can
// flip a later one.
__global__ void flash_decoding_kernel_bf16(const int32_t* positions, const int32_t* kv_offsets,
                                           int32_t num_slots, int32_t max_seq_len,
                                           int32_t layer_idx, int32_t dim,
                                           const __nv_bfloat16* query, __nv_bfloat16* partials,
                                           const __nv_bfloat16* key_cache,
                                           const __nv_bfloat16* value_cache, int32_t kv_dim,
                                           int32_t kv_head_num, int32_t head_num, int32_t head_size,
                                           int32_t num_splits) {
  const int pair = blockIdx.x / num_splits;  // pair = b * head_num + head
  const int split = blockIdx.x % num_splits;
  const int b = pair / head_num;
  const int head = pair % head_num;
  const int pos = positions[b];
  const int seq = pos + 1;

  const int tile_start = split * seq / num_splits;
  const int tile_end = (split + 1) * seq / num_splits;
  // The combine kernel reads every split's partial slot unconditionally, so
  // empty splits must record a neutral (o=0, m=-inf, l=0) partial.
  __nv_bfloat16* partial =
      partials + (static_cast<int64_t>(pair) * num_splits + split) * (head_size + 2);
  if (tile_start >= tile_end) {
    for (int dd = threadIdx.x; dd < head_size; dd += blockDim.x) {
      partial[dd] = __float2bfloat16(0.f);
    }
    if (threadIdx.x == 0) {
      partial[head_size] = __float2bfloat16(-FLT_MAX);  // -inf in bf16
      partial[head_size + 1] = __float2bfloat16(0.f);
    }
    return;  // empty split for short sequences
  }
  const int tile_len = tile_end - tile_start;

  extern __shared__ float s_mem[];
  float* s_query = s_mem;          // [head_size]
  float* s_p = s_mem + head_size;  // [max_tile] raw scores -> probs

  const float scale = 1.f / sqrtf(float(head_size));
  const int head_offset = (head * kv_head_num / head_num) * head_size;
  const int slot = kv_offsets[b];
  const int64_t cache_base =
      static_cast<int64_t>(layer_idx) * num_slots * kv_dim * max_seq_len +
      static_cast<int64_t>(slot) * kv_dim * max_seq_len + tile_start;
  const __nv_bfloat16* q_head = query + static_cast<int64_t>(b) * dim + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query[i] = __bfloat162float(q_head[i]);
  }
  __syncthreads();

  // Scores over this split's KV tile: s_t = sum_d q[d] * K[hd + d][t].
  for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
    float s_t = 0.f;
    for (int d = 0; d < head_size; ++d) {
      s_t += s_query[d] *
             __bfloat162float(key_cache[cache_base +
                                       static_cast<int64_t>(head_offset + d) * max_seq_len + t]);
    }
    s_p[t] = s_t * scale;
  }

  float m_i, l_i;
  flash_softmax_tile_bf16(s_p, tile_len, m_i, l_i);

  // Partial output: acc_d = sum_t p_t * V[head_offset + d][t].
  float acc = 0.f;
  const int d = threadIdx.x;
  if (d < head_size) {
    const __nv_bfloat16* v_dim =
        value_cache + cache_base + static_cast<int64_t>(head_offset + d) * max_seq_len;
    for (int tt = 0; tt < tile_len; ++tt) {
      acc += s_p[tt] * __bfloat162float(v_dim[tt]);
    }
  }

  // partials[pair * num_splits + split] -> [o (head_size) | m | l]
  if (d < head_size) {
    partial[d] = __float2bfloat16(acc);
  }
  if (threadIdx.x == 0) {
    partial[head_size] = __float2bfloat16(m_i);
    partial[head_size + 1] = __float2bfloat16(l_i);
  }
}

__global__ void flash_decoding_combine_kernel_bf16(const __nv_bfloat16* partials,
                                                   __nv_bfloat16* output, int32_t dim,
                                                   int32_t head_num, int32_t head_size,
                                                   int32_t num_splits, int32_t batch) {
  const int pair = blockIdx.x;
  const int b = pair / head_num;
  const int head = pair % head_num;
  if (b >= batch) {
    return;
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float s_global_m;

  const __nv_bfloat16* partial =
      partials + static_cast<int64_t>(pair) * num_splits * (head_size + 2);

  // Global max over splits.
  float local_m = (threadIdx.x < num_splits)
                      ? __bfloat162float(partial[threadIdx.x * (head_size + 2) + head_size])
                      : -FLT_MAX;
  float global_m = BlockReduce(temp).Reduce(local_m, cub::Max());
  __syncthreads();
  if (threadIdx.x == 0) {
    s_global_m = global_m;
  }
  __syncthreads();
  global_m = s_global_m;

  // o = sum_s o_s * exp(m_s - m) / sum_s l_s * exp(m_s - m)
  float o = 0.f;
  float l = 0.f;
  const int d = threadIdx.x;
  for (int s = 0; s < num_splits; ++s) {
    const __nv_bfloat16* ps = partial + static_cast<int64_t>(s) * (head_size + 2);
    const float w = expf(__bfloat162float(ps[head_size]) - global_m);
    l += w * __bfloat162float(ps[head_size + 1]);
    if (d < head_size) {
      o += w * __bfloat162float(ps[d]);
    }
  }
  if (d < head_size) {
    output[static_cast<int64_t>(b) * dim + head * head_size + d] = __float2bfloat16(o / l);
  }
}

void mha_kernel_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_slots,
                              int32_t max_seq_len, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& kv_offsets, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              CudaConfig* config) {
  int32_t batch = query_batch.get_dim(0);
  int32_t dim = query_batch.get_dim(1);
  int32_t num_splits = flash_decoding_num_splits(max_seq_len);
  CHECK_LE(head_size, 256)
      << "Flash Decoding requires head_size <= 256 (one output dim per thread).";
  CHECK(config != nullptr);

  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  __nv_bfloat16* partials = const_cast<__nv_bfloat16*>(score_batch.ptr<__nv_bfloat16>());
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  const __nv_bfloat16* kcache = key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = value_cache.ptr<__nv_bfloat16>();

  cudaStream_t stream = config->stream;
  // s_p must hold the largest tile any split can own.
  int max_tile = (max_seq_len + num_splits - 1) / num_splits;
  int smem_bytes = (head_size + max_tile) * sizeof(float);
  check_dynamic_smem_fits((const void*)flash_decoding_kernel_bf16, "flash_decoding_kernel_bf16",
                          smem_bytes, max_seq_len);
  int total_blocks = batch * head_num * num_splits;
  flash_decoding_kernel_bf16<<<total_blocks, 256, smem_bytes, stream>>>(
      positions.ptr<int32_t>(), kv_offsets.ptr<int32_t>(), num_slots, max_seq_len, layer_idx, dim,
      query, partials, kcache, vcache, kv_dim, kv_head_num, head_num, head_size, num_splits);
  CUDA_KERNEL_CHECK();
  flash_decoding_combine_kernel_bf16<<<batch * head_num, 256, 0, stream>>>(
      partials, output, dim, head_num, head_size, num_splits, batch);
  CUDA_KERNEL_CHECK();
}

// ========== Paged KV scatter ==========
// Raw bf16 copy through the block table. The row moves 8 bytes (4 bf16) at a
// time: every address term below is a multiple of kv_dim, so with kv_dim a
// multiple of 4 both sides sit on 8-byte boundaries of CUDA-aligned pool
// tensors and each row is a plain run of uint2. kv_dim = kv_head_num *
// head_size is a multiple of 4 in every served geometry (head_size >= 8), but
// the element-wise loop stays as the fallback so an odd kv_dim degrades to the
// old behaviour instead of misaligning a vector access.
__global__ void paged_kv_scatter_kernel_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                             const int32_t* block_table, int32_t table_stride,
                                             const int32_t* positions, int32_t kv_dim,
                                             int32_t num_blocks, int32_t block_size,
                                             int32_t layer_idx) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int pos = positions[b];
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int block_id = __ldg(table_row + (pos / block_size));
  // Offset within the page. NOT `pos - block_id * block_size`: block_id is the
  // *physical* page from the table, so that expression only equals pos %
  // block_size when the table happens to be the identity, and otherwise
  // cancels the page term out of the address entirely.
  const int off = pos % block_size;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const int64_t base = layer_base + static_cast<int64_t>(block_id) * (block_size * kv_dim) +
                       static_cast<int64_t>(off) * kv_dim;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(b) * kv_dim;
  if ((kv_dim & 3) == 0) {
    const uint2* src_vec = reinterpret_cast<const uint2*>(src_row);
    uint2* dst_vec = reinterpret_cast<uint2*>(dst + base);
    const int n_vec = kv_dim >> 2;
    for (int i = tid; i < n_vec; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int d = tid; d < kv_dim; d += blockDim.x) {
      dst[base + d] = src_row[d];
    }
  }
}

// Derive (scale, zero) for one asymmetric 4-bit group. The reader computes
// x = (q - zero) * scale with q in [0, 15], so the same rounding here is what
// the dequantization undoes:
//   scale = (max - min) / 15, zero = -min / scale.
// A constant group (range 0) still needs a positive step. Floors tied to the
// value magnitude keep `zero` small enough to be subtracted without
// cancellation: a floor of 1e-8 alone would make x / scale + zero lose all
// precision for an ordinary large constant.
__device__ __forceinline__ void s4_quant_params(float mn, float mx, float& scale, float& zero) {
  const float range = mx - mn;
  const float mag = fmaxf(fabsf(mn), fabsf(mx));
  const float step = fmaxf(range, fmaxf(mag * 1e-3f, 1e-8f));
  scale = step * (1.0f / 15.0f);
  zero = -mn / scale;
}

__device__ __forceinline__ int s4_quantize(float x, float inv_scale, float zero) {
  return min(15, max(0, __float2int_rn(x * inv_scale + zero)));
}

// V write path: asymmetric int4, one warp per (batch row, kv head), lane l
// owning head dims [l*4, l*4+4). The (min, max) pair is reduced across the
// lanes that own the head's dims; every lane then quantizes its own 4 values,
// and lane 0 writes the scale and zero. Data and metadata come out of one
// pass, so blocks stay independent of each other.
//
// head_size <= 128 is one pass (the served geometries: 64 and 128). A
// head_size in (128, 256] needs both passes' min/max before anything can be
// quantized, so the values stay in registers and one reduction covers the
// whole head — a per-chunk reduction would emit two scale pairs for one head
// and the reader, which dequantizes a head with one pair, would take half the
// row at the wrong magnitude. Lanes past a shorter head's end feed the
// reduction its neutral elements and write nothing.
__global__ void paged_kv_scatter_v4_kernel_bf16(const __nv_bfloat16* __restrict__ src,
                                                int8_t* __restrict__ dst,
                                                float* __restrict__ dst_scale,
                                                float* __restrict__ dst_zero,
                                                const int32_t* __restrict__ block_table,
                                                int32_t table_stride,
                                                const int32_t* __restrict__ positions,
                                                int32_t kv_dim, int32_t kv_head_num,
                                                int32_t num_blocks, int32_t block_size,
                                                int32_t layer_idx) {
  const int b = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int n_warps = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pos = positions[b];
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int block_id = __ldg(table_row + (pos / block_size));
  // Physical page: see the bf16 kernel for why this is pos % block_size and
  // not pos - block_id * block_size.
  const int off = pos % block_size;
  const int head_size = kv_dim / kv_head_num;
  const int n_chunks = (head_size + 127) >> 7;  // 1 for <= 128, 2 up to the 256 cap
  const int64_t layer_data_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim +
      static_cast<int64_t>(block_id) * (block_size * kv_dim) + static_cast<int64_t>(off) * kv_dim;
  const int64_t layer_scale_base =
      static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_head_num;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(b) * kv_dim;
  // 4-bit: dim d lives in byte d/2, so the row's element offsets halve. Every
  // term above is a multiple of kv_dim (even) and lane chunks are 4-aligned,
  // so the halvings are exact and the uint16 stores stay 2-byte aligned.
  const int64_t data_base = layer_data_base >> 1;
  const int64_t head_stride = head_size >> 1;

  for (int head = warp; head < kv_head_num; head += n_warps) {
    const __nv_bfloat16* head_src = src_row + static_cast<int64_t>(head) * head_size;
    float val[2][4];
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      if (c >= n_chunks) break;
      const int d = c * 128 + lane * 4;
      if (d >= head_size) break;
      const uint2 raw = *reinterpret_cast<const uint2*>(head_src + d);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      val[c][0] = __bfloat162float(h[0].x);
      val[c][1] = __bfloat162float(h[0].y);
      val[c][2] = __bfloat162float(h[1].x);
      val[c][3] = __bfloat162float(h[1].y);
      vmin = fminf(vmin, fminf(fminf(val[c][0], val[c][1]), fminf(val[c][2], val[c][3])));
      vmax = fmaxf(vmax, fmaxf(fmaxf(val[c][0], val[c][1]), fmaxf(val[c][2], val[c][3])));
    }
    // Lanes past a < 128-dim head must not contribute +/-FLT_MAX-like
    // sentinels to the other lanes' reduction.
    if (lane * 4 >= head_size) {
      vmin = 0.f;
      vmax = 0.f;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      vmin = fminf(vmin, __shfl_xor_sync(0xffffffffu, vmin, o));
      vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    }
    float scale = 0.f;
    float zero = 0.f;
    s4_quant_params(vmin, vmax, scale, zero);
    const float inv_scale = 1.0f / scale;
    int8_t* head_dst = dst + data_base + static_cast<int64_t>(head) * head_stride;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      if (c >= n_chunks) break;
      const int d = c * 128 + lane * 4;
      if (d >= head_size) break;
      uint32_t packed = 0;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        packed |= static_cast<uint32_t>(s4_quantize(val[c][v], inv_scale, zero)) << (4 * v);
      }
      *reinterpret_cast<uint16_t*>(head_dst + (d >> 1)) = static_cast<uint16_t>(packed);
    }
    if (lane == 0) {
      const int64_t idx = layer_scale_base +
                          (static_cast<int64_t>(block_id) * kv_head_num + head) * block_size + off;
      dst_scale[idx] = scale;
      dst_zero[idx] = zero;
    }
  }
}

// K write path, step 1: copy every new token's K row into the BF16 residual
// window at [layer][slot][pos % group_size][d]. The row is keyed by the
// sequence slot (not the physical block) because it is private state that the
// prefix cache must not share: only a COMPLETE group's int4 copy is
// block-addressable.
__global__ void paged_kv_residual_scatter_kernel_bf16(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ residual,
    const int32_t* __restrict__ kv_slots, const int32_t* __restrict__ positions,
    const int32_t* __restrict__ seq_row_start, int32_t num_decode_rows,
    int32_t num_prefill_seqs, int32_t kv_dim, int32_t max_batch, int32_t group_size,
    int32_t layer_idx) {
  const int b = blockIdx.x;
  const int pos = positions[b];
  // Only the sequence's LAST incomplete group needs the residual ring. The
  // quantizer (which runs before this scatter) reads every complete group's
  // tokens from `src`, so writing them here would only create a write-write
  // race: in a long prefill chunk, positions p and p + group_size would map to
  // the same ring slot and attention's tail could read whichever won. The
  // final boundary comes from the sequence's last row in this step, not from
  // this row's own position — rows earlier in the same chunk are already
  // covered by a group that completes later in the step.
  int last_pos = pos;
  if (seq_row_start != nullptr && b >= num_decode_rows) {
    for (int s = 0; s < num_prefill_seqs; ++s) {
      const int rs = seq_row_start[s];
      const int re = seq_row_start[s + 1];
      if (b >= rs && b < re) {
        last_pos = positions[re - 1];
        break;
      }
    }
  }
  const int boundary = ((last_pos + 1) / group_size) * group_size;
  if (pos < boundary) {
    return;
  }
  const int slot = kv_slots[b];
  const int64_t base =
      ((static_cast<int64_t>(layer_idx) * max_batch + slot) * group_size +
       (pos % group_size)) *
      kv_dim;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(b) * kv_dim;
  if ((kv_dim & 3) == 0) {
    const uint2* src_vec = reinterpret_cast<const uint2*>(src_row);
    uint2* dst_vec = reinterpret_cast<uint2*>(residual + base);
    const int n_vec = kv_dim >> 2;
    for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
      residual[base + d] = src_row[d];
    }
  }
}

// K write path, step 2: when a row completes a group ((pos+1) % group_size ==
// 0), quantize that group's K from the residual window and write the int4
// bytes plus one (scale, zero) pair per channel group. One block per (batch
// row, channel group); rows that do not complete a group return immediately.
//
// The kernel is launched on every step with a grid derived only from the batch
// and the channel count — never from positions — so a captured decode graph
// stays valid while its rows cross group boundaries. The decision itself is
// taken from `positions` inside the kernel, which is the only per-step input.
//
// group_size divides block_size, so every token of a group lives in one
// physical page and consecutive offsets; the block id and page offset are
// therefore read once from the block table and reused for the whole group.
__global__ void paged_kv_group_quantize_kernel_bf16(
    const __nv_bfloat16* __restrict__ src, const __nv_bfloat16* __restrict__ residual,
    int8_t* __restrict__ dst, float* __restrict__ dst_scale, float* __restrict__ dst_zero,
    const int32_t* __restrict__ block_table, int32_t table_stride,
    const int32_t* __restrict__ kv_slots, const int32_t* __restrict__ positions,
    const int32_t* __restrict__ seq_row_start, int32_t num_decode_rows,
    int32_t num_prefill_seqs, int32_t kv_dim, int32_t max_batch, int32_t group_size,
    int32_t channel_group, int32_t channel_groups, int32_t num_blocks, int32_t block_size,
    int32_t layer_idx) {
  // One block per batch row, not per (row, channel group): on a decode step
  // almost every row is a no-op, and a channel_groups-wide grid made those
  // no-ops cost more than the attention math they accompany (measured: the
  // 32x4096 PPL run went from 80s to 185s). A row that does complete a group
  // loops its channel groups here; rows that do not exit after two loads.
  const int b = blockIdx.x;
  const int pos = positions[b];
  if ((pos + 1) % group_size != 0) return;
  const int slot = kv_slots[b];
  const int base_pos = pos + 1 - group_size;
  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int block_id = __ldg(table_row + (base_pos / block_size));
  const int off = base_pos % block_size;
  const int64_t residual_base =
      ((static_cast<int64_t>(layer_idx) * max_batch + slot) * group_size) * kv_dim;

  // Group tokens that belong to THIS step are read from `src` (the freshly
  // projected K rows), not from the residual ring: the ring holds only
  // group_size slots per sequence, so a later group inside the same prefill
  // chunk would otherwise overwrite an earlier group before it is quantized.
  // The quantizer runs BEFORE the residual scatter for exactly this reason.
  // Older tokens (a group spanning a chunk boundary, and every decode token
  // but the newest) are still in the ring, untouched.
  //
  // Prefill rows of one sequence are contiguous and their positions increase
  // by one, so p maps to row_start + (p - positions[row_start]); decode rows
  // own only their own position. A decode row may also share its sequence with
  // a prefill run in a teacher-forcing batch shape (the PPL chunked driver);
  // matching on the KV slot covers that without affecting serving, where a
  // decode sequence is never also prefilling.
  int prefill_row_start = -1;
  int prefill_row_end = -1;
  if (seq_row_start != nullptr) {
    for (int s = 0; s < num_prefill_seqs; ++s) {
      const int rs = seq_row_start[s];
      const int re = seq_row_start[s + 1];
      const bool same_seq = b < num_decode_rows && rs < re && kv_slots[rs] == slot;
      if ((b >= rs && b < re) || same_seq) {
        prefill_row_start = rs;
        prefill_row_end = re;
        break;
      }
    }
  }
  const int prefill_pos0 = prefill_row_start >= 0 ? positions[prefill_row_start] : 0;
  auto value_at = [&](int p, int d) -> float {
    int row = -1;
    if (p == pos) {
      row = b;
    } else if (prefill_row_start >= 0) {
      const int idx = prefill_row_start + (p - prefill_pos0);
      if (idx >= prefill_row_start && idx < prefill_row_end && positions[idx] == p) {
        row = idx;
      }
    }
    if (row >= 0) {
      return __bfloat162float(src[static_cast<int64_t>(row) * kv_dim + d]);
    }
    return __bfloat162float(
        residual[residual_base + static_cast<int64_t>(p % group_size) * kv_dim + d]);
  };

  // One warp per channel group, no block-wide reductions: a 128-thread block
  // walking 128 channel groups paid ~900 __syncthreads per completing row
  // (measured 1.7ms). A warp reduction is five shuffles and the packing loop
  // needs no barrier at all.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const int64_t page_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim +
                            static_cast<int64_t>(block_id) * (block_size * kv_dim);
  for (int cg = warp; cg < channel_groups; cg += nwarps) {
    const int d0 = cg * channel_group;
    const int d_end = min(d0 + channel_group, kv_dim);
    const int cg_width = d_end - d0;

    float mn = FLT_MAX;
    float mx = -FLT_MAX;
    for (int j = 0; j < group_size; ++j) {
      for (int c = lane; c < cg_width; c += 32) {
        const float v = value_at(base_pos + j, d0 + c);
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float scale = 0.f;
    float zero = 0.f;
    s4_quant_params(mn, mx, scale, zero);
    const float inv_scale = 1.0f / scale;

    // One (scale, zero) pair per (page, group-in-page, channel group).
    const int64_t scale_idx =
        ((static_cast<int64_t>(layer_idx) * num_blocks + block_id) *
             (block_size / group_size) +
         off / group_size) *
            channel_groups +
        cg;
    if (lane == 0) {
      dst_scale[scale_idx] = scale;
      dst_zero[scale_idx] = zero;
    }

    // Pack adjacent dims into one byte: dim d lives in byte d/2, low nibble
    // for even d. cg_width is a multiple of 4, hence even.
    for (int by = lane; by < (cg_width >> 1); by += 32) {
      const int local_d = d0 + 2 * by;
      for (int j = 0; j < group_size; ++j) {
        const int p = base_pos + j;
        const int q0 = s4_quantize(value_at(p, local_d), inv_scale, zero);
        const int q1 = s4_quantize(value_at(p, local_d + 1), inv_scale, zero);
        const int64_t elem = page_base + static_cast<int64_t>(off + j) * kv_dim + local_d;
        dst[elem >> 1] = static_cast<int8_t>(q0 | (q1 << 4));
      }
    }
  }
}

// V scatter wrapper: bf16 pools get the raw copy, int4 pools the asymmetric
// quantization. K never takes this path in the quantized mode — it goes
// through the residual scatter + group quantize pair above — but the bf16
// baseline still scatters both halves with the raw copy.
void paged_kv_scatter_cu(const tensor::Tensor& src, tensor::Tensor& dst_cache,
                         const tensor::Tensor& dst_scale, const tensor::Tensor& dst_zero,
                         const tensor::Tensor& block_table, const tensor::Tensor& positions,
                         int32_t kv_dim, int32_t kv_head_num, int32_t num_blocks,
                         int32_t block_size, int32_t layer_idx, void* stream) {
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  const int32_t batch = src.get_dim(0);
  const int32_t table_stride = block_table.get_dim(1);
  if (dst_cache.data_type() == base::DataType::kDataTypeInt8) {
    CHECK(!dst_scale.is_empty() && dst_scale.data_type() == base::DataType::kDataTypeFp32 &&
          !dst_zero.is_empty() && dst_zero.data_type() == base::DataType::kDataTypeFp32)
        << "paged_kv_scatter_cu: int4 V pool needs non-empty fp32 scale and zero pools";
    CHECK(kv_head_num > 0 && kv_dim % kv_head_num == 0)
        << "kv_dim=" << kv_dim << " kv_head_num=" << kv_head_num;
    const int32_t head_size = kv_dim / kv_head_num;
    CHECK((head_size & 3) == 0 && head_size <= 256)
        << "the quantizing scatter needs a head_size that is a multiple of 4 and at most 256, got "
        << head_size;
    int8_t* dst = const_cast<int8_t*>(dst_cache.ptr<int8_t>());
    float* dst_scale_ptr = const_cast<float*>(dst_scale.ptr<float>());
    float* dst_zero_ptr = const_cast<float*>(dst_zero.ptr<float>());
    constexpr int kThreads = 256;  // 8 warps: one kv head per warp per pass, then stride
    if (stream_) {
      paged_kv_scatter_v4_kernel_bf16<<<batch, kThreads, 0, stream_>>>(
          src.ptr<__nv_bfloat16>(), dst, dst_scale_ptr, dst_zero_ptr, block_table.ptr<int32_t>(),
          table_stride, positions.ptr<int32_t>(), kv_dim, kv_head_num, num_blocks, block_size,
          layer_idx);
      CUDA_KERNEL_CHECK();
    } else {
      paged_kv_scatter_v4_kernel_bf16<<<batch, kThreads>>>(
          src.ptr<__nv_bfloat16>(), dst, dst_scale_ptr, dst_zero_ptr, block_table.ptr<int32_t>(),
          table_stride, positions.ptr<int32_t>(), kv_dim, kv_head_num, num_blocks, block_size,
          layer_idx);
      CUDA_KERNEL_CHECK();
    }
    return;
  }
  CHECK(dst_cache.data_type() == base::DataType::kDataTypeBF16)
      << "paged_kv_scatter_cu: dst_cache must be bf16 or int8";
  CHECK(dst_scale.is_empty() && dst_zero.is_empty())
      << "paged_kv_scatter_cu: bf16 dst_cache was passed with scale/zero pools";
  __nv_bfloat16* dst = const_cast<__nv_bfloat16*>(dst_cache.ptr<__nv_bfloat16>());
  if (stream_) {
    paged_kv_scatter_kernel_bf16<<<batch, 256, 0, stream_>>>(
        src.ptr<__nv_bfloat16>(), dst, block_table.ptr<int32_t>(), table_stride,
        positions.ptr<int32_t>(), kv_dim, num_blocks, block_size, layer_idx);
    CUDA_KERNEL_CHECK();
  } else {
    paged_kv_scatter_kernel_bf16<<<batch, 256>>>(
        src.ptr<__nv_bfloat16>(), dst, block_table.ptr<int32_t>(), table_stride,
        positions.ptr<int32_t>(), kv_dim, num_blocks, block_size, layer_idx);
    CUDA_KERNEL_CHECK();
  }
}

void paged_kv_residual_scatter_cu(const tensor::Tensor& src, const tensor::Tensor& residual,
                                  const tensor::Tensor& kv_slots,
                                  const tensor::Tensor& positions,
                                  const tensor::Tensor* seq_row_start, int32_t num_decode_rows,
                                  int32_t num_prefill_seqs, int32_t kv_dim, int32_t max_batch,
                                  int32_t group_size, int32_t layer_idx, void* stream) {
  CHECK(residual.data_type() == base::DataType::kDataTypeBF16)
      << "paged_kv_residual_scatter_cu: residual pool must be bf16";
  CHECK(!kv_slots.is_empty() && kv_slots.data_type() == base::DataType::kDataTypeInt32)
      << "paged_kv_residual_scatter_cu: kv_slots must be a non-empty int32 tensor";
  const int32_t batch = src.get_dim(0);
  __nv_bfloat16* dst = const_cast<__nv_bfloat16*>(residual.ptr<__nv_bfloat16>());
  const int32_t* row_start = seq_row_start != nullptr ? seq_row_start->ptr<int32_t>() : nullptr;
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  if (stream_) {
    paged_kv_residual_scatter_kernel_bf16<<<batch, 256, 0, stream_>>>(
        src.ptr<__nv_bfloat16>(), dst, kv_slots.ptr<int32_t>(), positions.ptr<int32_t>(),
        row_start, num_decode_rows, num_prefill_seqs, kv_dim, max_batch, group_size, layer_idx);
  } else {
    paged_kv_residual_scatter_kernel_bf16<<<batch, 256>>>(
        src.ptr<__nv_bfloat16>(), dst, kv_slots.ptr<int32_t>(), positions.ptr<int32_t>(),
        row_start, num_decode_rows, num_prefill_seqs, kv_dim, max_batch, group_size, layer_idx);
  }
  CUDA_KERNEL_CHECK();
}

void paged_kv_group_quantize_cu(const tensor::Tensor& src, const tensor::Tensor& residual,
                                const tensor::Tensor& dst_cache,
                                const tensor::Tensor& dst_scale,
                                const tensor::Tensor& dst_zero,
                                const tensor::Tensor& block_table,
                                const tensor::Tensor& kv_slots, const tensor::Tensor& positions,
                                const tensor::Tensor* seq_row_start, int32_t num_decode_rows,
                                int32_t num_prefill_seqs, int32_t kv_dim, int32_t max_batch,
                                int32_t group_size, int32_t channel_group,
                                int32_t channel_groups, int32_t num_blocks, int32_t block_size,
                                int32_t layer_idx, void* stream) {
  CHECK(src.data_type() == base::DataType::kDataTypeBF16)
      << "paged_kv_group_quantize_cu: src K rows must be bf16";
  CHECK(dst_cache.data_type() == base::DataType::kDataTypeInt8)
      << "paged_kv_group_quantize_cu: K pool must be int4 (int8 storage)";
  CHECK(residual.data_type() == base::DataType::kDataTypeBF16)
      << "paged_kv_group_quantize_cu: residual pool must be bf16";
  CHECK(!kv_slots.is_empty() && kv_slots.data_type() == base::DataType::kDataTypeInt32)
      << "paged_kv_group_quantize_cu: kv_slots must be a non-empty int32 tensor";
  CHECK_EQ(block_size % group_size, 0)
      << "S4 group_size=" << group_size << " must divide block_size=" << block_size;
  CHECK_EQ(kv_dim % channel_group, 0)
      << "S4 channel_group=" << channel_group << " must divide kv_dim=" << kv_dim;
  const int32_t batch = positions.get_dim(0);
  const int32_t table_stride = block_table.get_dim(1);
  // One block per row; it loops channel groups only when the row completes a
  // group. (A channel_groups-wide grid made every no-op row far too expensive
  // on the decode path.)
  const dim3 grid(static_cast<unsigned>(batch));
  const dim3 block(128);
  int8_t* dst = const_cast<int8_t*>(dst_cache.ptr<int8_t>());
  float* scale = const_cast<float*>(dst_scale.ptr<float>());
  float* zero = const_cast<float*>(dst_zero.ptr<float>());
  cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : nullptr;
  const int32_t* row_start = seq_row_start != nullptr ? seq_row_start->ptr<int32_t>() : nullptr;
  const __nv_bfloat16* src_ptr = src.ptr<__nv_bfloat16>();
  if (stream_) {
    paged_kv_group_quantize_kernel_bf16<<<grid, block, 0, stream_>>>(
        src_ptr, residual.ptr<__nv_bfloat16>(), dst, scale, zero, block_table.ptr<int32_t>(),
        table_stride, kv_slots.ptr<int32_t>(), positions.ptr<int32_t>(), row_start,
        num_decode_rows, num_prefill_seqs, kv_dim, max_batch, group_size, channel_group,
        channel_groups, num_blocks, block_size, layer_idx);
  } else {
    paged_kv_group_quantize_kernel_bf16<<<grid, block>>>(
        src_ptr, residual.ptr<__nv_bfloat16>(), dst, scale, zero, block_table.ptr<int32_t>(),
        table_stride, kv_slots.ptr<int32_t>(), positions.ptr<int32_t>(), row_start,
        num_decode_rows, num_prefill_seqs, kv_dim, max_batch, group_size, channel_group,
        channel_groups, num_blocks, block_size, layer_idx);
  }
  CUDA_KERNEL_CHECK();
}

// ========== Paged decode attention, split-KV over kv-head groups ==========
// One warp per (batch row, kv head, split), and each warp walks its split of
// the causal prefix for EVERY q head in the kv head's group: the K/V rows it
// loads stay in registers and are shared by all G = ceil(head_num /
// kv_head_num) heads of the group. A warp per (row, q head) — the layout this
// replaces — loads every KV row G times, so decode's L1/L2 (and, past the
// cache, DRAM) traffic scales with head_num / kv_head_num for no benefit.
//
// Split ranges are [split * per, min(seq, (split + 1) * per)) with `per`
// rounded up to 4 so every range starts on a group boundary; `num_splits` is
// derived from the KV capacity (see paged_decode_num_splits), never from the
// step's positions, so the launch configuration frozen into a captured decode
// graph stays correct as a batch's sequences grow. A warp whose range is empty
// publishes a neutral partial (m = -FLT_MAX, l = 0), which the combine's
// log-sum-exp drops exactly.
//
// S4 adds two reads on top of the bf16 path: K positions before the row's
// group boundary come from the packed int4 pool (dequantized with that token
// group's per-channel-group scale/zero), and the last partial group comes from
// the BF16 residual window. V is always packed (V is quantized immediately).
// QUANT is a compile-time arm split: the false instantiation is the
// pre-quantization kernel with the same registers and instructions.
//
// Partials are fp32 (acc[head_size] | m | l) at a stride of head_size + 4
// floats — the acc vector is written as one float4 per lane per chunk, so the
// row start must stay 16B aligned; the trailing 2-element pad keeps the stride
// a multiple of 4 floats. bf16 partials would round the running acc to 8 bits
// of mantissa on every split before the combine, which is exactly the
// precision the online softmax is meant to preserve. The fp32 rows are 2x the
// bf16 element count, which is what the callers size score_batch for.
template <int G, int CHUNKS, bool QUANT>
__global__ void __launch_bounds__(256, 1) paged_attn_split_kernel_bf16(
    const int32_t* __restrict__ positions, int32_t batch, const int32_t* __restrict__ block_table,
    int32_t table_stride, int32_t num_blocks, int32_t block_size, int32_t block_shift,
    int32_t layer_idx, int32_t dim, const __nv_bfloat16* __restrict__ query,
    float* __restrict__ partials, const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache, const int8_t* __restrict__ key_cache_i4,
    const int8_t* __restrict__ value_cache_i4, const float* __restrict__ key_scale,
    const float* __restrict__ key_zero, const float* __restrict__ value_scale,
    const float* __restrict__ value_zero, const __nv_bfloat16* __restrict__ key_residual,
    const int32_t* __restrict__ kv_slots, const int32_t* __restrict__ seq_row_start,
    int32_t num_decode_rows, int32_t num_prefill_seqs, int32_t kv_dim, int32_t kv_head_num,
    int32_t head_num, int32_t head_size, int32_t num_splits, int32_t head_groups,
    int32_t residual_batch, int32_t group_size, int32_t channel_group) {
  constexpr int kVec = 4;    // head dims per lane
  constexpr int kGroup = 4;  // positions per online-softmax step
  const int warp_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int warps_per_row = num_splits * head_groups;
  const int row_group = warp_id / warps_per_row;  // b * kv_head_num + kv_head
  const int rem = warp_id - row_group * warps_per_row;
  const int split = rem / head_groups;
  const int hgroup = rem - split * head_groups;
  const int b = row_group / kv_head_num;
  // The grid is a whole number of 8-warp blocks, so it can carry up to 7 warps
  // more than the launch needs: they own no row and must not index positions /
  // query / partials with b >= batch.
  if (b >= batch) return;
  const int kv_head = row_group - b * kv_head_num;
  const int pos = positions[b];
  const float scale = 1.f / sqrtf(static_cast<float>(head_size));

  // The q heads reading this kv head are [head_begin, head_end): the inverse
  // of the head_offset = head * kv_head_num / head_num mapping used per head,
  // and the same split paged_prefill_kernel_bf16 uses. With an uneven GQA
  // ratio these groups hold ceil or floor of head_num / kv_head_num heads.
  // This warp owns G of them starting at head_begin + hgroup * G; a head group
  // wider than G (head_size > 128 caps G at 4) is covered by several warps.
  const int head_begin = (kv_head * head_num + kv_head_num - 1) / kv_head_num;
  const int head_end = ((kv_head + 1) * head_num + kv_head_num - 1) / kv_head_num;
  const int my_begin = head_begin + hgroup * G;
  const int heads_here = min(G, head_end - my_begin);
  if (heads_here <= 0) return;  // hgroup past the group's last head

  // Head dims owned by this lane: chunk c carries [c*128, c*128 + 128) of the
  // row. Lanes past head_size contribute exact zeros (q, K and V alike), so
  // the padded tail of a short row costs nothing but lanes.
  const int d0 = lane * kVec;
  bool active[CHUNKS];
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    active[c] = (c * 128 + d0) < head_size;
  }

  const int seq = pos + 1;
  int per = (seq + num_splits - 1) / num_splits;
  per = (per + kGroup - 1) & ~(kGroup - 1);
  const int begin = min(split * per, seq);
  const int end = min(begin + per, seq);
  if (end <= begin) {
    // No work for this split: publish an empty partial. m is -FLT_MAX so the
    // combine's global max ignores it; l = 0 zeroes its weight exactly.
    float* empty =
        partials + (static_cast<int64_t>(b * head_num + my_begin) * num_splits + split) *
                       (head_size + 4);
    for (int i = 0; i < G; ++i) {
      if (i >= heads_here) break;
      float* row = empty + static_cast<int64_t>(i) * num_splits * (head_size + 4);
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        if (active[c]) {
          *reinterpret_cast<float4*>(row + c * 128 + d0) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
      if (lane == 0) {
        row[head_size] = -FLT_MAX;
        row[head_size + 1] = 0.f;
      }
    }
    return;
  }

  // fp32 q with the softmax scale folded in. The same representation serves
  // both arms: S4 dequantizes K/V while loading them, so the score is an
  // ordinary fp32 dot in the quantized instantiation too.
  float q[G][CHUNKS * kVec];
  const int64_t q_row = static_cast<int64_t>(b) * dim + static_cast<int64_t>(my_begin) * head_size;
#pragma unroll
  for (int i = 0; i < G; ++i) {
    if (i >= heads_here) break;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      const int d = c * 128 + d0;
      if (!active[c]) {
        for (int v = 0; v < kVec; ++v) q[i][c * kVec + v] = 0.f;
        continue;
      }
      const uint2 raw =
          *reinterpret_cast<const uint2*>(query + q_row + static_cast<int64_t>(i) * head_size + d);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      q[i][c * kVec + 0] = __bfloat162float(h[0].x) * scale;
      q[i][c * kVec + 1] = __bfloat162float(h[0].y) * scale;
      q[i][c * kVec + 2] = __bfloat162float(h[1].x) * scale;
      q[i][c * kVec + 3] = __bfloat162float(h[1].y) * scale;
    }
  }

  float m_i[G];  // online flash statistics, warp-uniform per head
  float l_i[G];
  // V's zero point is accumulated separately: with a per-token affine V, the
  // weighted sum is sum_j w_j*(q_j - zero_j)*scale_j, so the constant
  // -sum_j w_j*zero_j is a scalar per head that can be subtracted once at the
  // end instead of per value. (bf16 leaves it at zero.)
  float corr_i[G];
  float acc[G][CHUNKS * kVec];
#pragma unroll
  for (int i = 0; i < G; ++i) {
    m_i[i] = -FLT_MAX;
    l_i[i] = 0.f;
    corr_i[i] = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS * kVec; ++c) acc[i][c] = 0.f;
  }

  const int32_t* table_row = block_table + static_cast<int64_t>(b) * table_stride;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const int64_t kv_head_off = static_cast<int64_t>(kv_head) * head_size;
  // S4 metadata addressing. In the QUANT false instantiation none of these are
  // referenced (every use sits behind `if constexpr`), so the bf16 kernel keeps
  // its pre-quantization register file.
  //
  // K scales are per (token group, channel group): one lane owns exactly one
  // channel group (channel_group is a multiple of the 4 lane dims and divides
  // head_size), and the 4 positions of one step share a token group because
  // group_size is a multiple of 4 and the split ranges are 4-aligned. A single
  // scale/zero load therefore serves the whole 4-position, up-to-8-head step.
  //
  // V scales stay per (token, kv head), four positions at a time.
  const int slot = QUANT ? __ldg(kv_slots + b) : 0;
  // Quantized-token boundary for this row. It must use the sequence's LAST
  // position in this step, not this row's own: in a prefill chunk a group can
  // complete at a row after this one, and that group is already in the int4
  // pool by the time any attention runs. Clamp to pos + 1 because positions
  // past the query are not read anyway (causality).
  int last_pos = pos;
  if constexpr (QUANT) {
    if (seq_row_start != nullptr && b >= num_decode_rows) {
      for (int s = 0; s < num_prefill_seqs; ++s) {
        const int rs = seq_row_start[s];
        const int re = seq_row_start[s + 1];
        if (b >= rs && b < re) {
          last_pos = positions[re - 1];
          break;
        }
      }
    }
  }
  const int boundary =
      QUANT ? min(((last_pos + 1) / group_size) * group_size, pos + 1) : 0;
  const int groups_per_block = QUANT ? (block_size / group_size) : 1;
  const int channel_groups = QUANT ? (kv_dim / channel_group) : 1;
  const int lane_cg = QUANT ? ((static_cast<int>(kv_head_off) + d0) / channel_group) : 0;
  const int64_t key_meta_layer_base =
      QUANT ? static_cast<int64_t>(layer_idx) * num_blocks * groups_per_block * channel_groups : 0;
  const int64_t value_meta_layer_base =
      QUANT ? static_cast<int64_t>(layer_idx) * num_blocks * kv_head_num * block_size : 0;
  const int64_t residual_layer_base =
      QUANT ? static_cast<int64_t>(layer_idx) * residual_batch * group_size * kv_dim : 0;

  int64_t row[kGroup];
  int gpos = begin;
  for (; gpos + kGroup <= end; gpos += kGroup) {
    const int32_t pg = __ldg(table_row + (gpos >> block_shift));
    const int off = gpos & (block_size - 1);
    const int64_t row0 = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                         kv_head_off + static_cast<int64_t>(off) * kv_dim + d0;
#pragma unroll
    for (int j = 0; j < kGroup; ++j) row[j] = row0 + static_cast<int64_t>(j) * kv_dim;

    float ks = 1.f;
    float kz = 0.f;
    float vs[kGroup];
    float vz[kGroup];
    if constexpr (QUANT) {
      // The whole 4-position step is either quantized or residual: the
      // boundary is a multiple of kGroup, so no step straddles it.
      if (gpos < boundary) {
        const int64_t kidx = key_meta_layer_base +
                             (static_cast<int64_t>(pg) * groups_per_block + off / group_size) *
                                 channel_groups +
                             lane_cg;
        ks = __ldg(key_scale + kidx);
        kz = __ldg(key_zero + kidx);
      }
      const int64_t srow = value_meta_layer_base +
                           (static_cast<int64_t>(pg) * kv_head_num + kv_head) * block_size + off;
      // The pos index is the innermost V-metadata dimension and gpos is
      // 4-aligned, so these are one 16-byte-aligned float4 each.
      const float4 ts = *reinterpret_cast<const float4*>(value_scale + srow);
      const float4 tz = *reinterpret_cast<const float4*>(value_zero + srow);
      vs[0] = ts.x;
      vs[1] = ts.y;
      vs[2] = ts.z;
      vs[3] = ts.w;
      vz[0] = tz.x;
      vz[1] = tz.y;
      vz[2] = tz.z;
      vz[3] = tz.w;
    }

    float v_reg[kGroup][CHUNKS * kVec];
    float kf[kGroup][CHUNKS * kVec];
#pragma unroll
    for (int j = 0; j < kGroup; ++j) {
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        if (!active[c]) {
          for (int v = 0; v < kVec; ++v) {
            kf[j][c * kVec + v] = 0.f;
            v_reg[j][c * kVec + v] = 0.f;
          }
          continue;
        }
        if constexpr (QUANT) {
          if (gpos < boundary) {
            load_i4_raw_chunk4(kf[j] + c * kVec, key_cache_i4, row[j] + c * 128);
          } else {
            // K of the incomplete tail group comes from the BF16 residual
            // window; V below still comes from the packed pool (V is
            // quantized immediately, with no residual).
            const int rp = (gpos + j) % group_size;
            const int64_t roff = residual_layer_base +
                                 static_cast<int64_t>(slot) * group_size * kv_dim +
                                 static_cast<int64_t>(rp) * kv_dim + kv_head_off + d0 + c * 128;
            load_bf16_chunk4(kf[j] + c * kVec, key_residual, roff);
          }
          load_i4_raw_chunk4(v_reg[j] + c * kVec, value_cache_i4, row[j] + c * 128);
        } else {
          load_bf16_chunk4(kf[j] + c * kVec, key_cache, row[j] + c * 128);
          load_bf16_chunk4(v_reg[j] + c * kVec, value_cache, row[j] + c * 128);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < G; ++i) {
      if (i >= heads_here) break;
      float s[kGroup];
      const bool k_from_pool = QUANT && (gpos < boundary);
      // sum(q) over this lane's dims: the affine is
      //   sum_d q_d*(code_d - zero)*scale = scale*(dot - zero * sum_d q_d),
      // so the zero-point term multiplies sum(q), NOT sum(code). q is fixed
      // across the four positions, so this is hoisted out of the j loop.
      float sumq = q[i][0] + q[i][1] + q[i][2] + q[i][3];
#pragma unroll
      for (int c = 1; c < CHUNKS; ++c) {
        const int o = c * kVec;
        sumq += q[i][o + 0] + q[i][o + 1] + q[i][o + 2] + q[i][o + 3];
      }
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        float dot = q[i][0] * kf[j][0] + q[i][1] * kf[j][1] + q[i][2] * kf[j][2] +
                    q[i][3] * kf[j][3];
#pragma unroll
        for (int c = 1; c < CHUNKS; ++c) {
          const int o = c * kVec;
          dot += q[i][o + 0] * kf[j][o + 0] + q[i][o + 1] * kf[j][o + 1] +
                 q[i][o + 2] * kf[j][o + 2] + q[i][o + 3] * kf[j][o + 3];
        }
        // Raw int4 codes: the affine is per lane (one channel group), so it
        // lands here, before the cross-lane reduction. bf16/residual K needs
        // neither term.
        s[j] = k_from_pool ? (dot - kz * sumq) * ks : dot;
      }
      // Independent across j, so the four reductions overlap.
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
#pragma unroll
        for (int off = 16; off; off >>= 1) s[j] += __shfl_xor_sync(0xffffffffu, s[j], off);
      }
      float m_new = m_i[i];
#pragma unroll
      for (int j = 0; j < kGroup; ++j) m_new = fmaxf(m_new, s[j]);
      const float alpha = __expf(m_i[i] - m_new);
      float l_new = l_i[i] * alpha;
      float p[kGroup];
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        p[j] = __expf(s[j] - m_new);
        l_new += p[j];
      }
      // V's affine transform is folded into the weight: w_j = p_j * scale_j,
      // and the constant w_j * zero_j is carried in corr_i. l stays the sum of
      // the RAW weights, as required by the normalized output.
      float w[kGroup];
      float corr_add = 0.f;
#pragma unroll
      for (int j = 0; j < kGroup; ++j) {
        if constexpr (QUANT) {
          w[j] = p[j] * vs[j];
          corr_add += w[j] * vz[j];
        } else {
          w[j] = p[j];
        }
      }
#pragma unroll
      for (int c = 0; c < CHUNKS * kVec; ++c) {
        float a = acc[i][c] * alpha;
#pragma unroll
        for (int j = 0; j < kGroup; ++j) a += w[j] * v_reg[j][c];
        acc[i][c] = a;
      }
      corr_i[i] = corr_i[i] * alpha + corr_add;
      m_i[i] = m_new;
      l_i[i] = l_new;
    }
  }
  for (; gpos < end; ++gpos) {
    const int32_t pg = __ldg(table_row + (gpos >> block_shift));
    const int32_t poff = gpos & (block_size - 1);
    const int64_t row = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                        kv_head_off + static_cast<int64_t>(poff) * kv_dim + d0;
    float ks = 1.f;
    float kz = 0.f;
    float vs = 1.f;
    float vz = 0.f;
    if constexpr (QUANT) {
      const int64_t srow = value_meta_layer_base +
                           (static_cast<int64_t>(pg) * kv_head_num + kv_head) * block_size + poff;
      vs = __ldg(value_scale + srow);
      vz = __ldg(value_zero + srow);
      if (gpos < boundary) {
        const int64_t kidx = key_meta_layer_base +
                             (static_cast<int64_t>(pg) * groups_per_block + poff / group_size) *
                                 channel_groups +
                             lane_cg;
        ks = __ldg(key_scale + kidx);
        kz = __ldg(key_zero + kidx);
      }
    }
    float v_reg[CHUNKS * kVec];
    float kf[CHUNKS * kVec];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      if (!active[c]) {
        for (int v = 0; v < kVec; ++v) {
          kf[c * kVec + v] = 0.f;
          v_reg[c * kVec + v] = 0.f;
        }
        continue;
      }
      if constexpr (QUANT) {
        if (gpos < boundary) {
          load_i4_raw_chunk4(kf + c * kVec, key_cache_i4, row + c * 128);
        } else {
          const int rp = gpos % group_size;
          const int64_t roff = residual_layer_base +
                               static_cast<int64_t>(slot) * group_size * kv_dim +
                               static_cast<int64_t>(rp) * kv_dim + kv_head_off + d0 + c * 128;
          load_bf16_chunk4(kf + c * kVec, key_residual, roff);
        }
        load_i4_raw_chunk4(v_reg + c * kVec, value_cache_i4, row + c * 128);
      } else {
        load_bf16_chunk4(kf + c * kVec, key_cache, row + c * 128);
        load_bf16_chunk4(v_reg + c * kVec, value_cache, row + c * 128);
      }
    }
#pragma unroll
    for (int i = 0; i < G; ++i) {
      if (i >= heads_here) break;
      float dot = q[i][0] * kf[0] + q[i][1] * kf[1] + q[i][2] * kf[2] + q[i][3] * kf[3];
#pragma unroll
      for (int c = 1; c < CHUNKS; ++c) {
        const int o = c * kVec;
        dot += q[i][o + 0] * kf[o + 0] + q[i][o + 1] * kf[o + 1] + q[i][o + 2] * kf[o + 2] +
               q[i][o + 3] * kf[o + 3];
      }
      float sumq = q[i][0] + q[i][1] + q[i][2] + q[i][3];
#pragma unroll
      for (int c = 1; c < CHUNKS; ++c) {
        const int o = c * kVec;
        sumq += q[i][o + 0] + q[i][o + 1] + q[i][o + 2] + q[i][o + 3];
      }
      float s = (QUANT && gpos < boundary) ? (dot - kz * sumq) * ks : dot;
#pragma unroll
      for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
      const float m_new = fmaxf(m_i[i], s);
      const float alpha = __expf(m_i[i] - m_new);
      const float pr = __expf(s - m_new);
      l_i[i] = l_i[i] * alpha + pr;
      // Same folded affine as the group loop.
      float w = pr;
      float corr_add = 0.f;
      if constexpr (QUANT) {
        w = pr * vs;
        corr_add = w * vz;
      }
      corr_i[i] = corr_i[i] * alpha + corr_add;
#pragma unroll
      for (int c = 0; c < CHUNKS * kVec; ++c) acc[i][c] = acc[i][c] * alpha + w * v_reg[c];
      m_i[i] = m_new;
    }
  }

  // Unnormalized acc: l is published alongside it so the combine can do one
  // global rescale over all splits.
#pragma unroll
  for (int i = 0; i < G; ++i) {
    if (i >= heads_here) break;
    float* out_partial =
        partials + (static_cast<int64_t>(b * head_num + my_begin + i) * num_splits + split) *
                       (head_size + 4);
    // Fold the V zero-point correction into the numerator before publishing.
    // corr_i is exactly 0 in the bf16 arm, so this is a no-op there.
#pragma unroll
    for (int c = 0; c < CHUNKS * kVec; ++c) acc[i][c] -= corr_i[i];
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
      if (active[c]) {
        const int o = c * kVec;
        *reinterpret_cast<float4*>(out_partial + c * 128 + d0) =
            make_float4(acc[i][o + 0], acc[i][o + 1], acc[i][o + 2], acc[i][o + 3]);
      }
    }
    if (lane == 0) {
      out_partial[head_size] = m_i[i];
      out_partial[head_size + 1] = l_i[i];
    }
  }
}

// Log-sum-exp combine over the num_splits partials of one (row, q head) pair:
// o = sum_s exp(m_s - m) * acc_s / sum_s exp(m_s - m) * l_s. One block per
// pair, one output dim per thread; kThreads must cover head_size.
template <int kThreads>
__global__ void paged_attn_split_combine_kernel_bf16(const float* __restrict__ partials,
                                                     __nv_bfloat16* __restrict__ output,
                                                     int32_t head_num, int32_t head_size,
                                                     int32_t num_splits, int32_t batch) {
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float s_global_m;
  const int pair = blockIdx.x;
  if (pair >= batch * head_num) return;
  const int d = threadIdx.x;
  const float* base = partials + static_cast<int64_t>(pair) * num_splits * (head_size + 4);

  // Global max over splits (every thread reads the same values; the reduction
  // spreads the num_splits maxima over the block).
  float local_m = -FLT_MAX;
  for (int s = d; s < num_splits; s += kThreads) {
    local_m = fmaxf(local_m, __ldg(base + static_cast<int64_t>(s) * (head_size + 4) + head_size));
  }
  const float global_m = BlockReduce(temp).Reduce(local_m, cub::Max());
  if (threadIdx.x == 0) {
    s_global_m = global_m;
  }
  __syncthreads();
  const float m = s_global_m;

  float o = 0.f;
  float l = 0.f;
  for (int s = 0; s < num_splits; ++s) {
    const float* ps = base + static_cast<int64_t>(s) * (head_size + 4);
    // Empty splits carry m = -FLT_MAX, so w underflows to exactly 0 and the
    // slot drops out of both sums.
    const float w = __expf(ps[head_size] - m);
    l += w * ps[head_size + 1];
    if (d < head_size) {
      o += w * __ldg(ps + d);
    }
  }
  if (d < head_size) {
    output[static_cast<int64_t>(pair) * head_size + d] =
        __float2bfloat16(l > 0.f ? o / l : 0.f);
  }
}

// Paged kernels map a token position to its page with `pos >> block_shift`;
// the block_size CHECK in paged_attention_cu_batch already guarantees a power
// of two, so a shift is exact (and cheaper than the divide).
static int block_shift_for(int32_t block_size) {
  int shift = 0;
  while ((1 << shift) < block_size) ++shift;
  return shift;
}

// Launch paged_attn_split_kernel_bf16 for the (q heads per warp, 128-dim
// chunks per row) pair paged_attention_cu_batch picked. Seven instantiations:
// 1/2/4/8 heads per warp for head_size <= 128 and 1/2/4 for head_size <= 256,
// where the second chunk doubles the per-head register footprint. QUANT picks
// the arm split: the bf16 callers reach only the QUANT false instantiations,
// which carry no int4 code.
template <bool QUANT>
static void launch_paged_attn_split(int32_t heads_per_warp, int32_t chunks, unsigned grid,
                                    const int32_t* positions, int32_t batch,
                                    const int32_t* block_table, int32_t table_stride,
                                    int32_t num_blocks, int32_t block_size, int32_t block_shift,
                                    int32_t layer_idx, int32_t dim, const __nv_bfloat16* query,
                                    float* partials, const __nv_bfloat16* key_cache,
                                    const __nv_bfloat16* value_cache, const int8_t* key_cache_i4,
                                    const int8_t* value_cache_i4, const float* key_scale,
                                    const float* key_zero, const float* value_scale,
                                    const float* value_zero, const __nv_bfloat16* key_residual,
                                    const int32_t* kv_slots, const int32_t* seq_row_start,
                                    int32_t num_decode_rows, int32_t num_prefill_seqs,
                                    int32_t kv_dim, int32_t kv_head_num, int32_t head_num,
                                    int32_t head_size, int32_t num_splits, int32_t head_groups,
                                    int32_t residual_batch, int32_t group_size,
                                    int32_t channel_group, cudaStream_t stream) {
#define LAUNCH_PAGED_SPLIT(G, C)                                                                \
  paged_attn_split_kernel_bf16<G, C, QUANT><<<grid, 256, 0, stream>>>(                      \
      positions, batch, block_table, table_stride, num_blocks, block_size, block_shift,         \
      layer_idx, dim, query, partials, key_cache, value_cache, key_cache_i4, value_cache_i4,    \
      key_scale, key_zero, value_scale, value_zero, key_residual, kv_slots, seq_row_start,    \
      num_decode_rows, num_prefill_seqs, kv_dim, kv_head_num, head_num, head_size, num_splits, \
      head_groups, residual_batch, group_size, channel_group)
  if (chunks == 1) {
    switch (heads_per_warp) {
      case 1: LAUNCH_PAGED_SPLIT(1, 1); break;
      case 2: LAUNCH_PAGED_SPLIT(2, 1); break;
      case 4: LAUNCH_PAGED_SPLIT(4, 1); break;
      default: LAUNCH_PAGED_SPLIT(8, 1); break;
    }
  } else {
    switch (heads_per_warp) {
      case 1: LAUNCH_PAGED_SPLIT(1, 2); break;
      case 2: LAUNCH_PAGED_SPLIT(2, 2); break;
      default: LAUNCH_PAGED_SPLIT(4, 2); break;
    }
  }
#undef LAUNCH_PAGED_SPLIT
}

void paged_attention_cu_batch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, const tensor::Tensor& positions,
                              const tensor::Tensor& block_table, const tensor::Tensor& query_batch,
                              tensor::Tensor& score_batch, const tensor::Tensor& mha_out,
                              const tensor::Tensor& key_cache, const tensor::Tensor& value_cache,
                              const tensor::Tensor& key_scale, const tensor::Tensor& value_scale,
                              const tensor::Tensor& key_zero, const tensor::Tensor& value_zero,
                              const tensor::Tensor& key_residual, const tensor::Tensor& kv_slots,
                              const tensor::Tensor* seq_row_start, int32_t num_decode_rows,
                              int32_t num_prefill_seqs, int32_t group_size,
                              int32_t channel_group, CudaConfig* config) {
  const bool quantized =
      check_kv_pools(key_cache, value_cache, key_scale, key_zero, value_scale, value_zero,
                     key_residual, "paged_attention_cu_batch");
  CHECK(!quantized || (!kv_slots.is_empty() &&
                       kv_slots.data_type() == base::DataType::kDataTypeInt32))
      << "paged_attention_cu_batch: S4 needs a non-empty int32 kv_slots tensor";
  const int32_t batch = query_batch.get_dim(0);
  const int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Per-seq KV capacity derives from the block table width (continuous
  // layout: stride 1 x max_seq_len).
  const int32_t max_seq_len = table_stride * block_size;
  CHECK(config != nullptr);
  CHECK(paged_decode_geometry_ok(head_size, table_stride, block_size))
      << "paged decode attention needs a head_size that is a multiple of 8 and <= 256, a "
      << "block table (table_stride > 0) and a power-of-two block_size >= 4; got head_size="
      << head_size << " table_stride=" << table_stride << " block_size=" << block_size;

  // One warp per (row, kv head, split, head sub-group), each walking its split
  // for up to `heads_per_warp` q heads of the group — see
  // paged_attn_split_kernel_bf16. num_splits already folds in the GQA group
  // size, so the warp count comes out at batch * head_num *
  // flash_decoding_num_splits(capacity): the parallelism of the per-q-head
  // split it replaces, with each K/V row loaded once per warp. Splitting is
  // also what fills the device at batch 1 (32 warps over 108 SMs without it)
  // and what keeps a mixed batch from costing max(seq) instead of mean(seq).
  const int32_t num_splits = paged_decode_num_splits(max_seq_len, head_num, kv_head_num);
  const int32_t chunks = (head_size + 127) / 128;
  const int32_t head_group = (head_num + kv_head_num - 1) / kv_head_num;
  // A warp holds at most 8 q heads — 4 once the row needs two 128-dim chunks,
  // which doubles the per-head register footprint. Wider head groups (an
  // extreme GQA ratio) are covered by several warps over the same split, so
  // the K/V row they share is still loaded once per warp.
  const int32_t warp_cap = chunks == 1 ? 8 : 4;
  int32_t heads_per_warp = 1;
  while (heads_per_warp * 2 <= head_group && heads_per_warp * 2 <= warp_cap) {
    heads_per_warp *= 2;
  }
  const int32_t head_groups = (head_group + heads_per_warp - 1) / heads_per_warp;

  // Partials are fp32 rows of (head_size + 4) floats: a bf16 (o | m | l) row
  // would round each split's running acc before the combine. Callers size
  // score_batch with flash_decoding_partials_elements; a buffer that is too
  // small would leave the combine reading stale partials, so fail here instead.
  const int64_t partial_floats =
      static_cast<int64_t>(batch) * head_num * num_splits * (head_size + 4);
  CHECK(static_cast<int64_t>(score_batch.size()) * 2 >= partial_floats * 4)
      << "score_batch holds " << score_batch.size() << " elements, too few for the fp32 "
      << "split-KV partials of batch=" << batch << " head_num=" << head_num
      << " num_splits=" << num_splits << " head_size=" << head_size
      << "; size it with flash_decoding_partials_elements().";

  const int32_t block_shift = block_shift_for(block_size);
  const int64_t total_warps = static_cast<int64_t>(batch) * kv_head_num * num_splits * head_groups;
  CHECK_LE(total_warps, static_cast<int64_t>(1) << 31)
      << "paged decode launch would need " << total_warps << " warps";
  constexpr int kWarpsPerBlock = 8;
  const unsigned grid = static_cast<unsigned>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  float* partials = reinterpret_cast<float*>(const_cast<__nv_bfloat16*>(score_batch.ptr<__nv_bfloat16>()));
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  cudaStream_t stream = config->stream;

  if (quantized) {
    CHECK(group_size > 0 && block_size % group_size == 0 && group_size % 4 == 0)
        << "paged_attention_cu_batch: group_size=" << group_size
        << " must be a multiple of 4 dividing block_size=" << block_size;
    CHECK(channel_group > 0 && kv_dim % channel_group == 0 && channel_group % 4 == 0)
        << "paged_attention_cu_batch: channel_group=" << channel_group
        << " must be a multiple of 4 dividing kv_dim=" << kv_dim;
  }
  // Exactly one of the two cache views is meaningful: the int8 one is built
  // from a different element type and cannot be aliased from the bf16 one, so
  // the unused set is passed as nullptr and the kernel selects on it.
  const bool kq = quantized;
  const __nv_bfloat16* kcache = kq ? nullptr : key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = kq ? nullptr : value_cache.ptr<__nv_bfloat16>();
  const int8_t* kcache_i4 = kq ? key_cache.ptr<int8_t>() : nullptr;
  const int8_t* vcache_i4 = kq ? value_cache.ptr<int8_t>() : nullptr;
  const float* kscale = kq ? key_scale.ptr<float>() : nullptr;
  const float* kzero = kq ? key_zero.ptr<float>() : nullptr;
  const float* vscale = kq ? value_scale.ptr<float>() : nullptr;
  const float* vzero = kq ? value_zero.ptr<float>() : nullptr;
  const __nv_bfloat16* kresidual = kq ? key_residual.ptr<__nv_bfloat16>() : nullptr;
  const int32_t* slots = kq ? kv_slots.ptr<int32_t>() : nullptr;
  const int32_t* row_start =
      (kq && seq_row_start != nullptr) ? seq_row_start->ptr<int32_t>() : nullptr;
  // key_residual is [num_layers, max_batch, group_size, kv_dim].
  const int32_t residual_batch = kq ? key_residual.get_dim(1) : 0;

#define PAGED_SPLIT_ARGS                                                                          \
  heads_per_warp, chunks, grid, positions.ptr<int32_t>(), batch, block_table.ptr<int32_t>(),      \
      table_stride, num_blocks, block_size, block_shift, layer_idx, dim, query, partials, kcache, \
      vcache, kcache_i4, vcache_i4, kscale, kzero, vscale, vzero, kresidual, slots, row_start,    \
      num_decode_rows, num_prefill_seqs, kv_dim, kv_head_num, head_num, head_size, num_splits,    \
      head_groups, residual_batch, group_size, channel_group, stream
  if (kq) {
    launch_paged_attn_split<true>(PAGED_SPLIT_ARGS);
  } else {
    launch_paged_attn_split<false>(PAGED_SPLIT_ARGS);
  }
#undef PAGED_SPLIT_ARGS
  CUDA_KERNEL_CHECK();

  // Combine pass: one block per (row, head) pair, one output dim per thread.
  const dim3 combine_grid(static_cast<unsigned>(batch) * head_num);
  if (head_size <= 64) {
    paged_attn_split_combine_kernel_bf16<64><<<combine_grid, 64, 0, stream>>>(
        partials, output, head_num, head_size, num_splits, batch);
  } else if (head_size <= 128) {
    paged_attn_split_combine_kernel_bf16<128><<<combine_grid, 128, 0, stream>>>(
        partials, output, head_num, head_size, num_splits, batch);
  } else {
    paged_attn_split_combine_kernel_bf16<256><<<combine_grid, 256, 0, stream>>>(
        partials, output, head_num, head_size, num_splits, batch);
  }
  CUDA_KERNEL_CHECK();
}

// ========== Paged prefill attention (FA2-style, one KV read per tile) =======
// One 256-thread block per (q block of B_q = 8 rows, prefill sequence, kv
// head): warp w owns row q_begin + w and all G = head_num / kv_head_num q
// heads that read that kv head. The K/V rows of a kv tile are staged into
// smem once per (CTA, tile) and consumed by every warp and every q head of
// the group, so a prefill chunk's KV traffic drops from the decode kernel's
// O(head_num * N^2 / 2) (each row re-reads its whole causal prefix once per q
// head) to O(kv_head_num * ceil(N / B_q) * N / 2) — for Qwen3-4B (H=32,
// kv_head=8) and B_q = 8 that is 32x less traffic per layer.
//
// The per-(row, head, position) arithmetic is deliberately the same sequence
// of fp32 ops as paged_attn_split_kernel_bf16: 4 contiguous head dims per
// lane, the softmax scale folded into q, groups of 4 positions sharing one
// rescale, a 5-step shfl_xor score reduction, and the same < 4-position tail.
// A row run here and the same row run by a single (unsplit) decode warp
// therefore agree to the last bit; the split decode path adds the combine's
// log-sum-exp on top, which is exact only up to the fp32 rounding of one
// rescale, so mixed steps stay within a few ulps of an unsplit run — every
// sampled token was unchanged across both paths (verify_tokens A/B).
template <int G, bool QUANT>
__global__ void paged_prefill_kernel_bf16(
    const int32_t* __restrict__ positions, const int32_t* __restrict__ seq_row_start,
    const int32_t* __restrict__ block_table, int32_t table_stride, int32_t num_blocks,
    int32_t block_size, int32_t block_shift, int32_t layer_idx, int32_t dim,
    const __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ key_cache, const __nv_bfloat16* __restrict__ value_cache,
    const int8_t* __restrict__ key_cache_i4, const int8_t* __restrict__ value_cache_i4,
    const float* __restrict__ key_scale, const float* __restrict__ key_zero,
    const float* __restrict__ value_scale, const float* __restrict__ value_zero,
    const __nv_bfloat16* __restrict__ key_residual, const int32_t* __restrict__ kv_slots,
    int32_t kv_dim, int32_t kv_head_num, int32_t head_num, int32_t head_size,
    int32_t residual_batch, int32_t group_size, int32_t channel_group) {
  constexpr int kVec = 4;       // head dims per lane (one uint2 load per row)
  constexpr int kGroup = 4;     // positions sharing one softmax rescale (as in decode)
  constexpr int kTileKv = 64;   // KV rows staged per tile
  constexpr int kLoadVec = 8;   // bf16 per loader vector (16B, one uint4)
  constexpr int kChunks = 128 / kLoadVec;  // loader vectors per row (head_size == 128)
  constexpr int kInt4Vec = 4;   // dims per loader vector in a packed int4 row (2B)
  constexpr int kInt4Chunks = 128 / kInt4Vec;  // ... same row, int4 arm
  constexpr int kPad = 136;     // 136 * 2B = 272B: rows stay 16B aligned
  __shared__ __align__(16) __nv_bfloat16 s_k[kTileKv][kPad];
  __shared__ __align__(16) __nv_bfloat16 s_v[kTileKv][kPad];
  // Per-position V scales and zero points (one pair per staged row, since the
  // CTA covers one kv head). K scales are per (token group, channel group) and
  // therefore lane-dependent, so they are loaded directly in the compute loop
  // rather than staged. Sized away in the QUANT false instantiation so its
  // smem footprint is the pre-quantization kernel's.
  __shared__ float s_vs[QUANT ? kTileKv : 1];
  __shared__ float s_vz[QUANT ? kTileKv : 1];
  // Physical page of each staged row (int4 K metadata is addressed through it).
  __shared__ int s_pg[QUANT ? kTileKv : 1];
  __shared__ int s_max_pos;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = blockDim.x >> 5;
  // The quantized arm stages the raw packed rows in the same storage — they
  // need half of it — and dequantizes in the compute loop. Dequantizing on the
  // way in (int4 -> bf16 in smem) would round every K/V value to a 4-bit
  // mantissa, and it would put the prefill and the decode arm of the same row
  // on different arithmetic. A 136-byte int4 row keeps the same alignment the
  // bf16 row's uint2 reads use. As in the decode kernel, only the QUANT true
  // instantiation has both arms compiled in at all.
  int8_t (*s_k4)[kPad] = reinterpret_cast<int8_t (*)[kPad]>(&s_k[0][0]);
  int8_t (*s_v4)[kPad] = reinterpret_cast<int8_t (*)[kPad]>(&s_v[0][0]);

  const int seq = blockIdx.y;
  const int kv_head = blockIdx.z;
  const int row_start = seq_row_start[seq];
  const int row_end = seq_row_start[seq + 1];
  const int q_begin = row_start + blockIdx.x * nwarps;  // one q row per warp
  // grid.x is sized from the batch's total prefill rows, so blocks past a
  // short sequence's end exist and exit here (a few hundred ns each).
  if (q_begin >= row_end) return;
  const int row_stop = min(q_begin + nwarps, row_end);
  const int my_row = q_begin + warp;
  const bool active = my_row < row_stop;
  const int limit = active ? positions[my_row] + 1 : 0;  // positions [0, limit)

  // CTA-wide causal limit: no position beyond this needs to be staged. Rows
  // are normally consecutive chunk tokens, but nothing guarantees it (a
  // rewritten sequence can hold stale rows), so take a real max.
  if (tid == 0) {
    int m = 0;
    for (int r = q_begin; r < row_stop; ++r) m = max(m, positions[r] + 1);
    s_max_pos = m;
  }
  __syncthreads();
  const int tiles = (s_max_pos + kTileKv - 1) / kTileKv;

  const float scale = 1.f / sqrtf(static_cast<float>(head_size));
  // q heads served by this kv head; G may exceed the exact group size when
  // head_num is not a multiple of kv_head_num (head_num=36 / 8 -> 5,5,4,...).
  const int head_begin = (kv_head * head_num + kv_head_num - 1) / kv_head_num;
  const int head_end = ((kv_head + 1) * head_num + kv_head_num - 1) / kv_head_num;
  const int heads_here = head_end - head_begin;

  float q[G][kVec];
  if (active) {
#pragma unroll
    for (int i = 0; i < G; ++i) {
      if (i >= heads_here) break;
      const __nv_bfloat16* q_head =
          query + static_cast<int64_t>(my_row) * dim + (head_begin + i) * head_size + lane * kVec;
      const uint2 raw = *reinterpret_cast<const uint2*>(q_head);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
      q[i][0] = __bfloat162float(h[0].x) * scale;
      q[i][1] = __bfloat162float(h[0].y) * scale;
      q[i][2] = __bfloat162float(h[1].x) * scale;
      q[i][3] = __bfloat162float(h[1].y) * scale;
    }
  }

  float m_i[G];
  float l_i[G];
  // V zero-point correction (see the decode kernel): a scalar per head,
  // subtracted from the accumulated numerator once at the end.
  float corr_i[G];
  float acc[G][kVec];
#pragma unroll
  for (int i = 0; i < G; ++i) {
    m_i[i] = -FLT_MAX;
    l_i[i] = 0.f;
    corr_i[i] = 0.f;
#pragma unroll
    for (int c = 0; c < kVec; ++c) acc[i][c] = 0.f;
  }

  // All rows of a prefill sequence share one block table row (the scheduler
  // writes it per batch row from the sequence's page table), so the CTA reads
  // the first row's copy. All rows also share the sequence slot the residual
  // window is indexed by.
  const int32_t* table_row = block_table + static_cast<int64_t>(row_start) * table_stride;
  const int64_t layer_base = static_cast<int64_t>(layer_idx) * num_blocks * block_size * kv_dim;
  const int64_t value_meta_layer_base =
      static_cast<int64_t>(layer_idx) * num_blocks * kv_head_num * block_size;
  const int kv_head_off = (head_begin * kv_head_num / head_num) * head_size;
  // S4 addressing (QUANT only). K's boundary is per warp row: positions below
  // it are packed, the rest of the row's prefix is the BF16 residual window.
  const int slot = QUANT ? __ldg(kv_slots + row_start) : 0;
  // Same rule as the decode kernel: the boundary is the whole sequence
  // chunk's quantized length, clamped to this row's causal prefix. Groups
  // completed anywhere in this chunk are in the int4 pool before attention.
  const int boundary =
      (QUANT && active)
          ? min(((positions[row_end - 1] + 1) / group_size) * group_size, positions[my_row] + 1)
          : 0;
  const int groups_per_block = QUANT ? (block_size / group_size) : 1;
  const int channel_groups = QUANT ? (kv_dim / channel_group) : 1;
  const int lane_cg = QUANT ? ((kv_head_off + lane * kVec) / channel_group) : 0;
  const int64_t key_meta_layer_base =
      QUANT ? static_cast<int64_t>(layer_idx) * num_blocks * groups_per_block * channel_groups : 0;
  const int64_t residual_layer_base =
      QUANT ? static_cast<int64_t>(layer_idx) * residual_batch * group_size * kv_dim +
                  static_cast<int64_t>(slot) * group_size * kv_dim
            : 0;

  // Loader geometry: the int4 arm moves 4 elements per access (a uint16 of two
  // packed bytes) and the bf16 arm moves 8 (a uint4 of 8 bf16), so their chunk
  // counts and tile addresses differ. `ch` is a ROW OFFSET IN ELEMENTS in both
  // arms; the packed tile address divides it by two (dim d lives in byte d/2).
  const int chunks = QUANT ? kInt4Chunks : kChunks;
  for (int t = 0; t < tiles; ++t) {
    const int tile_pos = t * kTileKv;
    const int tile_n = min(kTileKv, s_max_pos - tile_pos);
    if constexpr (QUANT) {
      // One thread per row stages that row's V scale/zero pair and page id.
      for (int r = tid; r < tile_n; r += blockDim.x) {
        const int gpos = tile_pos + r;
        const int32_t pg = __ldg(table_row + (gpos >> block_shift));
        s_pg[r] = pg;
        const int64_t srow = value_meta_layer_base +
                             (static_cast<int64_t>(pg) * kv_head_num + kv_head) * block_size +
                             (gpos & (block_size - 1));
        s_vs[r] = __ldg(value_scale + srow);
        s_vz[r] = __ldg(value_zero + srow);
      }
    }
    {
      const int chunk = tid % chunks;
      const int ch = chunk * (QUANT ? kInt4Vec : kLoadVec);
      for (int r = tid / chunks; r < tile_n; r += blockDim.x / chunks) {
        const int gpos = tile_pos + r;
        const int32_t pg = __ldg(table_row + (gpos >> block_shift));
        const int64_t src = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                            static_cast<int64_t>(gpos & (block_size - 1)) * kv_dim + kv_head_off +
                            ch;
        if constexpr (QUANT) {
          stage_i4_row_chunk(&s_k4[r][ch >> 1], key_cache_i4, src);
        } else {
          *reinterpret_cast<uint4*>(&s_k[r][ch]) =
              *reinterpret_cast<const uint4*>(key_cache + src);
        }
      }
    }
    {
      const int chunk = tid % chunks;
      const int ch = chunk * (QUANT ? kInt4Vec : kLoadVec);
      for (int r = tid / chunks; r < tile_n; r += blockDim.x / chunks) {
        const int gpos = tile_pos + r;
        const int32_t pg = __ldg(table_row + (gpos >> block_shift));
        const int64_t src = layer_base + static_cast<int64_t>(pg) * (block_size * kv_dim) +
                            static_cast<int64_t>(gpos & (block_size - 1)) * kv_dim + kv_head_off +
                            ch;
        if constexpr (QUANT) {
          stage_i4_row_chunk(&s_v4[r][ch >> 1], value_cache_i4, src);
        } else {
          *reinterpret_cast<uint4*>(&s_v[r][ch]) =
              *reinterpret_cast<const uint4*>(value_cache + src);
        }
      }
    }
    __syncthreads();

    if (active) {
      const int p_end = min(tile_pos + kTileKv, limit);
      int p = tile_pos;
      // Complete groups of 4 positions: one rescale per group, exactly the
      // walk the decode kernel makes over the same prefix.
      for (; p + kGroup <= p_end; p += kGroup) {
        float kf[kGroup][kVec];
        float vf[kGroup][kVec];
        const int sp = p - tile_pos;
        // The whole 4-position step is either packed or residual: the group
        // boundary is a multiple of kGroup.
        const bool residual_k = QUANT && (p >= boundary);
        float ks = 1.f;
        float kz = 0.f;
        if constexpr (QUANT) {
          if (!residual_k) {
            const int64_t kidx = key_meta_layer_base +
                                 (static_cast<int64_t>(s_pg[sp]) * groups_per_block +
                                  ((p & (block_size - 1)) / group_size)) *
                                     channel_groups +
                                 lane_cg;
            ks = __ldg(key_scale + kidx);
            kz = __ldg(key_zero + kidx);
          }
        }
#pragma unroll
        for (int j = 0; j < kGroup; ++j) {
          // The K/V slice of a kv head is shared by all G q heads, so it is
          // read once per (position, lane) and reused across the head loop.
          if constexpr (QUANT) {
            if (residual_k) {
              const int rp = (p + j) % group_size;
              const int64_t roff = residual_layer_base +
                                   static_cast<int64_t>(rp) * kv_dim + kv_head_off + lane * kVec;
              load_bf16_chunk4(kf[j], key_residual, roff);
            } else {
              load_i4_tile_raw_chunk4(kf[j], s_k4[sp + j], lane * kVec);
            }
            load_i4_tile_raw_chunk4(vf[j], s_v4[sp + j], lane * kVec);
          } else {
            load_bf16_chunk4(kf[j], s_k[sp + j], lane * kVec);
            load_bf16_chunk4(vf[j], s_v[sp + j], lane * kVec);
          }
        }
#pragma unroll
        for (int i = 0; i < G; ++i) {
          if (i >= heads_here) break;
          float s[kGroup];
          const float sumq = q[i][0] + q[i][1] + q[i][2] + q[i][3];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            const float dot = q[i][0] * kf[j][0] + q[i][1] * kf[j][1] + q[i][2] * kf[j][2] +
                              q[i][3] * kf[j][3];
            // Raw int4 codes: the affine is per lane (one channel group), so
            // it lands here, before the cross-lane reduction. The zero term
            // multiplies sum(q), not sum(code).
            s[j] = residual_k ? dot : (dot - kz * sumq) * ks;
          }
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
#pragma unroll
            for (int off = 16; off; off >>= 1) {
              s[j] += __shfl_xor_sync(0xffffffffu, s[j], off);
            }
          }
          float m_new = m_i[i];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            m_new = fmaxf(m_new, s[j]);
          }
          const float alpha = __expf(m_i[i] - m_new);
          float l_new = l_i[i] * alpha;
          float pr[kGroup];
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            pr[j] = __expf(s[j] - m_new);
            l_new += pr[j];
          }
          // Fold V's affine into the weight (w = p*scale) and keep the
          // constant w*zero in corr_i; l stays the sum of the raw weights.
          float w[kGroup];
          float corr_add = 0.f;
#pragma unroll
          for (int j = 0; j < kGroup; ++j) {
            if constexpr (QUANT) {
              w[j] = pr[j] * s_vs[sp + j];
              corr_add += w[j] * s_vz[sp + j];
            } else {
              w[j] = pr[j];
            }
          }
#pragma unroll
          for (int c = 0; c < kVec; ++c) {
            float a = acc[i][c] * alpha;
#pragma unroll
            for (int j = 0; j < kGroup; ++j) {
              a += w[j] * vf[j][c];
            }
            acc[i][c] = a;
          }
          corr_i[i] = corr_i[i] * alpha + corr_add;
          m_i[i] = m_new;
          l_i[i] = l_new;
        }
      }
      // Fewer than 4 positions left in this row's prefix (same tail as the
      // decode kernel's).
      for (; p < p_end; ++p) {
        const int sp = p - tile_pos;
        float kf[kVec];
        float vf[kVec];
        float ksa = 1.f;
        float kza = 0.f;
        if constexpr (QUANT) {
          if (p >= boundary) {
            const int rp = p % group_size;
            const int64_t roff = residual_layer_base +
                                 static_cast<int64_t>(rp) * kv_dim + kv_head_off + lane * kVec;
            load_bf16_chunk4(kf, key_residual, roff);
          } else {
            const int64_t kidx = key_meta_layer_base +
                                 (static_cast<int64_t>(s_pg[sp]) * groups_per_block +
                                  ((p & (block_size - 1)) / group_size)) *
                                     channel_groups +
                                 lane_cg;
            ksa = __ldg(key_scale + kidx);
            kza = __ldg(key_zero + kidx);
            load_i4_tile_raw_chunk4(kf, s_k4[sp], lane * kVec);
          }
          load_i4_tile_raw_chunk4(vf, s_v4[sp], lane * kVec);
        } else {
          load_bf16_chunk4(kf, s_k[sp], lane * kVec);
          load_bf16_chunk4(vf, s_v[sp], lane * kVec);
        }
#pragma unroll
        for (int i = 0; i < G; ++i) {
          if (i >= heads_here) break;
          float dot = q[i][0] * kf[0] + q[i][1] * kf[1] + q[i][2] * kf[2] + q[i][3] * kf[3];
          const float sumq = q[i][0] + q[i][1] + q[i][2] + q[i][3];
          float s = (QUANT && p < boundary) ? (dot - kza * sumq) * ksa : dot;
#pragma unroll
          for (int off = 16; off; off >>= 1) {
            s += __shfl_xor_sync(0xffffffffu, s, off);
          }
          const float m_new = fmaxf(m_i[i], s);
          const float alpha = __expf(m_i[i] - m_new);
          const float pr = __expf(s - m_new);
          l_i[i] = l_i[i] * alpha + pr;
          float w = pr;
          float corr_add = 0.f;
          if constexpr (QUANT) {
            w = pr * s_vs[sp];
            corr_add = w * s_vz[sp];
          }
          corr_i[i] = corr_i[i] * alpha + corr_add;
          acc[i][0] = acc[i][0] * alpha + w * vf[0];
          acc[i][1] = acc[i][1] * alpha + w * vf[1];
          acc[i][2] = acc[i][2] * alpha + w * vf[2];
          acc[i][3] = acc[i][3] * alpha + w * vf[3];
          m_i[i] = m_new;
        }
      }
    }
    __syncthreads();
  }

  if (!active) return;
#pragma unroll
  for (int i = 0; i < G; ++i) {
    if (i >= heads_here) break;
    const float inv_l = 1.f / l_i[i];
    const int64_t out_base =
        static_cast<int64_t>(my_row) * dim + (head_begin + i) * head_size + lane * kVec;
    __nv_bfloat162 o[2];
    o[0] = __floats2bfloat162_rn((acc[i][0] - corr_i[i]) * inv_l,
                                 (acc[i][1] - corr_i[i]) * inv_l);
    o[1] = __floats2bfloat162_rn((acc[i][2] - corr_i[i]) * inv_l,
                                 (acc[i][3] - corr_i[i]) * inv_l);
    *reinterpret_cast<uint2*>(output + out_base) = *reinterpret_cast<const uint2*>(o);
  }
}

// Launch paged_prefill_kernel_bf16 for the q-heads-per-kv-head group G the
// caller measured (one instantiation per G in 1..8). QUANT picks the arm
// split: the bf16 callers reach only the QUANT false instantiations, which
// carry no int4 code.
template <bool QUANT>
static void launch_paged_prefill(int32_t heads_per_kv, const dim3& grid, const dim3& block,
                                 const int32_t* positions, const int32_t* seq_row_start,
                                 const int32_t* block_table, int32_t table_stride,
                                 int32_t num_blocks, int32_t block_size, int32_t block_shift,
                                 int32_t layer_idx, int32_t dim, const __nv_bfloat16* query,
                                 __nv_bfloat16* output, const __nv_bfloat16* key_cache,
                                 const __nv_bfloat16* value_cache, const int8_t* key_cache_i4,
                                 const int8_t* value_cache_i4, const float* key_scale,
                                 const float* key_zero, const float* value_scale,
                                 const float* value_zero, const __nv_bfloat16* key_residual,
                                 const int32_t* kv_slots, int32_t kv_dim, int32_t kv_head_num,
                                 int32_t head_num, int32_t head_size, int32_t residual_batch,
                                 int32_t group_size, int32_t channel_group, cudaStream_t stream) {
#define LAUNCH_PAGED_PREFILL(G)                                                             \
  paged_prefill_kernel_bf16<G, QUANT><<<grid, block, 0, stream>>>(                          \
      positions, seq_row_start, block_table, table_stride, num_blocks, block_size,          \
      block_shift, layer_idx, dim, query, output, key_cache, value_cache, key_cache_i4,      \
      value_cache_i4, key_scale, key_zero, value_scale, value_zero, key_residual, kv_slots,  \
      kv_dim, kv_head_num, head_num, head_size, residual_batch, group_size, channel_group)
  switch (heads_per_kv) {
    case 1: LAUNCH_PAGED_PREFILL(1); break;
    case 2: LAUNCH_PAGED_PREFILL(2); break;
    case 3: LAUNCH_PAGED_PREFILL(3); break;
    case 4: LAUNCH_PAGED_PREFILL(4); break;
    case 5: LAUNCH_PAGED_PREFILL(5); break;
    case 6: LAUNCH_PAGED_PREFILL(6); break;
    case 7: LAUNCH_PAGED_PREFILL(7); break;
    default: LAUNCH_PAGED_PREFILL(8); break;
  }
#undef LAUNCH_PAGED_PREFILL
  CUDA_KERNEL_CHECK();
}

// ========== Mixed decode + prefill dispatch (see paged_kernels.cuh) ==========
namespace {
// Zero-copy view of rows [row_offset, row_offset + rows) of a [batch, width]
// tensor (width == 0: a flat [batch] tensor). Rows are contiguous at a fixed
// stride, so the view needs no data movement.
tensor::Tensor row_range_view(const tensor::Tensor& src, int32_t row_offset, int32_t rows,
                              int32_t width) {
  const size_t elem = base::DataTypeSize(src.data_type());
  uint8_t* base = const_cast<uint8_t*>(src.ptr<uint8_t>());
  if (width > 0) {
    base += static_cast<int64_t>(row_offset) * width * elem;
    tensor::Tensor view(src.data_type(), std::vector<int32_t>{rows, width}, false, nullptr, base);
    view.set_device_type(src.device_type());
    return view;
  }
  base += static_cast<int64_t>(row_offset) * elem;
  tensor::Tensor view(src.data_type(), std::vector<int32_t>{rows}, false, nullptr, base);
  view.set_device_type(src.device_type());
  return view;
}

// Geometry paged_prefill_kernel_bf16 can serve: the kernel reproduces the
// split-KV decode kernel's arithmetic, so it inherits its head_size == 128
// (4 head dims per lane), and it needs a paged block table (table_stride > 0)
// and at least 4 positions per page (kGroup positions per rescale must not
// straddle pages). Its CTA maps one
// q row per warp and one kv head per block, with one kernel instantiation per
// q-heads-per-kv-head group, so an uneven GQA split is fine as long as the
// ceil ratio fits in the 1..8 range of instantiations.
static bool prefill_geometry_ok(int32_t head_num, int32_t kv_head_num, int32_t head_size,
                                int32_t table_stride, int32_t block_size) {
  if (head_num <= 0 || kv_head_num <= 0) return false;
  const int32_t heads_per_kv = (head_num + kv_head_num - 1) / kv_head_num;
  return head_size == 128 && table_stride > 0 && block_size >= 4 &&
         (block_size & (block_size - 1)) == 0 && heads_per_kv <= 8;
}
}  // namespace

// Prefill-row attention: rows [row_begin, batch) are prefill rows of
// num_prefill_seqs sequences, seq_row_start[s] .. seq_row_start[s+1) being the
// rows of sequence s. Each row is one prompt token with its own position, and
// every row attends to the full causal prefix [0, pos] — the KV of earlier
// chunks (and of a prefix-cache hit) is already in the cache by the time
// attention runs, so no chunk bookkeeping is needed here.
//
// This wrapper picks the kernel's G = q heads per kv head instantiation and
// sizes the grid. Callers must have checked prefill_geometry_ok() first:
// paged_attention_dispatch owns that decision because the only correct thing
// to do with an unservable geometry is to leave the prefill rows on the decode
// kernel, which needs the caller's partials buffer.
void paged_prefill_attention_cu_batch(int32_t row_begin, int32_t head_num, int32_t layer_idx,
                                      int32_t num_blocks, int32_t block_size, int32_t kv_dim,
                                      int32_t kv_head_num, int32_t head_size,
                                      const tensor::Tensor* seq_row_start,
                                      int32_t num_prefill_seqs, const tensor::Tensor& positions,
                                      const tensor::Tensor& block_table,
                                      const tensor::Tensor& query_batch,
                                      const tensor::Tensor& mha_out,
                                      const tensor::Tensor& key_cache,
                                      const tensor::Tensor& value_cache,
                                      const tensor::Tensor& key_scale,
                                      const tensor::Tensor& value_scale,
                                      const tensor::Tensor& key_zero,
                                      const tensor::Tensor& value_zero,
                                      const tensor::Tensor& key_residual,
                                      const tensor::Tensor& kv_slots, int32_t group_size,
                                      int32_t channel_group, CudaConfig* config) {
  const int32_t batch = positions.get_dim(0);
  const int32_t rows = batch - row_begin;
  if (rows <= 0 || num_prefill_seqs <= 0 || seq_row_start == nullptr) return;
  const bool quantized =
      check_kv_pools(key_cache, value_cache, key_scale, key_zero, value_scale, value_zero,
                     key_residual, "paged_prefill_attention_cu_batch");
  CHECK(!quantized || (!kv_slots.is_empty() &&
                       kv_slots.data_type() == base::DataType::kDataTypeInt32))
      << "paged_prefill_attention_cu_batch: S4 needs a non-empty int32 kv_slots tensor";
  CHECK(config != nullptr);
  const int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Rounded up: with an uneven GQA split (head_num not a multiple of
  // kv_head_num) the per-kv-head groups hold ceil or floor of the ratio.
  const int32_t heads_per_kv = (head_num + kv_head_num - 1) / kv_head_num;
  CHECK(prefill_geometry_ok(head_num, kv_head_num, head_size, table_stride, block_size))
      << "paged_prefill_kernel_bf16 cannot serve head_size=" << head_size
      << " table_stride=" << table_stride << " block_size=" << block_size
      << " kv_head_num=" << kv_head_num << "; the prefill rows must stay on the decode kernel.";

  const int32_t block_shift = block_shift_for(block_size);
  constexpr int kWarpsPerBlock = 8;  // one q row per warp, 8 rows per block
  // grid.x is bounded by the batch's total prefill rows, which is >= the
  // longest prefill sequence in the batch: blocks past a sequence's end exit
  // immediately (2 loads) instead of needing a host-side max_q_len.
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock, num_prefill_seqs, kv_head_num);
  const dim3 block(32 * kWarpsPerBlock);
  const int32_t* pos = positions.ptr<int32_t>();
  const int32_t* row_map = seq_row_start->ptr<int32_t>();
  const int32_t* table = block_table.ptr<int32_t>();
  const __nv_bfloat16* query = query_batch.ptr<__nv_bfloat16>();
  __nv_bfloat16* output = const_cast<__nv_bfloat16*>(mha_out.ptr<__nv_bfloat16>());
  if (quantized) {
    CHECK(group_size > 0 && block_size % group_size == 0 && group_size % 4 == 0)
        << "paged_prefill_attention_cu_batch: group_size=" << group_size
        << " must be a multiple of 4 dividing block_size=" << block_size;
    CHECK(channel_group > 0 && kv_dim % channel_group == 0 && channel_group % 4 == 0)
        << "paged_prefill_attention_cu_batch: channel_group=" << channel_group
        << " must be a multiple of 4 dividing kv_dim=" << kv_dim;
  }
  // Exactly one view per cache is meaningful; the unused set is nullptr and
  // the kernel selects on it (see paged_attention_cu_batch).
  const bool kq = quantized;
  const __nv_bfloat16* kcache = kq ? nullptr : key_cache.ptr<__nv_bfloat16>();
  const __nv_bfloat16* vcache = kq ? nullptr : value_cache.ptr<__nv_bfloat16>();
  const int8_t* kcache_i4 = kq ? key_cache.ptr<int8_t>() : nullptr;
  const int8_t* vcache_i4 = kq ? value_cache.ptr<int8_t>() : nullptr;
  const float* kscale = kq ? key_scale.ptr<float>() : nullptr;
  const float* kzero = kq ? key_zero.ptr<float>() : nullptr;
  const float* vscale = kq ? value_scale.ptr<float>() : nullptr;
  const float* vzero = kq ? value_zero.ptr<float>() : nullptr;
  const __nv_bfloat16* kresidual = kq ? key_residual.ptr<__nv_bfloat16>() : nullptr;
  const int32_t* slots = kq ? kv_slots.ptr<int32_t>() : nullptr;
  const int32_t residual_batch = kq ? key_residual.get_dim(1) : 0;
  cudaStream_t stream = config->stream;
  if (kq) {
    launch_paged_prefill<true>(heads_per_kv, grid, block, pos, row_map, table, table_stride,
                               num_blocks, block_size, block_shift, layer_idx, dim, query, output,
                               kcache, vcache, kcache_i4, vcache_i4, kscale, kzero, vscale, vzero,
                               kresidual, slots, kv_dim, kv_head_num, head_num, head_size,
                               residual_batch, group_size, channel_group, stream);
  } else {
    launch_paged_prefill<false>(heads_per_kv, grid, block, pos, row_map, table, table_stride,
                                num_blocks, block_size, block_shift, layer_idx, dim, query, output,
                                kcache, vcache, kcache_i4, vcache_i4, kscale, kzero, vscale, vzero,
                                kresidual, slots, kv_dim, kv_head_num, head_num, head_size,
                                residual_batch, group_size, channel_group, stream);
  }
}

void paged_attention_dispatch(int32_t head_num, int32_t layer_idx, int32_t num_blocks,
                              int32_t block_size, int32_t kv_dim, int32_t kv_head_num,
                              int32_t head_size, int32_t num_decode_rows,
                              const tensor::Tensor* seq_row_start, int32_t num_prefill_seqs,
                              const tensor::Tensor& positions, const tensor::Tensor& block_table,
                              const tensor::Tensor& query_batch, tensor::Tensor& score_batch,
                              const tensor::Tensor& mha_out, const tensor::Tensor& key_cache,
                              const tensor::Tensor& value_cache, const tensor::Tensor& key_scale,
                              const tensor::Tensor& value_scale, const tensor::Tensor& key_zero,
                              const tensor::Tensor& value_zero,
                              const tensor::Tensor& key_residual, const tensor::Tensor& kv_slots,
                              int32_t group_size, int32_t channel_group, CudaConfig* config) {
  const int32_t batch = query_batch.get_dim(0);
  CHECK(config != nullptr);
  const int32_t decode_rows = std::max(0, std::min(num_decode_rows, batch));
  const bool has_prefill = decode_rows < batch && seq_row_start != nullptr &&
                           num_prefill_seqs > 0;
  const int32_t dim = query_batch.get_dim(1);
  const int32_t table_stride = block_table.get_dim(1);
  // Only a geometry both kernels serve can be split by row type: the decode
  // rows go to the split-KV decode kernel and the prefill rows to the prefill
  // kernel, each writing its own row range of mha_out. Anything else
  // (head_size != 128, an unsupported page size, no prefill rows at all) keeps
  // the whole batch on paged_attention_cu_batch, whose split-KV decode kernel
  // serves decode and single-token prefill rows alike.
  if (!has_prefill ||
      !prefill_geometry_ok(head_num, kv_head_num, head_size, table_stride, block_size)) {
    paged_attention_cu_batch(head_num, layer_idx, num_blocks, block_size, kv_dim, kv_head_num,
                             head_size, positions, block_table, query_batch, score_batch, mha_out,
                             key_cache, value_cache, key_scale, value_scale, key_zero, value_zero,
                             key_residual, kv_slots, seq_row_start, num_decode_rows,
                             num_prefill_seqs, group_size, channel_group, config);
    return;
  }
  if (decode_rows > 0) {
    // The decode rows keep their batch rows [0, decode_rows), and both kernels
    // read the same cache, so a row's arithmetic does not depend on which side
    // of the split it landed. kv_slots is sliced with them: a decode row's
    // residual window is indexed by the sequence slot, which is a row in this
    // tensor just like positions/block_table.
    // kv_slots is empty for a bf16 cache; slicing it would build a rank-0
    // temporary and fail. Only the quantized arm has a slot table to slice.
    const tensor::Tensor slots_slice =
        kv_slots.is_empty() ? kv_slots : row_range_view(kv_slots, 0, decode_rows, 0);
    paged_attention_cu_batch(head_num, layer_idx, num_blocks, block_size, kv_dim, kv_head_num,
                             head_size, row_range_view(positions, 0, decode_rows, 0),
                             row_range_view(block_table, 0, decode_rows, table_stride),
                             row_range_view(query_batch, 0, decode_rows, dim), score_batch,
                             row_range_view(mha_out, 0, decode_rows, dim), key_cache, value_cache,
                             key_scale, value_scale, key_zero, value_zero, key_residual,
                             slots_slice, nullptr, decode_rows, 0, group_size, channel_group,
                             config);
  }
  paged_prefill_attention_cu_batch(decode_rows, head_num, layer_idx, num_blocks, block_size,
                                   kv_dim, kv_head_num, head_size, seq_row_start, num_prefill_seqs,
                                   positions, block_table, query_batch, mha_out, key_cache,
                                   value_cache, key_scale, value_scale, key_zero, value_zero,
                                   key_residual, kv_slots, group_size, channel_group, config);
}

// ========== Fused-QKV de-interleave ========================================
// Split the [batch, dim + 2 * kv_dim] fused QKV projection output into its
// contiguous q / k / v buffers (see model::split_fused_qkv_output, which keeps
// the batch == 1 zero-copy view and calls this for every multi-token batch).
// Replaces three cudaMemcpy2DAsync calls: a 2D copy is one strided walk per
// request, so a 512-row chunk pays three of them (plus their setup) for a
// layout that is a single flat pass per row here. One block per row, 16 bytes
// per thread step.
__global__ void split_fused_qkv_kernel_bf16(const uint4* __restrict__ src,
                                            uint4* __restrict__ dst_q, uint4* __restrict__ dst_k,
                                            uint4* __restrict__ dst_v, int32_t q_vec,
                                            int32_t kv_vec, int32_t row_vec) {
  const uint4* srow = src + static_cast<int64_t>(blockIdx.x) * row_vec;
  uint4* dq = dst_q + static_cast<int64_t>(blockIdx.x) * q_vec;
  uint4* dk = dst_k + static_cast<int64_t>(blockIdx.x) * kv_vec;
  uint4* dv = dst_v + static_cast<int64_t>(blockIdx.x) * kv_vec;
  // The three segments share one walk: every thread reads consecutive source
  // words (fully coalesced on the load side) and writes them to whichever
  // output buffer owns them. dim and kv_dim are multiples of 8 bf16 values
  // (one uint4) in every geometry this path is used for.
  for (int32_t i = threadIdx.x; i < row_vec; i += blockDim.x) {
    const uint4 v = srow[i];
    if (i < q_vec) {
      dq[i] = v;
    } else if (i < q_vec + kv_vec) {
      dk[i - q_vec] = v;
    } else {
      dv[i - q_vec - kv_vec] = v;
    }
  }
}

void split_fused_qkv_bf16_cu(const void* fused_src, void* dst_q, void* dst_k, void* dst_v,
                             int32_t batch, int32_t dim, int32_t kv_dim, cudaStream_t stream) {
  CHECK(batch > 0 && dim > 0 && kv_dim > 0);
  CHECK_EQ(dim % 8, 0);
  CHECK_EQ(kv_dim % 8, 0);
  const int32_t q_vec = dim / 8;
  const int32_t kv_vec = kv_dim / 8;
  const int32_t row_vec = q_vec + 2 * kv_vec;
  split_fused_qkv_kernel_bf16<<<batch, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(fused_src), reinterpret_cast<uint4*>(dst_q),
      reinterpret_cast<uint4*>(dst_k), reinterpret_cast<uint4*>(dst_v), q_vec, kv_vec, row_vec);
  CUDA_KERNEL_CHECK();
}

}  // namespace kernel
