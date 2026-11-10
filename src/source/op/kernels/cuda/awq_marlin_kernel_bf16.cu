// Marlin-style AWQ W4A16 GEMM (Ampere+).
//
// The checkpoint's qweight [K, N/8] packs eight output columns per int32 in
// AWQ's interleaved nibble order. Loading repacks each output column's 16-bit
// k slice into one int32 pair (a uint2 = one 16x8x16 B fragment). The packed B
// tile is streamed into shared memory with cp.async, one k unit ahead of the
// MMA loop, and dequantized in registers straight out of shared. Initially the
// B fragment was loaded directly from global inside the step loop; that left
// only two 8-byte loads in flight per warp and ran the GEMM at ~200 GB/s
// regardless of batch size (latency-bound). Staging the tile moves the weight
// stream onto the async copy engine and is worth ~1.8x.
//
// Tiling: 256 threads, block 64x128 (or 32/16 rows for small batches) x 128 k.
// Each warp owns 16 output columns (two 8-column MMA fragments) and all of the
// block's 16-row m tiles, so one dequantized B fragment feeds MT MMAs. The
// scale/zero-point handling is specialized at compile time on whether one
// group spans the whole k block, which keeps a runtime division and two
// metadata loads out of the hot loop for the common group_size == 128.
//
// Split-K is chosen to fill the SMs: the un-split o/down/qkv grids are only
// 40-56 blocks at decode batch sizes, well short of the 4-blocks/SM residency
// limit, so the kernel multiplies the block count with split-K and accumulates
// the partials in a workspace slab.
#include <base/cuda_check.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "../kernels_interface.h"
#include "awq_marlin_kernel.cuh"

namespace kernel {
namespace {
using bf16 = __nv_bfloat16;

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBlockK = kAwqMarlinBlockK;
constexpr int kMmaSteps = kBlockK / 16;
constexpr int kAPad = kBlockK + 8;
constexpr int kPackThreads = 256;
constexpr int kMaxSplit = 8;  // past this the partial-sum traffic outweighs the fill

__host__ __device__ __forceinline__ int awq_unpack(uint32_t packed, int i) {
  return static_cast<int>((packed >> (4 * (i >> 1) + 16 * (i & 1))) & 0xFu);
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int kPending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t dequant_pair(uint32_t word, int shift, float scale,
                                                 float neg_zero_scale) {
  union {
    __nv_bfloat162 v;
    uint32_t u;
  } p;
  p.v = __floats2bfloat162_rn(fmaf(static_cast<float>((word >> shift) & 0xFu), scale,
                                   neg_zero_scale),
                              fmaf(static_cast<float>((word >> (shift + 4)) & 0xFu), scale,
                                   neg_zero_scale));
  return p.u;
}

// One block-load of the checkpoint qweight: 16 k rows x 32 packed columns,
// which produce 16 k x 256 output columns of the Marlin layout.
__global__ void awq_marlin_pack_kernel(const int32_t* __restrict__ qweight,
                                       uint2* __restrict__ packed, int packed_cols,
                                       int total_out) {
  __shared__ int32_t tile[16][32];
  const int c0 = blockIdx.x * 32;
  const int k16 = blockIdx.y;
  const int tid = threadIdx.x;

#pragma unroll
  for (int r = 0; r < 512 / kPackThreads; ++r) {
    const int idx = tid + r * kPackThreads;
    const int i = idx >> 5;
    const int c = idx & 31;
    const int cc = c0 + c;
    tile[i][c] = cc < packed_cols
                     ? qweight[static_cast<int64_t>(k16 * 16 + i) * packed_cols + cc]
                     : 0;
  }
  __syncthreads();

  const int c_rel = tid >> 3;
  const int i = tid & 7;                 // output column inside the packed group
  const int nib = (i >> 1) + 4 * (i & 1);  // AWQ nibble position for that column
  const int shift = 4 * nib;
  if (c0 + c_rel >= packed_cols) {
    return;
  }
  uint32_t lo = 0;
  uint32_t hi = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    lo |= static_cast<uint32_t>((tile[k][c_rel] >> shift) & 0xF) << (4 * k);
    hi |= static_cast<uint32_t>((tile[k + 8][c_rel] >> shift) & 0xF) << (4 * k);
  }
  const int64_t n = static_cast<int64_t>(k16) * total_out + (c0 + c_rel) * 8 + i;
  packed[n] = make_uint2(lo, hi);
}

template <int BLOCK_M, int STAGES, int BLOCK_N>
__global__ __launch_bounds__(kThreads, 4) void awq_marlin_gemm_kernel(const bf16* __restrict__ input,
                                       const uint2* __restrict__ qweight,
                                       const int32_t* __restrict__ qzeros,
                                       const bf16* __restrict__ scales, bf16* __restrict__ output,
                                       bf16* __restrict__ partial, int M, int K, int N_total,
                                       int N_layer, int out_offset, int packed_total,
                                       int group_size, bool zero_point, int k_units_per_split) {
  constexpr int MT = BLOCK_M / 16;
  constexpr int kColsPerWarp = BLOCK_N / kWarps;
  constexpr int kFrags = kColsPerWarp / 8;
  static_assert(BLOCK_N % (kWarps * 8) == 0, "block N must split into 8-column fragments");
  constexpr int kAVecs = BLOCK_M * (kBlockK / 8);

  __align__(16) __shared__ bf16 sA[STAGES][BLOCK_M * kAPad];
  // Packed int4 B tiles, staged asynchronously one k unit ahead. Keeping the
  // weights in shared memory turns the GEMM's weight stream from a chain of
  // dependent 8-byte global loads (one per MMA step, ~200 GB/s) into a deep
  // cp.async pipeline; the inner loop then dequantizes straight out of shared.
  __align__(16) __shared__ uint2 sB[STAGES][kMmaSteps * BLOCK_N];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = lane >> 2;
  const int tq = lane & 3;
  const int lmat = lane >> 3;
  const int a_row_off = ((lmat & 1) << 3) + (lane & 7);
  const int a_col_off = (lmat >> 1) << 3;
  const int sh = tq * 8;

  const int m0 = blockIdx.y * BLOCK_M;
  const int n0 = blockIdx.x * BLOCK_N;
  const int col_base = n0 + warp * kColsPerWarp;
  int ns[kFrags];
#pragma unroll
  for (int f = 0; f < kFrags; ++f) {
    const int bc = out_offset + col_base + f * 8 + group;
    ns[f] = bc < N_total ? bc : N_total - 1;
  }

  float acc[MT][kFrags][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[mt][f][e] = 0.f;
      }
    }
  }

  const int z = static_cast<int>(blockIdx.z);
  const int k_begin = z * k_units_per_split * kBlockK;
  const int k_units = min(k_units_per_split, K / kBlockK - z * k_units_per_split);

  auto copy_input = [&](int stage, int k0) {
    bf16* dst = sA[stage];
    for (int idx = tid; idx < kAVecs; idx += kThreads) {
      const int row = idx >> 4;
      const int seg = idx & 15;
      const int m = m0 + row;
      if (m < M) {
        cp_async16(&dst[row * kAPad + seg * 8],
                   input + static_cast<int64_t>(m) * K + k0 + seg * 8);
      } else {
        *reinterpret_cast<uint4*>(&dst[row * kAPad + seg * 8]) = make_uint4(0, 0, 0, 0);
      }
    }
  };

  // One 16B cp.async per two packed output columns. Each staged row is the
  // block's 128-column slice of one k16 group in the Marlin layout.
  constexpr int kBChunks = kMmaSteps * (BLOCK_N / 2);
  auto copy_weights = [&](int stage, int k0) {
    uint2* dstb = sB[stage];
    const int64_t krow = static_cast<int64_t>(k0 / 16) * N_total;
    for (int v = tid; v < kBChunks; v += kThreads) {
      const int step = v / (BLOCK_N / 2);
      const int c2 = (v % (BLOCK_N / 2)) * 2;
      const int ncol = n0 + c2;
      uint2* dst = &dstb[step * BLOCK_N + c2];
      if (ncol + 2 <= N_total) {
        cp_async16(dst, qweight + krow + static_cast<int64_t>(step) * N_total + ncol);
      } else {
        for (int j = 0; j < 2; ++j) {
          const int c = ncol + j;
          dst[j] = qweight[krow + static_cast<int64_t>(step) * N_total +
                           (c < N_total ? c : N_total - 1)];
        }
      }
    }
  };

  if (k_units > 0) {
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < k_units) {
        copy_input(s, k_begin + s * kBlockK);
        copy_weights(s, k_begin + s * kBlockK);
      }
      cp_commit();
    }
    for (int i = 0; i < k_units; ++i) {
      if (STAGES > 2) {
        cp_wait<STAGES - 2>();
      } else {
        cp_wait<0>();
      }
      __syncthreads();
      const int pf = i + STAGES - 1;
      if (pf < k_units) {
        copy_input(pf % STAGES, k_begin + pf * kBlockK);
        copy_weights(pf % STAGES, k_begin + pf * kBlockK);
      }
      cp_commit();

      const bf16* sAb = sA[i % STAGES];
      const uint2* sBb = sB[i % STAGES];
      const int k0 = k_begin + i * kBlockK;
      // A group that spans the whole k block lets the scale/zero loads move
      // out of the unrolled MMA steps.
      const bool one_group = group_size % kBlockK == 0;
      float hscale[kFrags];
      int hzero[kFrags];
      if (one_group) {
        const int gb = k0 / group_size;
#pragma unroll
        for (int f = 0; f < kFrags; ++f) {
          hscale[f] = __bfloat162float(scales[static_cast<int64_t>(gb) * N_total + ns[f]]);
          hzero[f] = zero_point
                         ? awq_unpack(static_cast<uint32_t>(
                                          qzeros[static_cast<int64_t>(gb) * packed_total +
                                                 (ns[f] >> 3)]),
                                      ns[f] & 7)
                         : 8;
        }
      }
      // Specialize on whether one scale/zero group spans the whole k block.
      // The generic branch needs a runtime division and a scale/zero load per
      // MMA step; folding `one_group` into a compile-time flag keeps all of
      // that out of the common (group_size == kBlockK) hot loop instead of
      // merely predicating it off.
      auto run_steps = [&](auto one_group_tag) {
        constexpr bool ONE_GROUP = decltype(one_group_tag)::value;
        const int gb = k0 / group_size;
#pragma unroll
        for (int step = 0; step < kMmaSteps; ++step) {
          const int kk = k0 + step * 16;
          const int g = ONE_GROUP ? gb : (kk / group_size);
          uint32_t bf[kFrags][2];
#pragma unroll
          for (int f = 0; f < kFrags; ++f) {
            const uint2 w = sBb[step * BLOCK_N + warp * kColsPerWarp + f * 8 + group];
            const float sf =
                ONE_GROUP
                    ? hscale[f]
                    : __bfloat162float(scales[static_cast<int64_t>(g) * N_total + ns[f]]);
            const int zf =
                ONE_GROUP
                    ? hzero[f]
                    : (zero_point ? awq_unpack(static_cast<uint32_t>(
                                                    qzeros[static_cast<int64_t>(g) *
                                                               packed_total +
                                                           (ns[f] >> 3)]),
                                               ns[f] & 7)
                                  : 8);
            bf[f][0] = dequant_pair(w.x, sh, sf, -zf * sf);
            bf[f][1] = dequant_pair(w.y, sh, sf, -zf * sf);
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            uint32_t a0, a1, a2, a3;
            ldmatrix_x4(smem_addr(&sAb[(mt * 16 + a_row_off) * kAPad + step * 16 + a_col_off]),
                        a0, a1, a2, a3);
#pragma unroll
            for (int f = 0; f < kFrags; ++f) {
              mma_m16n8k16(acc[mt][f][0], acc[mt][f][1], acc[mt][f][2], acc[mt][f][3], a0, a1,
                           a2, a3, bf[f][0], bf[f][1]);
            }
          }
        }
      };
      if (one_group) {
        run_steps(std::integral_constant<bool, true>{});
      } else {
        run_steps(std::integral_constant<bool, false>{});
      }
    }
  }

  const bool split = gridDim.z > 1;
  bf16* part = split ? partial + static_cast<int64_t>(blockIdx.z) * M * N_layer : nullptr;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      const int col = col_base + f * 8 + tq * 2;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = m0 + mt * 16 + group + half * 8;
        if (row >= M) {
          continue;
        }
        const float v0 = acc[mt][f][half * 2];
        const float v1 = acc[mt][f][half * 2 + 1];
        const int64_t o = static_cast<int64_t>(row) * N_layer + col;
        if (split) {
          if (col + 1 < N_layer) {
            *reinterpret_cast<__nv_bfloat162*>(part + o) = __floats2bfloat162_rn(v0, v1);
          } else if (col < N_layer) {
            part[o] = __float2bfloat16(v0);
          }
        } else if (col + 1 < N_layer) {
          *reinterpret_cast<__nv_bfloat162*>(output + o) = __floats2bfloat162_rn(v0, v1);
        } else if (col < N_layer) {
          output[o] = __float2bfloat16(v0);
        }
      }
    }
  }
}

template <int BM, int STAGES, int BN, typename... A>
void launch_marlin(const dim3& grid, cudaStream_t stream, A... args) {
  awq_marlin_gemm_kernel<BM, STAGES, BN><<<grid, kThreads, 0, stream>>>(args...);
}

__global__ void awq_marlin_reduce_kernel(const bf16* __restrict__ partial,
                                         bf16* __restrict__ output, int total, int split,
                                         int64_t stride) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float acc = __bfloat162float(partial[i]);
    for (int j = 1; j < split; ++j) {
      acc += __bfloat162float(partial[static_cast<int64_t>(j) * stride + i]);
    }
    output[i] = __float2bfloat16(acc);
  }
}

int device_sm_count() {
  static int sm_count = -1;
  if (sm_count < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  }
  return sm_count;
}

// Split-K partial sums. A fixed slab keeps the path free of allocations once
// the first decode step has warmed the kernel up (CUDA graph capture would
// otherwise see a cudaMalloc).
constexpr int64_t kPartialElems = 16 << 20;  // 32 MB of bf16 partial sums
bf16* partial_workspace() {
  static bf16* ptr = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    if (cudaMalloc(&ptr, kPartialElems * sizeof(bf16)) != cudaSuccess) {
      ptr = nullptr;
    }
  }
  return ptr;
}
}  // namespace

bool awq_marlin_supported(int32_t K, int32_t N, int32_t out_offset, int32_t group_size) {
  return K > 0 && K % kBlockK == 0 && N > 0 && N % 8 == 0 && out_offset % 8 == 0 &&
         group_size > 0 && group_size % 16 == 0;
}

tensor::Tensor awq_marlin_pack(const tensor::Tensor& qweight, int32_t K, int32_t N_total,
                               cudaStream_t stream) {
  if (!awq_marlin_supported(K, N_total, 0, 16) || N_total % 8 != 0) {
    return tensor::Tensor();
  }
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  // Flat 2-D view [K/16, 2*N_total] so the layer's tensor checks stay unchanged.
  tensor::Tensor packed(base::DataType::kDataTypeInt32, K / 16, N_total * 2, true, alloc);
  const int packed_cols = N_total / 8;
  const dim3 grid((packed_cols + 31) / 32, K / 16);
  awq_marlin_pack_kernel<<<grid, kPackThreads, 0, stream>>>(
      qweight.ptr<int32_t>(), reinterpret_cast<uint2*>(packed.ptr<int32_t>()), packed_cols,
      N_total);
  return packed;
}

void awq_marlin_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                          const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                          const tensor::Tensor& output, const AwqMatmulParams& params,
                          const CudaConfig* config) {
  CHECK_EQ(qweight.dims_size(), 2);
  CHECK_EQ(qweight.get_dim(1) % 2, 0);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  const int K = qweight.get_dim(0) * 16;
  const int N_total = qweight.get_dim(1) / 2;
  const int N = params.out_features;
  const int out_offset = params.out_offset;
  const int group_size = params.group_size;
  const bool zero_point = params.zero_point && !qzeros.is_empty();
  CHECK_EQ(scales.get_dim(1), N_total);
  CHECK_EQ(K % kBlockK, 0);
  CHECK_EQ(input.size() % K, 0);
  const int M = static_cast<int>(input.size() / K);
  const int packed_total = N_total / 8;
  if (zero_point) {
    CHECK_EQ(qzeros.get_dim(1), packed_total);
  }

  cudaStream_t stream = config ? config->stream : nullptr;
  const bf16* input_ptr = input.ptr<bf16>();
  const uint2* qweight_ptr = reinterpret_cast<const uint2*>(qweight.ptr<int32_t>());
  const int32_t* qzeros_ptr = zero_point ? qzeros.ptr<int32_t>() : nullptr;
  const bf16* scales_ptr = scales.ptr<bf16>();
  bf16* output_ptr = const_cast<bf16*>(output.ptr<bf16>());

  const int k_units = K / kBlockK;
  bf16* partial = partial_workspace();

  auto launch = [&](auto bm, auto bn) {
    using Bm = decltype(bm);
    using Bn = decltype(bn);
    constexpr int BM = Bm::value;
    constexpr int BN = Bn::value;
    constexpr int kMTPerWarp = BM / 16;
    constexpr int kFrags = BN / (kWarps * 8);
    if (kMTPerWarp * kFrags > 8) {
      return;  // keep the accumulator file under 32 registers
    }
    const int gx = (N + BN - 1) / BN;
    const int base_gy = (M + BM - 1) / BM;
    // Fill the GPU with blocks: the un-split grid is often only 40-56 blocks
    // for the o/down/qkv projections at decode batch sizes, far below the
    // 4-blocks/SM x 108-SM residency. Split-K multiplicatively raises the block
    // count; past the fill point it only adds partial-sum traffic, so cap it.
    const int base_blocks = gx * base_gy;
    int split = (8 * device_sm_count() + base_blocks - 1) / base_blocks;
    split = std::min(split, kMaxSplit);
    split = std::min(split, std::max(1, k_units));
    split = std::max(1, split);
    const int64_t plane = static_cast<int64_t>(M) * N;
    if (partial == nullptr || plane > kPartialElems) {
      split = 1;
    } else {
      split = std::min<int64_t>(split, kPartialElems / plane);
    }
    split = std::max(1, split);
    const int k_units_per_split = (k_units + split - 1) / split;
    const dim3 grid(gx, base_gy, split);
    // Shared memory budget: the staged B tile adds 8 KB per stage for
    // BLOCK_N=128. 16-row blocks keep three stages (37 KB); 32-row blocks drop
    // to two (34 KB) so four blocks still fit per SM; 64-row blocks use the
    // narrower 64-column tile (43 KB) to stay under the 48 KB static limit.
    launch_marlin<BM, (BM == 16 ? 3 : 2), BN>(grid, stream, input_ptr, qweight_ptr, qzeros_ptr,
                                              scales_ptr, output_ptr, partial, M, K, N_total, N,
                                              out_offset, packed_total, group_size, zero_point,
                                              k_units_per_split);
    if (split > 1) {
      const int total = M * N;
      const int blocks = std::min((total + 255) / 256, 65535);
      awq_marlin_reduce_kernel<<<blocks, 256, 0, stream>>>(
          partial, output_ptr, total, split, static_cast<int64_t>(M) * N);
    }
  };
  if (M <= 16) {
    launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
  } else if (M <= 32) {
    launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{});
  } else {
    launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
  }
  CUDA_KERNEL_CHECK();
}
}  // namespace kernel
