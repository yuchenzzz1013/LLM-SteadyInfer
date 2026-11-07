// CUDA AWQ W4A16 GEMM.
//
// Matmul semantics: weight [N, K] row-major (stored as AWQ packed int4) x
// input [batch, K] -> [batch, N].
//
// The checkpoint keeps the weights packed (`qweight` int32, eight 4-bit codes
// per word); nothing is dequantized at load time. The GEMM dequantizes a
// [64 outputs, K chunk] weight tile into shared memory per K chunk and
// feeds it to BF16 Tensor Core MMAs with FP32 accumulation, so the resident
// weight footprint stays at the checkpoint's int4 size while the dequantized
// tile is reused by every batch row in the block.
//
// Block layout: 64 output rows (8 packed int32 columns) x up to 64 batch rows
// (4 warps). Eight consecutive packed columns make one warp's qweight loads
// cover four contiguous 32B segments — the AWQ "gemm" layout packs 8 output
// rows per int32, so this is the natural coalescing unit. A decode step only
// activates as many 16-row batch tiles as it can fill (1/2/4 warps); the other
// warps keep helping with the dequantization, which is shared by the block.
//
// The K chunk is 64 (four 16-wide MMA steps) rather than 16: a decode step
// is latency-bound on the weight stream, so one shared-memory fill (two
// barriers) per 64 inputs keeps the 14B-parameter stream moving instead of
// stalling every 16. The dequantized B tile is row-major [k][n] with a padded
// leading dimension so the eight bf16 values of one packed word go out as one
// 16B shared store instead of eight bank-conflicting scalar stores.
#include <base/cuda_check.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include "../kernels_interface.h"
#include "awq_matmul_kernel.cuh"

namespace kernel {
namespace {
using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

// AWQ's "gemm" packing stores eight 4-bit codes per int32; output column i of
// the packed group sits at bit 4 * ((i >> 1) + 4 * (i & 1)). Zero points use
// the same interleaving.
__host__ __device__ __forceinline__ int awq_unpack(uint32_t packed, int i) {
  return static_cast<int>((packed >> (4 * (i >> 1) + 16 * (i & 1))) & 0xFu);
}

constexpr int kWarps = 4;                 // one warp per 16-row batch tile
constexpr int kBatchPerBlock = kWarps * 16;
constexpr int kOutPerBlock = 64;          // 8 packed int32 columns per block
constexpr int kOutGroups = kOutPerBlock / 16;
constexpr int kKChunk = 64;               // 4 MMA k-steps per shared-memory fill
constexpr int kMmaSteps = kKChunk / 16;
// B is row-major [k][n] with a 24-element leading dimension. The 8 padding
// halves make the dequantizer's 16B stores land on distinct shared banks (a
// 16-wide leading dimension maps all eight packed columns onto one bank).
constexpr int kBLd = 24;

__global__ void awq_matmul_bf16_wmma(const bf16* __restrict__ input,
                                     const int32_t* __restrict__ qweight,
                                     const int32_t* __restrict__ qzeros,
                                     const bf16* __restrict__ scales, bf16* __restrict__ output,
                                     int K, int N, int packed_total, int total_out,
                                     int out_offset, int group_size, int batch,
                                     int warps_m) {
  const int n0 = blockIdx.x * kOutPerBlock;
  const int m0 = blockIdx.y * kBatchPerBlock;
  const int c_base = out_offset / 8 + blockIdx.x * (kOutPerBlock / 8);
  // Split-K: blockIdx.z selects one K slice; each slice accumulates into the
  // (pre-zeroed) output with atomics. Decode GEMMs produce far fewer blocks
  // than the GPU has SMs otherwise, and the int4 weight stream is
  // latency-bound, so more independent blocks in flight is what buys
  // bandwidth here. Slice boundaries are chunk aligned so every shared-memory
  // chunk sits in exactly one AWQ group (see the scale hoisting below).
  const int chunk_count = (K + kKChunk - 1) / kKChunk;
  const int chunks_per_split = (chunk_count + gridDim.z - 1) / gridDim.z;
  const int k_begin = blockIdx.z * chunks_per_split * kKChunk;
  const int k_end = min(k_begin + chunks_per_split * kKChunk, K);

  __align__(16) __shared__ bf16 sB[kOutGroups][kKChunk][kBLd];  // sB[group][k][n]
  __shared__ bf16 sA[kWarps][16][kKChunk];      // sA[warp][m][k], row-major A tiles
  __shared__ float sC[kWarps][16][16];           // guarded store staging, one tile per warp

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kOutGroups];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag;
#pragma unroll
  for (int g = 0; g < kOutGroups; ++g) {
    wmma::fill_fragment(acc[g], 0.0f);
  }

  static_assert(kWarps == kOutGroups, "warp <-> output-group mapping assumes 4 warps");
  const int g_out = warp;

  // The host only selects this kernel when the whole kKChunk-wide chunk sits
  // in one AWQ group, so the zero point and the eight scales of a packed
  // column are loaded once per chunk instead of once per weight word.
  const int c_local = tid & 7;  // fixed for this thread across the chunk
  const int n_local = c_local * 8;

  for (int k0 = k_begin; k0 < k_end; k0 += kKChunk) {
    constexpr int kWords = kKChunk * (kOutPerBlock / 8) / (kWarps * 32);
    const int cc = (c_base + c_local) < packed_total ? (c_base + c_local) : packed_total - 1;
    int32_t packed[kWords];
#pragma unroll
    for (int r = 0; r < kWords; ++r) {
      const int k = k0 + (tid >> 3) + r * 16;
      const int kc = k < K ? k : K - 1;
      packed[r] = qweight[kc * packed_total + cc];
    }

    // The chunk's group: uniform_group guarantees every k in [k0, k0+kKChunk) has
    // the same group index (the split points are kChunk aligned).
    const int g = k0 / group_size;
    uint32_t zpacked = 0u;
    if (qzeros != nullptr) {
      zpacked = static_cast<uint32_t>(qzeros[g * packed_total + cc]);
    }
    const bf16* scale_row = scales + static_cast<int64_t>(g) * total_out + out_offset + n0;
    bf16 scale[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      scale[i] = __float2bfloat16(0.f);
    }
    if (n0 + n_local < N) {
      if (n0 + n_local + 8 <= N) {
        // Eight bf16 scales = one 16B vector load (alignment is guaranteed:
        // total_out / out_offset / n0 are multiples of 8).
        const uint4 v = *reinterpret_cast<const uint4*>(scale_row + n_local);
        scale[0] = __ushort_as_bfloat16(static_cast<uint16_t>(v.x & 0xffffu));
        scale[1] = __ushort_as_bfloat16(static_cast<uint16_t>(v.x >> 16));
        scale[2] = __ushort_as_bfloat16(static_cast<uint16_t>(v.y & 0xffffu));
        scale[3] = __ushort_as_bfloat16(static_cast<uint16_t>(v.y >> 16));
        scale[4] = __ushort_as_bfloat16(static_cast<uint16_t>(v.z & 0xffffu));
        scale[5] = __ushort_as_bfloat16(static_cast<uint16_t>(v.z >> 16));
        scale[6] = __ushort_as_bfloat16(static_cast<uint16_t>(v.w & 0xffffu));
        scale[7] = __ushort_as_bfloat16(static_cast<uint16_t>(v.w >> 16));
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (n0 + n_local + i < N) {
            scale[i] = scale_row[n_local + i];
          }
        }
      }
    }

#pragma unroll
    for (int r = 0; r < kWords; ++r) {
      const int kk = (tid >> 3) + r * 16;  // k position inside the chunk
      const uint32_t word = static_cast<uint32_t>(packed[r]);
      float f[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int q = awq_unpack(word, i);
        const int zero = qzeros != nullptr ? awq_unpack(zpacked, i) : 8;
        f[i] = static_cast<float>(q - zero) * __bfloat162float(scale[i]);
      }
      union Pack {
        __nv_bfloat162 v;
        uint32_t u;
      };
      Pack p0, p1, p2, p3;
      p0.v = __floats2bfloat162_rn(f[0], f[1]);
      p1.v = __floats2bfloat162_rn(f[2], f[3]);
      p2.v = __floats2bfloat162_rn(f[4], f[5]);
      p3.v = __floats2bfloat162_rn(f[6], f[7]);
      // One 16B store per packed word (eight output rows), conflict free.
      *reinterpret_cast<uint4*>(&sB[c_local >> 1][kk][(c_local & 1) * 8]) =
          make_uint4(p0.u, p1.u, p2.u, p3.u);
    }

    // Stage the input chunk for the batch tiles this step actually has; rows
    // past the batch are zero padded.
    {
      const int rows = warps_m * 16;
      for (int idx = tid; idx < rows * kKChunk; idx += blockDim.x) {
        const int mm = idx / kKChunk;  // batch row inside the block
        const int kk = idx % kKChunk;  // k position inside the chunk
        const int m = m0 + mm;
        const int k = k0 + kk;
        sA[mm >> 4][mm & 15][kk] =
            (m < batch && k < k_end) ? input[static_cast<int64_t>(m) * K + k]
                                     : __float2bfloat16(0.f);
      }
    }
    __syncthreads();

    // 每个 warp 负责一个 16 列输出 group,遍历本步所有 batch tile。
#pragma unroll
    for (int step = 0; step < kMmaSteps; ++step) {
      wmma::load_matrix_sync(b_frag, &sB[g_out][step * 16][0], kBLd);
#pragma unroll
      for (int mt = 0; mt < kWarps; ++mt) {
        if (mt >= warps_m) continue;
        wmma::load_matrix_sync(a_frag, &sA[mt][0][step * 16], kKChunk);
        wmma::mma_sync(acc[mt], a_frag, b_frag, acc[mt]);
      }
    }
    __syncthreads();
  }

  const int col_base = n0 + g_out * 16;
#pragma unroll
  for (int mt = 0; mt < kWarps; ++mt) {
    if (mt >= warps_m) continue;
    const int row_base = m0 + mt * 16;
    __syncwarp();
    wmma::store_matrix_sync(&sC[warp][0][0], acc[mt], 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
      const int mm = idx >> 4;
      const int nn = idx & 15;
      const int m = row_base + mm;
      const int col = col_base + nn;
      if (m < batch && col < N) {
        bf16* dst = output + static_cast<int64_t>(m) * N + col;
        const bf16 value = __float2bfloat16(sC[warp][mm][nn]);
        if (gridDim.z == 1) {
          *dst = value;
        } else {
          atomicAdd(dst, value);
        }
      }
    }
  }
}

// Fallback for pre-Ampere parts: one thread per (batch, output) pair,
// dequantizing inside the dot product.
__global__ void awq_matmul_bf16_scalar(const bf16* __restrict__ input,
                                       const int32_t* __restrict__ qweight,
                                       const int32_t* __restrict__ qzeros,
                                       const bf16* __restrict__ scales, bf16* __restrict__ output,
                                       int K, int N, int packed_total, int total_out,
                                       int out_offset, int group_size, int batch) {
  const int64_t total = static_cast<int64_t>(batch) * N;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx / N);
    const int n = static_cast<int>(idx % N);
    const int c = out_offset / 8 + (n >> 3);
    const int code = n & 7;
    const bf16* in_row = input + static_cast<int64_t>(b) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
      const uint32_t packed = static_cast<uint32_t>(
          qweight[static_cast<int64_t>(k) * packed_total + c]);
      int zero = 8;
      if (qzeros != nullptr) {
        const uint32_t zpacked = static_cast<uint32_t>(
            qzeros[static_cast<int64_t>(k / group_size) * packed_total + c]);
        zero = awq_unpack(zpacked, code);
      }
      const float scale = __bfloat162float(
          scales[static_cast<int64_t>(k / group_size) * total_out + out_offset + n]);
      sum += static_cast<float>(awq_unpack(packed, code) - zero) * scale *
             __bfloat162float(in_row[k]);
    }
    output[idx] = __float2bfloat16(sum);
  }
}
}  // namespace

void awq_matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& qweight,
                          const tensor::Tensor& qzeros, const tensor::Tensor& scales,
                          const tensor::Tensor& output, const AwqMatmulParams& params,
                          const CudaConfig* config) {
  CHECK(!input.is_empty() && !qweight.is_empty() && !scales.is_empty() && !output.is_empty());
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(qweight.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  const int32_t K = qweight.get_dim(0);             // input dim per sample
  const int32_t packed_total = qweight.get_dim(1);  // packed output columns
  const int32_t N = params.out_features;
  const int32_t total_out = scales.get_dim(1);
  const int32_t group_size = params.group_size;
  CHECK_GT(K, 0);
  CHECK_GT(N, 0);
  CHECK_EQ(N % 8, 0);
  CHECK_EQ(params.out_offset % 8, 0);
  CHECK_EQ(input.size() % K, 0);
  CHECK_LE(params.out_offset + N, total_out);
  CHECK_EQ(scales.get_dim(0), K / group_size);

  const int32_t batch = static_cast<int32_t>(input.size() / K);
  const bool zero_point = params.zero_point && !qzeros.is_empty();
  const int32_t* qzeros_ptr = zero_point ? qzeros.ptr<int32_t>() : nullptr;
  if (zero_point) {
    CHECK(qzeros.device_type() == base::DeviceType::kDeviceCUDA);
  }

  const bf16* input_ptr = input.ptr<bf16>();
  const int32_t* qweight_ptr = qweight.ptr<int32_t>();
  const bf16* scales_ptr = scales.ptr<bf16>();
  bf16* output_ptr = const_cast<bf16*>(output.ptr<bf16>());
  cudaStream_t stream = config ? config->stream : nullptr;

  static int cc_major = -1;
  static int sm_count = -1;
  if (cc_major < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  }

  // The tiled kernel hoists one group's scales/zero point per kKChunk-wide chunk.
  if (cc_major >= 8 && group_size % kKChunk == 0) {
    const int gx = (N + kOutPerBlock - 1) / kOutPerBlock;
    const int gy = (batch + kBatchPerBlock - 1) / kBatchPerBlock;
    // Split K enough to keep ~4 blocks per SM in flight; cap each slice at one
    // full shared-memory chunk.
    const int chunks = (K + kKChunk - 1) / kKChunk;
    int split = (4 * sm_count + gx * gy - 1) / (gx * gy);
    split = split < 1 ? 1 : split;
    const int min_split = batch <= 64 ? 8 : 4;
    split = split < min_split ? min_split : split;
    split = split > chunks ? chunks : split;
    split = split > 16 ? 16 : split;
    if (split > 1) {
      // Split-K blocks accumulate with atomics; the direct-store path writes
      // every output element exactly once and needs no zero fill.
      const size_t out_bytes = static_cast<size_t>(batch) * N * sizeof(bf16);
      if (stream) {
        cudaMemsetAsync(output_ptr, 0, out_bytes, stream);
      } else {
        cudaMemset(output_ptr, 0, out_bytes);
      }
    }
    const int warps_m = batch <= 16 ? 1 : (batch <= 32 ? 2 : kWarps);
    const dim3 grid(gx, gy, split);
    if (stream) {
      awq_matmul_bf16_wmma<<<grid, kWarps * 32, 0, stream>>>(
          input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, K, N, packed_total,
          total_out, params.out_offset, group_size, batch, warps_m);
    } else {
      awq_matmul_bf16_wmma<<<grid, kWarps * 32>>>(
          input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, K, N, packed_total,
          total_out, params.out_offset, group_size, batch, warps_m);
    }
  } else {
    const int64_t total = static_cast<int64_t>(batch) * N;
    const int threads = 256;
    const int64_t want_blocks = (total + threads - 1) / threads;
    const int blocks = static_cast<int>(want_blocks > 65535 ? 65535 : want_blocks);
    if (stream) {
      awq_matmul_bf16_scalar<<<blocks, threads, 0, stream>>>(
          input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, K, N, packed_total,
          total_out, params.out_offset, group_size, batch);
    } else {
      awq_matmul_bf16_scalar<<<blocks, threads>>>(
          input_ptr, qweight_ptr, qzeros_ptr, scales_ptr, output_ptr, K, N, packed_total,
          total_out, params.out_offset, group_size, batch);
    }
  }
  CUDA_KERNEL_CHECK();
}
}  // namespace kernel
