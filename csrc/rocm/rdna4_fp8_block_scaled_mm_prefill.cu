// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string>
#include <torch/all.h>

namespace {

using i32x2 = int __attribute__((ext_vector_type(2)));
using i32x4 = int __attribute__((ext_vector_type(4)));
using f32x8 = float __attribute__((ext_vector_type(8)));

constexpr int BK = 128;
constexpr int LDA = BK + 8;
constexpr int LDB = BK;

__device__ __forceinline__ uint32_t bf16_rne(float value) {
  uint32_t bits = __builtin_bit_cast(uint32_t, value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return bits >> 16;
}

// Every geometry contains eight native wave32 waves, each owning an
// independent 32x64 output region. The aspect ratio changes which operand is
// reused most heavily without changing the WMMA work per workgroup.
// Each K128 partial is accumulated independently in FP32 before its row/block
// scales are applied to the long-lived FP32 output accumulators.
template <int TileM, int TileN, int APrefetch, int BPrefetch, int GroupM,
          int FixedM, int FixedN, int FixedK, int FixedWeightStride>
__global__ __launch_bounds__((TileM / 32) * (TileN / 64) * 32)
    __attribute__((amdgpu_waves_per_eu(1))) void block_scaled_fp8_gemm_kernel(
        const uint8_t* __restrict__ a, const uint8_t* __restrict__ weight,
        const float* __restrict__ a_scale,
        const float* __restrict__ weight_scale,
        hip_bfloat16* __restrict__ output, int runtime_m, int runtime_n,
        int runtime_k, int runtime_weight_stride) {
  constexpr int BM = TileM;
  constexpr int BN = TileN;
  constexpr int Threads = (BM / 32) * (BN / 64) * 32;
  constexpr int AWaves = BN / 64;
  constexpr int ALoads = (BM * BK) / (16 * Threads);
  constexpr int BLoads = (BN * BK) / (16 * Threads);
  static_assert(Threads == 128 || Threads == 256);
  static_assert(APrefetch <= ALoads && BPrefetch <= BLoads);
  __shared__ __align__(16) uint8_t lds[BM * LDA + BN * LDB];
  uint8_t* const lds_a = lds;
  uint8_t* const lds_b = lds + BM * LDA;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wave_m = wave / AWaves;
  const int wave_n = wave % AWaves;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int lane_half = lane >> 4;
  const int m = FixedM != 0 ? FixedM : runtime_m;
  const int n = FixedN != 0 ? FixedN : runtime_n;
  const int k = FixedK != 0 ? FixedK : runtime_k;
  const int weight_stride =
      FixedWeightStride != 0 ? FixedWeightStride : runtime_weight_stride;
  const int num_pid_m = (m + BM - 1) / BM;
  const int num_pid_n = n / BN;
  const int pid = blockIdx.x;
  const int num_pid_in_group = GroupM * num_pid_n;
  const int group_id = pid / num_pid_in_group;
  const int first_pid_m = group_id * GroupM;
  const int remaining_pid_m = num_pid_m - first_pid_m;
  const int group_size_m = remaining_pid_m < GroupM ? remaining_pid_m : GroupM;
  const int pid_in_group = pid % num_pid_in_group;
  const int pid_m = first_pid_m + pid_in_group % group_size_m;
  const int pid_n = pid_in_group / group_size_m;
  const int m0 = pid_m * BM;
  const int n0 = pid_n * BN;
  const int num_k_blocks = k >> 7;
  const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<uint8_t*>(a), 0, static_cast<int64_t>(m) * k, 0x31004000);
  const auto b_rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<uint8_t*>(weight), 0, static_cast<int64_t>(n) * weight_stride,
      0x31004000);
  const auto as_rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<float*>(a_scale), 0,
      static_cast<int64_t>(m) * num_k_blocks * sizeof(float), 0x31004000);
  const auto out_rsrc = __builtin_amdgcn_make_buffer_rsrc(
      output, 0, static_cast<int64_t>(m) * n * sizeof(hip_bfloat16),
      0x31004000);
  f32x8 total[8];
#pragma unroll
  for (int ni = 0; ni < 8; ++ni) {
    total[ni] = f32x8{};
  }

  int4 staged_a[APrefetch];
  int4 staged_b[BPrefetch];
#pragma unroll
  for (int q = 0; q < APrefetch; ++q) {
    const int v = tid + q * Threads;
    const int row = v >> 3;
    const int ko = (v & 7) << 4;
    const int global_row = m0 + row;
    staged_a[q] =
        __builtin_bit_cast(int4, __builtin_amdgcn_raw_buffer_load_b128(
                                     a_rsrc, global_row * k + ko, 0, 0));
  }
#pragma unroll
  for (int q = 0; q < BPrefetch; ++q) {
    const int v = tid + q * Threads;
    const int row = v >> 3;
    const int ko = (v & 7) << 4;
    staged_b[q] = __builtin_bit_cast(
        int4, __builtin_amdgcn_raw_buffer_load_b128(
                  b_rsrc, (n0 + row) * weight_stride + ko, 0, 0));
  }

  for (int kb = 0; kb < num_k_blocks; ++kb) {
    if (kb != 0) {
      __syncthreads();
    }

    // Commit the single prefetched tile into one physical LDS stage.
#pragma unroll
    for (int q = 0; q < APrefetch; ++q) {
      const int v = tid + q * Threads;
      const int row = v >> 3;
      const int ko = (v & 7) << 4;
      *reinterpret_cast<int4*>(lds_a + row * LDA + ko) = staged_a[q];
    }
    for (int q = APrefetch; q < ALoads; ++q) {
      const int v = tid + q * Threads;
      const int row = v >> 3;
      const int ko = (v & 7) << 4;
      const int global_row = m0 + row;
      const int4 value = __builtin_bit_cast(
          int4, __builtin_amdgcn_raw_buffer_load_b128(
                    a_rsrc, global_row * k + kb * BK + ko, 0, 0));
      *reinterpret_cast<int4*>(lds_a + row * LDA + ko) = value;
    }
#pragma unroll
    for (int q = 0; q < BPrefetch; ++q) {
      const int v = tid + q * Threads;
      const int row = v >> 3;
      const int ko = (v & 7) << 4;
      const int swizzled_ko = ((ko >> 4) ^ (row & 7)) << 4;
      *reinterpret_cast<int4*>(lds_b + row * LDB + swizzled_ko) = staged_b[q];
    }
    for (int q = BPrefetch; q < BLoads; ++q) {
      const int v = tid + q * Threads;
      const int row = v >> 3;
      const int ko = (v & 7) << 4;
      const int4 value = __builtin_bit_cast(
          int4, __builtin_amdgcn_raw_buffer_load_b128(
                    b_rsrc, (n0 + row) * weight_stride + kb * BK + ko, 0, 0));
      const int swizzled_ko = ((ko >> 4) ^ (row & 7)) << 4;
      *reinterpret_cast<int4*>(lds_b + row * LDB + swizzled_ko) = value;
    }
    asm volatile("s_wait_dscnt 0\n\ts_barrier_signal -1" ::: "memory");

    // Start the next tile's global reads before consuming the current LDS
    // tile. Registers provide latency overlap without a second LDS stage.
    {
      const int next_k0 = (kb + 1) * BK;
#pragma unroll
      for (int q = 0; q < APrefetch; ++q) {
        const int v = tid + q * Threads;
        const int row = v >> 3;
        const int ko = (v & 7) << 4;
        const int global_row = m0 + row;
        staged_a[q] = __builtin_bit_cast(
            int4, __builtin_amdgcn_raw_buffer_load_b128(
                      a_rsrc, global_row * k + next_k0 + ko, 0, 0));
      }
#pragma unroll
      for (int q = 0; q < BPrefetch; ++q) {
        const int v = tid + q * Threads;
        const int row = v >> 3;
        const int ko = (v & 7) << 4;
        staged_b[q] = __builtin_bit_cast(
            int4, __builtin_amdgcn_raw_buffer_load_b128(
                      b_rsrc, (n0 + row) * weight_stride + next_k0 + ko, 0, 0));
      }
    }
    asm volatile("s_barrier_wait 0xffff\n\tglobal_inv scope:SCOPE_SE" ::
                     : "memory");
    // Issue scale reads before the WMMA train so its 64 matrix instructions
    // can cover their memory latency.  Values remain live only until this
    // K128 partial is folded into the long-lived accumulators.
    float row_scales[2][8];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ri = 0; ri < 8; ++ri) {
        const int row = m0 + wave_m * 32 + mi * 16 + lane_half * 8 + ri;
        row_scales[mi][ri] = __builtin_bit_cast(
            float,
            __builtin_amdgcn_raw_buffer_load_b32(
                as_rsrc, (row * num_k_blocks + kb) * sizeof(float), 0, 0));
      }
    }
    const int n_scale_block = (n0 + wave_n * 64) >> 7;
    const float b_scale = weight_scale[n_scale_block * num_k_blocks + kb];

    f32x8 partial[8];
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      partial[ni] = f32x8{};
    }

#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      const int k_lane = ki * 16 + lane_half * 8;
      i32x2 a_frag[2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int a_row = wave_m * 32 + mi * 16 + lane16;
        a_frag[mi] =
            *reinterpret_cast<const i32x2*>(lds_a + a_row * LDA + k_lane);
      }
      i32x2 b_frag[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int b_row = wave_n * 64 + ni * 16 + lane16;
        const int swizzled_k =
            (((k_lane >> 4) ^ (b_row & 7)) << 4) + (k_lane & 15);
        b_frag[ni] =
            *reinterpret_cast<const i32x2*>(lds_b + b_row * LDB + swizzled_k);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int idx = mi * 4 + ni;
          partial[idx] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
              a_frag[mi], b_frag[ni], partial[idx]);
        }
      }
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
        for (int ri = 0; ri < 8; ++ri) {
          const int idx = mi * 4 + ni;
          total[idx][ri] += partial[idx][ri] * (row_scales[mi][ri] * b_scale);
        }
      }
    }
  }

  // Reuse the now-dead input stage for the largest aligned output exchange
  // that fits. Taller geometries exchange several 32-row wave strips at once,
  // amortizing barriers without allocating more LDS.
  __syncthreads();
  float* const lds_output = reinterpret_cast<float*>(lds);
  constexpr int OutputRows = BM == 64 ? 32 : (BM == 128 ? 64 : 128);
  constexpr int OutputWaveRows = OutputRows / 32;
  static_assert(OutputRows * BN * sizeof(float) <= BM * LDA + BN * LDB);
#pragma unroll
  for (int group = 0; group < BM / OutputRows; ++group) {
    if (wave_m / OutputWaveRows == group) {
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
          for (int ri = 0; ri < 8; ++ri) {
            const int idx = mi * 4 + ni;
            const int local_wave_m = wave_m % OutputWaveRows;
            lds_output[(local_wave_m * 32 + mi * 16 + lane_half * 8 + ri) * BN +
                       wave_n * 64 + ni * 16 + lane16] = total[idx][ri];
          }
        }
      }
    }
    __syncthreads();

    // The row-major exchange lets every lane emit aligned 16-byte stores.
#pragma unroll
    for (int q = 0; q < (OutputRows * BN / 8) / Threads; ++q) {
      const int v = tid + q * Threads;
      const int ri = v / (BN / 8);
      const int col = (v % (BN / 8)) << 3;
      const int row = m0 + group * OutputRows + ri;
      if (row < m) {
        const float4 lo =
            *reinterpret_cast<const float4*>(lds_output + ri * BN + col);
        const float4 hi =
            *reinterpret_cast<const float4*>(lds_output + ri * BN + col + 4);
        uint4 packed;
        packed.x = bf16_rne(lo.x) | (bf16_rne(lo.y) << 16);
        packed.y = bf16_rne(lo.z) | (bf16_rne(lo.w) << 16);
        packed.z = bf16_rne(hi.x) | (bf16_rne(hi.y) << 16);
        packed.w = bf16_rne(hi.z) | (bf16_rne(hi.w) << 16);
        __builtin_amdgcn_raw_buffer_store_b128(
            __builtin_bit_cast(i32x4, packed), out_rsrc,
            (static_cast<int64_t>(row) * n + n0 + col) * sizeof(hip_bfloat16),
            0, 0);
      }
    }
    __syncthreads();
  }
}

template <int TileM, int TileN, int APrefetch, int BPrefetch, int GroupM,
          int FixedM = 0, int FixedN = 0, int FixedK = 0,
          int FixedWeightStride = 0>
void launch_tiled(const void* a, const void* weight, const float* a_scale,
                  const float* weight_scale, void* output, int m, int n, int k,
                  int weight_stride, hipStream_t stream) {
  constexpr int Threads = (TileM / 32) * (TileN / 64) * 32;
  const dim3 grid((n / TileN) * ((m + TileM - 1) / TileM));
  hipLaunchKernelGGL(
      (block_scaled_fp8_gemm_kernel<TileM, TileN, APrefetch, BPrefetch, GroupM,
                                    FixedM, FixedN, FixedK, FixedWeightStride>),
      grid, dim3(Threads), 0, stream, static_cast<const uint8_t*>(a),
      static_cast<const uint8_t*>(weight), a_scale, weight_scale,
      static_cast<hip_bfloat16*>(output), m, n, k, weight_stride);
}

}  // namespace

static void launch_block_scaled_fp8_gemm(const void* a, const void* weight,
                                         const float* a_scale,
                                         const float* weight_scale,
                                         void* output, int m, int n, int k,
                                         int weight_stride,
                                         hipStream_t stream) {
  // Reuse B across nearby M tiles once the weight surface is large enough to
  // dominate.  The thresholds are regions from the coarse and interpolation
  // sweeps, not individual shape exceptions.
  const bool balanced_reuse =
      m >= 128 &&
      ((n >= 16384 && k >= 4096) || (n >= 10240 && k >= 6144) ||
       (n >= 7168 && k >= 10240) ||
       // At large M the balanced tile amortizes B traffic and its full
       // register prefetch hides the reduction latency across the full N
       // surface, including the low-K prefill band.
       (m >= 4096 && n >= 5120) || (m >= 896 && n >= 12288 && k >= 3072) ||
       (m >= 768 && n >= 6144 && k >= 6144) ||
       (m >= 384 && m <= 640 && n >= 5120 && n < 8192));
  // A 64-row tile avoids wasting half of a 128-row workgroup for these
  // residual-M bands while preserving the grouped B-reuse launch order.
  const bool short_m_weight_reuse =
      m >= 96 && m < 384 && (m & 127) != 0 && n >= 10240 && k >= 6144;
  // Long reductions at small N benefit from the smaller 128-column tile:
  // more independent workgroups hide the K-loop latency without losing B
  // locality because the grouped scheduler visits up to 32 M tiles together.
  // Switch large-M small-N reductions back to the wide default geometry.  The
  // narrower tile remains better while M is too short to fill that grid.
  const bool small_n_long_k = m >= 128 && m < 4096 && n <= 4096 && k >= 6144;
  if ((n & 255) != 0 || short_m_weight_reuse || small_n_long_k ||
      (m >= 192 && m <= 384 && n >= 4096 && n <= 7168 && k >= 6144)) {
    launch_tiled<64, 128, 2, 5, 32>(a, weight, a_scale, weight_scale, output, m,
                                    n, k, weight_stride, stream);
  } else if (m == 256 && n == 8192 && k == 5120 && weight_stride == 5376) {
    launch_tiled<64, 256, 2, 2, 1, 256, 8192, 5120, 5376>(
        a, weight, a_scale, weight_scale, output, m, n, k, weight_stride,
        stream);
  } else if (balanced_reuse) {
    launch_tiled<128, 128, 4, 4, 32>(a, weight, a_scale, weight_scale, output,
                                     m, n, k, weight_stride, stream);
  } else if (n == 8192 && k == 5120) {
    launch_tiled<64, 256, 2, 2, 1, 0, 8192, 5120, 0>(a, weight, a_scale,
                                                     weight_scale, output, m, n,
                                                     k, weight_stride, stream);
  } else {
    launch_tiled<64, 256, 2, 2, 1>(a, weight, a_scale, weight_scale, output, m,
                                   n, k, weight_stride, stream);
  }
}

torch::Tensor rdna4_fp8_block_scaled_mm_prefill(
    const torch::Tensor& a, const torch::Tensor& weight,
    const torch::Tensor& a_scale, const torch::Tensor& weight_scale) {
  TORCH_CHECK(a.is_cuda() && weight.is_cuda() && a_scale.is_cuda() &&
                  weight_scale.is_cuda(),
              "RDNA4 block-FP8 inputs must be on the GPU");
  TORCH_CHECK(a.dim() == 2 && weight.dim() == 2 && a_scale.dim() == 2 &&
                  weight_scale.dim() == 2,
              "RDNA4 block-FP8 inputs must be rank two");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn &&
                  weight.scalar_type() == at::kFloat8_e4m3fn,
              "RDNA4 block-FP8 operands must use float8_e4m3fn");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "RDNA4 block-FP8 scales must use float32");
  TORCH_CHECK(a.device() == weight.device() && a.device() == a_scale.device() &&
                  a.device() == weight_scale.device(),
              "RDNA4 block-FP8 inputs must share a device");

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(weight.size(0));
  TORCH_CHECK(m > 64, "RDNA4 prefill route requires M > 64");
  TORCH_CHECK(weight.size(1) == k, "RDNA4 block-FP8 K dimensions must match");
  TORCH_CHECK(n > 0 && n % 128 == 0,
              "RDNA4 prefill route requires N divisible by 128");
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "RDNA4 prefill route requires K divisible by 128");
  TORCH_CHECK(a.is_contiguous() && a_scale.is_contiguous() &&
                  weight_scale.is_contiguous() && weight.stride(1) == 1,
              "RDNA4 prefill route received an unsupported layout");
  TORCH_CHECK(a_scale.size(0) == m && a_scale.size(1) == k / 128,
              "RDNA4 prefill route activation scale shape mismatch");
  TORCH_CHECK(
      weight_scale.size(0) == n / 128 && weight_scale.size(1) == k / 128,
      "RDNA4 prefill route weight scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  TORCH_CHECK(arch.find("gfx1200") != std::string::npos ||
                  arch.find("gfx1201") != std::string::npos,
              "RDNA4 prefill route requires gfx1200 or gfx1201, got ", arch);

  auto out = torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_block_scaled_fp8_gemm(
      a.data_ptr(), weight.data_ptr(), a_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(), out.data_ptr(), m, n, k,
      static_cast<int>(weight.stride(0)), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
