// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace {

using F8 = rocwmma::float8_t;
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, F8,
                                rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, F8,
                                rocwmma::col_major>;
using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__global__ __launch_bounds__(128) void decode_splitk_wmma(
    const F8* __restrict__ a, const F8* __restrict__ b,
    const float* __restrict__ as, const float* __restrict__ bs,
    hip_bfloat16* __restrict__ out, int m, int n, int k, int stride_b) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int pair = wave >> 1;
  const int half_id = wave & 1;
  const int n0 = blockIdx.x * 128 + pair * 64;
  const int scale_blocks = k >> 7;
  const int half_blocks = (scale_blocks + 1) >> 1;
  const int kb_begin = half_id * half_blocks;
  const int kb_end =
      kb_begin + half_blocks < scale_blocks ? kb_begin + half_blocks
                                            : scale_blocks;

  __shared__ float partial[4][4][256];

#pragma unroll
  for (int row = 0; row < 2; ++row) {
    if (row >= m) break;

    FragC total[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      rocwmma::fill_fragment(total[j], 0.0f);
    }

    for (int kb = kb_begin; kb < kb_end; ++kb) {
      FragC block_acc[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        rocwmma::fill_fragment(block_acc[j], 0.0f);
      }

      const int kbase = kb << 7;
#pragma unroll
      for (int kk = 0; kk < 128; kk += 16) {
        FragA af;
        rocwmma::load_matrix_sync(af, a + row * k + kbase + kk, 0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          FragB bf;
          rocwmma::load_matrix_sync(
              bf, b + (n0 + j * 16) * stride_b + kbase + kk,
              static_cast<uint32_t>(stride_b));
          rocwmma::mma_sync(block_acc[j], af, bf, block_acc[j]);
        }
      }

      const float scale = as[row * scale_blocks + kb] *
                          bs[blockIdx.x * scale_blocks + kb];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int x = 0; x < FragC::num_elements; ++x) {
          total[j].x[x] += block_acc[j].x[x] * scale;
        }
      }
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      rocwmma::store_matrix_sync(partial[wave][j], total[j], 16,
                                 rocwmma::layout_t::mem_row_major);
    }
    __syncthreads();

    const int col = tid;
    const int out_pair = col >> 6;
    const int in_pair = col & 63;
    const int frag = in_pair >> 4;
    const int frag_col = in_pair & 15;
    const int wave0 = out_pair << 1;
    const float value = partial[wave0][frag][frag_col] +
                        partial[wave0 + 1][frag][frag_col];
    out[row * n + blockIdx.x * 128 + col] = hip_bfloat16(value);
    __syncthreads();
  }
}

__global__ __launch_bounds__(64) void decode_splitk_wmma_m2(
    const F8* __restrict__ a, const F8* __restrict__ b,
    const float* __restrict__ as, const float* __restrict__ bs,
    hip_bfloat16* __restrict__ out, int n, int k, int stride_b) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n0 = blockIdx.x * 64;
  const int scale_blocks = k >> 7;
  const int half_blocks = (scale_blocks + 1) >> 1;
  const int kb_begin = wave * half_blocks;
  const int kb_end =
      kb_begin + half_blocks < scale_blocks ? kb_begin + half_blocks
                                            : scale_blocks;

  __shared__ float partial[2][4][32];

  FragC total[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    rocwmma::fill_fragment(total[j], 0.0f);
  }

  for (int kb = kb_begin; kb < kb_end; ++kb) {
    FragC block_acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      rocwmma::fill_fragment(block_acc[j], 0.0f);
    }

    const int kbase = kb << 7;
#pragma unroll
    for (int kk = 0; kk < 128; kk += 16) {
      FragA af;
      rocwmma::fill_fragment(af, 0.0f);
      const int arow = lane & 15;
      if (arow < 2) {
        const int ak = kbase + kk + (lane >> 4) * 8;
        af.mStorage = *reinterpret_cast<const FragA::Traits::StorageT*>(
            a + arow * k + ak);
      }
      FragB bf[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        rocwmma::load_matrix_sync(
            bf[j], b + (n0 + j * 16) * stride_b + kbase + kk,
            static_cast<uint32_t>(stride_b));
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        rocwmma::mma_sync(block_acc[j], af, bf[j], block_acc[j]);
      }
    }

    const float b_scale = bs[(blockIdx.x >> 1) * scale_blocks + kb];
    const float scale0 = as[kb] * b_scale;
    const float scale1 = as[scale_blocks + kb] * b_scale;
    if (lane < 16) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        total[j].x[0] += block_acc[j].x[0] * scale0;
        total[j].x[1] += block_acc[j].x[1] * scale1;
      }
    }
  }

  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      partial[wave][j][lane] = total[j].x[0];
      partial[wave][j][16 + lane] = total[j].x[1];
    }
  }
  __syncthreads();

  const int frag = tid >> 4;
  const int frag_col = tid & 15;
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    const int idx = row * 16 + frag_col;
    const float value = partial[0][frag][idx] + partial[1][frag][idx];
    out[row * n + blockIdx.x * 64 + tid] = hip_bfloat16(value);
  }
}

__global__ __launch_bounds__(128) void decode_nosplit_wmma_m2_wide(
    const F8* __restrict__ a, const F8* __restrict__ b,
    const float* __restrict__ as, const float* __restrict__ bs,
    hip_bfloat16* __restrict__ out, int n, int k, int stride_b) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int n0 = blockIdx.x * 128 + wave * 32;
  const int scale_blocks = k >> 7;

  FragC total[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    rocwmma::fill_fragment(total[j], 0.0f);
  }

  for (int kb = 0; kb < scale_blocks; ++kb) {
    FragC block_acc[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      rocwmma::fill_fragment(block_acc[j], 0.0f);
    }

    const int kbase = kb << 7;
#pragma unroll
    for (int kk = 0; kk < 128; kk += 16) {
      FragA af;
      rocwmma::fill_fragment(af, 0.0f);
      const int arow = lane & 15;
      if (arow < 2) {
        const int ak = kbase + kk + (lane >> 4) * 8;
        af.mStorage = *reinterpret_cast<const FragA::Traits::StorageT*>(
            a + arow * k + ak);
      }

      FragB bf[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        rocwmma::load_matrix_sync(
            bf[j], b + (n0 + j * 16) * stride_b + kbase + kk,
            static_cast<uint32_t>(stride_b));
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        rocwmma::mma_sync(block_acc[j], af, bf[j], block_acc[j]);
      }
    }

    const float b_scale = bs[blockIdx.x * scale_blocks + kb];
    const float scale0 = as[kb] * b_scale;
    const float scale1 = as[scale_blocks + kb] * b_scale;
    if (lane < 16) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        total[j].x[0] += block_acc[j].x[0] * scale0;
        total[j].x[1] += block_acc[j].x[1] * scale1;
      }
    }
  }

  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int col = n0 + j * 16 + lane;
      out[col] = hip_bfloat16(total[j].x[0]);
      out[n + col] = hip_bfloat16(total[j].x[1]);
    }
  }
}

}  // namespace

torch::Tensor rdna4_fp8_block_scaled_mm_decode(
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
  TORCH_CHECK(m == 1 || m == 2, "RDNA4 decode supports only M=1 or M=2");
  TORCH_CHECK(weight.size(1) == k, "RDNA4 block-FP8 K dimensions must match");
  TORCH_CHECK(n % 128 == 0, "RDNA4 decode requires N divisible by 128");
  TORCH_CHECK(k % 128 == 0, "RDNA4 decode requires K divisible by 128");
  TORCH_CHECK(a.is_contiguous() && a_scale.is_contiguous() &&
                  weight_scale.is_contiguous() && weight.stride(1) == 1,
              "RDNA4 decode received an unsupported layout");
  TORCH_CHECK(a_scale.size(0) == m && a_scale.size(1) == k / 128,
              "RDNA4 decode activation scale shape mismatch");
  TORCH_CHECK(weight_scale.size(0) == n / 128 &&
                  weight_scale.size(1) == k / 128,
              "RDNA4 decode weight scale shape mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  TORCH_CHECK(arch.find("gfx1200") != std::string::npos ||
                  arch.find("gfx1201") != std::string::npos,
              "RDNA4 decode requires gfx1200 or gfx1201, got ", arch);
  auto out = torch::empty({m, n}, a.options().dtype(torch::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (m == 2 && n == 17408) {
    hipLaunchKernelGGL(
        decode_nosplit_wmma_m2_wide, dim3(n / 128), dim3(128), 0, stream,
        reinterpret_cast<const F8*>(a.data_ptr()),
        reinterpret_cast<const F8*>(weight.data_ptr()),
        a_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        reinterpret_cast<hip_bfloat16*>(out.data_ptr()), n, k,
        static_cast<int>(weight.stride(0)));
  } else if (m == 2) {
    hipLaunchKernelGGL(
        decode_splitk_wmma_m2, dim3(n / 64), dim3(64), 0, stream,
        reinterpret_cast<const F8*>(a.data_ptr()),
        reinterpret_cast<const F8*>(weight.data_ptr()),
        a_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        reinterpret_cast<hip_bfloat16*>(out.data_ptr()), n, k,
        static_cast<int>(weight.stride(0)));
  } else {
    hipLaunchKernelGGL(
        decode_splitk_wmma, dim3(n / 128), dim3(128), 0, stream,
        reinterpret_cast<const F8*>(a.data_ptr()),
        reinterpret_cast<const F8*>(weight.data_ptr()),
        a_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
        reinterpret_cast<hip_bfloat16*>(out.data_ptr()), m, n, k,
        static_cast<int>(weight.stride(0)));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
