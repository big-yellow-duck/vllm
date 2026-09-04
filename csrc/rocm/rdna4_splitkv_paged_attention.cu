// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include <cstdint>

namespace {

constexpr int kHead = 256;
constexpr int kQPerKV = 6;
constexpr int kKVHeads = 2;
constexpr int kTile = 32;
constexpr int kPitch = 33;
constexpr int kPage = 1568;

using WmmaBF16 = rocwmma::bfloat16_t;
using WmmaA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, WmmaBF16,
                                rocwmma::row_major>;
using WmmaB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, WmmaBF16,
                                rocwmma::row_major>;
using WmmaBCol = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, WmmaBF16,
                                   rocwmma::col_major>;
using WmmaC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float fp8e4m3fn_to_float(uint8_t bits) {
  return __builtin_amdgcn_cvt_f32_fp8(static_cast<uint32_t>(bits), 0);
}

__device__ __forceinline__ float bf16_to_float(hip_bfloat16 value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ hip_bfloat16 float_to_bf16(float value) {
  return hip_bfloat16::round_to_bfloat16(value);
}

using NativeBF16x2 = __bf16 __attribute__((ext_vector_type(2)));
using NativeU16x2 = uint16_t __attribute__((ext_vector_type(2)));
using NativeU32x4 = uint32_t __attribute__((ext_vector_type(4)));

template <bool HighPair>
__device__ __forceinline__ NativeU16x2 fp8x2_scaled_to_bf16x2(uint32_t bits,
                                                              float scale) {
  auto fp32 = __builtin_amdgcn_cvt_pk_f32_fp8(bits, HighPair);
  fp32 *= scale;
  const NativeBF16x2 bf16 = __builtin_convertvector(fp32, NativeBF16x2);
  return __builtin_bit_cast(NativeU16x2, bf16);
}

__device__ __forceinline__ float wave_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset, 32));
  }
  return __shfl(value, 0, 32);
}

__device__ __forceinline__ float wave_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down(value, offset, 32);
  }
  return __shfl(value, 0, 32);
}

template <bool FP8, bool ExactRows, bool StaggerK, bool TokenHalves = false>
__global__ __launch_bounds__(256, 2) void splitkv_stage1(
    const hip_bfloat16* __restrict__ query, const void* __restrict__ key_cache,
    const void* __restrict__ value_cache, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ query_start_loc,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float* __restrict__ mid_out, float* __restrict__ mid_lse, int batch,
    int splits, int table_stride, int64_t q_stride0, int64_t q_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2, int64_t k_stride3,
    int64_t k_stride4, int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    int64_t v_stride3, int64_t mo_stride0, int64_t mo_stride1,
    int64_t mo_stride2, int64_t ml_stride0, int64_t ml_stride1,
    int64_t ml_stride2) {
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int half = TokenHalves ? static_cast<int>(blockIdx.x >> 5) : 0;
  const int logical = TokenHalves ? static_cast<int>(blockIdx.x & 31)
                                  : static_cast<int>(blockIdx.x);
  const int split = logical % splits;
  const int item = logical / splits;
  const int kv_head = item & 1;
  const int seq = item >> 1;
  if (seq >= batch) return;
  const int query_row = query_start_loc[seq];
  if (query_start_loc[seq + 1] - query_row != 1) return;
  constexpr int tile_size = 64;
  constexpr int tile_pitch = 65;
  constexpr bool RawK = FP8 && ExactRows;
  constexpr int raw_k_tile_bytes = kHead * tile_size;
  constexpr int raw_k_buffers = TokenHalves ? 1 : 2;

  // The exact-row FP8 path consumes V directly from its prefetched registers
  // and retains the six live output rows in registers.  It consequently only
  // needs K, scores, Q and probabilities in LDS; V/output never make the
  // global->LDS->barrier->WMMA->LDS->barrier round trip.
  __shared__ __align__(
      16) unsigned char workspace[RawK ? (TokenHalves ? 21808 : 38192)
                                       : (ExactRows ? 38704 : 59616)];
  auto* q_shared = reinterpret_cast<hip_bfloat16*>(
      workspace +
      (RawK ? (TokenHalves ? 17952 : 34336) : (ExactRows ? 34848 : 51424)));
  auto* k_shared = reinterpret_cast<hip_bfloat16*>(workspace);
  auto* score_shared = reinterpret_cast<float*>(
      workspace + (RawK ? raw_k_buffers * raw_k_tile_bytes
                        : kHead * tile_pitch * sizeof(hip_bfloat16)));
  auto* output_shared = reinterpret_cast<float*>(
      workspace + (ExactRows ? tile_size * 257 * sizeof(hip_bfloat16) : 0));
  auto* v_shared = reinterpret_cast<hip_bfloat16*>(
      workspace + (ExactRows ? 0 : 16 * 257 * sizeof(float)));
  auto* probability = reinterpret_cast<hip_bfloat16*>(
      workspace + (RawK        ? (TokenHalves ? 21024 : 37408)
                   : ExactRows ? 37920
                               : 16 * 257 * sizeof(float) +
                                     tile_size * 257 * sizeof(hip_bfloat16)));
  auto* raw_k = reinterpret_cast<uint8_t*>(workspace);
  __shared__ int physical[RawK ? 2 * tile_size : tile_size];
  __shared__ int page_offset[RawK ? 2 * tile_size : tile_size];
  __shared__ float row_alpha[kQPerKV];
  __shared__ float row_inv_sum[kQPerKV];
  __shared__ float first_weight[TokenHalves ? kQPerKV : 1];
  __shared__ float second_weight[TokenHalves ? kQPerKV : 1];

  const int length = seq_lens[seq];
  const int split_len =
      ((length + splits - 1) / splits + tile_size - 1) / tile_size * tile_size;
  const int full_begin = split * split_len;
  const int full_end = min(full_begin + split_len, length);
  const int tile_count =
      (max(full_end - full_begin, 0) + tile_size - 1) / tile_size;
  const int half_span = ((tile_count + 1) / 2) * tile_size;
  const int begin =
      TokenHalves ? min(full_begin + half * half_span, full_end) : full_begin;
  const int end = TokenHalves ? min(begin + half_span, full_end) : full_end;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float output_acc[2][kQPerKV] = {};
  const float ks = FP8 ? k_scale[0] : 1.0f;
  const float vs = FP8 ? v_scale[0] : 1.0f;
  const auto* key8 = static_cast<const uint8_t*>(key_cache);
  const auto* value8 = static_cast<const uint8_t*>(value_cache);
  const auto* key16 = static_cast<const hip_bfloat16*>(key_cache);
  const auto* value16 = static_cast<const hip_bfloat16*>(value_cache);

  if constexpr (TokenHalves) {
    if (wave < kQPerKV && lane == 0) {
      row_inv_sum[wave] = 0.0f;
    }
  }

  // Keep Q in a non-aliased tail so it is loaded once and survives the
  // K/score -> V/output phase reuse of the main workspace.
  for (int i = tid; i < (ExactRows ? kQPerKV : 16) * kHead; i += blockDim.x) {
    const int row = i / kHead;
    const int d = i & (kHead - 1);
    q_shared[i] = row < kQPerKV
                      ? query[query_row * q_stride0 +
                              (kv_head * kQPerKV + row) * q_stride1 + d]
                      : float_to_bf16(0.0f);
  }
  for (int i = tid; i < (ExactRows ? kQPerKV : 16) * tile_pitch;
       i += blockDim.x) {
    probability[i] = float_to_bf16(0.0f);
  }

  constexpr int pack = FP8 ? 16 : 8;
  constexpr int kVecs = (kHead / pack) * tile_size;
  constexpr int kLoadsPerThread = kVecs / 256;

  // Prime the raw-FP8 K pipeline.  K stays in its cache representation in
  // two compact LDS tiles; conversion is deferred to the four QK waves.
  if constexpr (RawK) {
    if (tid < tile_size) {
      const int token = begin + tid;
      if (token < end) {
        physical[tid] = block_tables[seq * table_stride + token / kPage];
        page_offset[tid] = token % kPage;
      } else {
        physical[tid] = 0;
        page_offset[tid] = 0;
      }
    }
    __syncthreads();
    const int t = tid & (tile_size - 1);
    const int group = tid / tile_size;
    uint4 first_k0 = make_uint4(0, 0, 0, 0);
    uint4 first_k1 = make_uint4(0, 0, 0, 0);
    uint4 first_k2 = make_uint4(0, 0, 0, 0);
    uint4 first_k3 = make_uint4(0, 0, 0, 0);
    if (begin + t < end) {
      const int64_t ki = static_cast<int64_t>(physical[t]) * k_stride0 +
                         kv_head * k_stride1 + group * k_stride2 +
                         page_offset[t] * k_stride3;
      first_k0 = *reinterpret_cast<const uint4*>(key8 + ki);
      first_k1 = *reinterpret_cast<const uint4*>(key8 + ki + 4 * k_stride2);
      first_k2 = *reinterpret_cast<const uint4*>(key8 + ki + 8 * k_stride2);
      first_k3 = *reinterpret_cast<const uint4*>(key8 + ki + 12 * k_stride2);
    }
    *reinterpret_cast<uint4*>(raw_k + (group * tile_size + t) * pack) =
        first_k0;
    *reinterpret_cast<uint4*>(raw_k + ((group + 4) * tile_size + t) * pack) =
        first_k1;
    *reinterpret_cast<uint4*>(raw_k + ((group + 8) * tile_size + t) * pack) =
        first_k2;
    *reinterpret_cast<uint4*>(raw_k + ((group + 12) * tile_size + t) * pack) =
        first_k3;
    __syncthreads();
  }

  for (int tile = begin; tile < end; tile += tile_size) {
    const int pipe = ((tile - begin) / tile_size) & 1;
    if constexpr (!RawK) {
      if (tid < tile_size) {
        const int token = tile + tid;
        if (token < end) {
          physical[tid] = block_tables[seq * table_stride + token / kPage];
          page_offset[tid] = token % kPage;
        } else {
          physical[tid] = 0;
          page_offset[tid] = 0;
        }
      }
      __syncthreads();

      uint4 k_packed[kLoadsPerThread];
#pragma unroll
      for (int iter = 0; iter < kLoadsPerThread; ++iter) {
        const int vec = tid + iter * blockDim.x;
        const int group = vec / tile_size;
        const int t = vec & (tile_size - 1);
        const bool valid = tile + t < end;
        k_packed[iter] = make_uint4(0, 0, 0, 0);
        if (valid) {
          const int64_t ki = static_cast<int64_t>(physical[t]) * k_stride0 +
                             kv_head * k_stride1 + group * k_stride2 +
                             page_offset[t] * k_stride3;
          k_packed[iter] = *reinterpret_cast<const uint4*>(
              (FP8 ? static_cast<const void*>(key8 + ki)
                   : static_cast<const void*>(key16 + ki)));
        }
      }
#pragma unroll
      for (int iter = 0; iter < kLoadsPerThread; ++iter) {
        const int vec = tid + iter * blockDim.x;
        const int group = vec / tile_size;
        const int t = vec & (tile_size - 1);
        const uint4 packed = k_packed[iter];
#pragma unroll
        for (int j = 0; j < pack; ++j) {
          if constexpr (FP8) {
            if ((j & 1) == 0) {
              const uint32_t word = j < 4    ? packed.x
                                    : j < 8  ? packed.y
                                    : j < 12 ? packed.z
                                             : packed.w;
              const auto converted =
                  (j & 2) ? fp8x2_scaled_to_bf16x2<true>(word, ks)
                          : fp8x2_scaled_to_bf16x2<false>(word, ks);
              hip_bfloat16 lo;
              hip_bfloat16 hi;
              lo.data = static_cast<uint16_t>(converted[0]);
              hi.data = static_cast<uint16_t>(converted[1]);
              k_shared[(group * pack + j) * tile_pitch + t] = lo;
              k_shared[(group * pack + j + 1) * tile_pitch + t] = hi;
            }
          } else {
            const uint32_t word = j < 2   ? packed.x
                                  : j < 4 ? packed.y
                                  : j < 6 ? packed.z
                                          : packed.w;
            const uint16_t raw = static_cast<uint16_t>(word >> (16 * (j & 1)));
            hip_bfloat16 value;
            value.data = raw;
            k_shared[(group * pack + j) * tile_pitch + t] = value;
          }
        }
      }
    }

    // K has reached LDS and its packed register batch is dead.  Reuse that
    // register lifetime to launch V well before it is consumed: the K LDS
    // fence, QK WMMA, and softmax reduction now cover the V memory latency.
    constexpr int values_per_vec = FP8 ? 16 : 8;
    constexpr int vLoadsPerThread = tile_size / values_per_vec;
    uint4 v_packed[vLoadsPerThread];
    WmmaBCol v16_packed[8];
    NativeU32x4 next_k0;
    NativeU32x4 next_k1;
    NativeU32x4 next_k2;
    NativeU32x4 next_k3;
    if constexpr (ExactRows) {
      // Issued after K publication below: a workgroup barrier drains VMEM,
      // so placing V here would serialize it behind the K handoff.
    } else if constexpr (FP8) {
#pragma unroll
      for (int vec = 0; vec < vLoadsPerThread; ++vec) {
        const int d = tid;
        const int t0 = vec * values_per_vec;
        const bool valid = tile + t0 < end;
        v_packed[vec] = make_uint4(0, 0, 0, 0);
        if (valid) {
          const int64_t vi = static_cast<int64_t>(physical[t0]) * v_stride0 +
                             kv_head * v_stride1 + d * v_stride2 +
                             page_offset[t0] * v_stride3;
          v_packed[vec] = *reinterpret_cast<const uint4*>(value8 + vi);
        }
      }
    }
    if constexpr (!RawK) {
      __syncthreads();
    }

    if constexpr (ExactRows) {
      // A gfx12 col-major B fragment assigns lane&15 to its matrix column
      // and lane>>4 to one eight-token half.  Load exactly that native
      // fragment shape here.  The eight waves own disjoint 32-D slabs, so
      // compulsory V traffic is unchanged while the LDS handoff disappears.
      auto* v_pairs = reinterpret_cast<uint2*>(v_packed);
#pragma unroll
      for (int token_frag = 0; token_frag < 4; ++token_frag) {
#pragma unroll
        for (int col_frag = 0; col_frag < 2; ++col_frag) {
          const int d = wave * 32 + col_frag * 16 + (lane & 15);
          const int t0 = token_frag * 16 + (lane >> 4) * 8;
          const bool valid = tile + t0 < end;
          const int direct_idx = token_frag * 2 + col_frag;
          if constexpr (FP8) {
            v_pairs[direct_idx] = make_uint2(0, 0);
            if (valid) {
              const int meta = (RawK ? pipe * tile_size : 0) + t0;
              const int64_t vi =
                  static_cast<int64_t>(physical[meta]) * v_stride0 +
                  kv_head * v_stride1 + d * v_stride2 +
                  page_offset[meta] * v_stride3;
              if (tile + t0 + 8 <= end) {
                v_pairs[direct_idx] =
                    *reinterpret_cast<const uint2*>(value8 + vi);
              } else {
                uint64_t packed = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                  if (tile + t0 + i < end) {
                    packed |= static_cast<uint64_t>(value8[vi + i]) << (8 * i);
                  }
                }
                v_pairs[direct_idx] =
                    make_uint2(static_cast<uint32_t>(packed),
                               static_cast<uint32_t>(packed >> 32));
              }
            }
          } else {
            const int d0 = wave * 32 + col_frag * 16;
            const int token0 = token_frag * 16;
            const int meta = (RawK ? pipe * tile_size : 0) + token0;
            const int64_t vi =
                static_cast<int64_t>(physical[meta]) * v_stride0 +
                kv_head * v_stride1 + d0 * v_stride2 +
                page_offset[meta] * v_stride3;
            if (tile + token0 + 16 <= end) {
              rocwmma::load_matrix_sync(v16_packed[direct_idx], value16 + vi,
                                        static_cast<uint32_t>(v_stride2));
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                const int token = t0 + i;
                v16_packed[direct_idx][i] =
                    tile + token < end
                        ? WmmaBF16(value16[vi + (lane & 15) * v_stride2 +
                                           (lane >> 4) * 8 + i])
                        : WmmaBF16(0.0f);
              }
            }
          }
        }
      }
    }

    // Fetch next-tile page metadata on the two otherwise idle QK waves.  The
    // score handoff barrier below makes it visible without adding a barrier.
    const bool has_next_k = tile + tile_size < end;
    if constexpr (RawK) {
      if (tid >= 192) {
        const int t = tid - 192;
        const int meta = (pipe ^ 1) * tile_size + t;
        const int token = tile + tile_size + t;
        if (has_next_k && token < end) {
          physical[meta] = block_tables[seq * table_stride + token / kPage];
          page_offset[meta] = token % kPage;
        } else {
          physical[meta] = 0;
          page_offset[meta] = 0;
        }
      }
      // Waves 4-7 have no QK work.  Start their half of the next K tile now;
      // waves 0-3 retain the original post-QK issue point so QK itself never
      // waits behind a speculative global load.
      if constexpr (StaggerK) {
        if (wave >= 4 && has_next_k) {
          const int t = tid & (tile_size - 1);
          const int group = tid / tile_size;
          const int token = tile + tile_size + t;
          if (token < end) {
            const int next_physical =
                block_tables[seq * table_stride + token / kPage];
            const int next_page_offset = token % kPage;
            const int64_t ki = static_cast<int64_t>(next_physical) * k_stride0 +
                               kv_head * k_stride1 + group * k_stride2 +
                               next_page_offset * k_stride3;
            next_k0 = *reinterpret_cast<const NativeU32x4*>(key8 + ki);
            next_k1 = *reinterpret_cast<const NativeU32x4*>(key8 + ki +
                                                            4 * k_stride2);
            next_k2 = *reinterpret_cast<const NativeU32x4*>(key8 + ki +
                                                            8 * k_stride2);
            next_k3 = *reinterpret_cast<const NativeU32x4*>(key8 + ki +
                                                            12 * k_stride2);
          }
        }
      }
    }

    if (wave < 4) {
      WmmaC score_frag;
      rocwmma::fill_fragment(score_frag, 0.0f);
#pragma unroll
      for (int d = 0; d < kHead; d += 16) {
        WmmaA q_frag;
        WmmaB k_frag;
        if constexpr (ExactRows) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int row = lane & 15;
            const int col = (lane >> 4) * 8 + i;
            q_frag[i] = row < kQPerKV ? reinterpret_cast<const WmmaBF16*>(
                                            q_shared)[row * kHead + d + col]
                                      : WmmaBF16(0.0f);
          }
        } else {
          rocwmma::load_matrix_sync(
              q_frag, reinterpret_cast<const WmmaBF16*>(q_shared) + d,
              static_cast<uint32_t>(kHead));
        }
        if constexpr (RawK) {
          // gfx12 row-major B owns one token column per lane&15 and an
          // eight-D half per lane>>4.  Convert only this native fragment.
          const int token_col = wave * 16 + (lane & 15);
          const auto packed = *reinterpret_cast<const uint2*>(
              raw_k + (TokenHalves ? 0 : pipe * raw_k_tile_bytes) +
              ((d / pack) * tile_size + token_col) * pack + (lane >> 4) * 8);
          const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const auto converted =
                (pair & 1)
                    ? fp8x2_scaled_to_bf16x2<true>(words[pair >> 1], ks)
                    : fp8x2_scaled_to_bf16x2<false>(words[pair >> 1], ks);
            hip_bfloat16 lo;
            hip_bfloat16 hi;
            lo.data = static_cast<uint16_t>(converted[0]);
            hi.data = static_cast<uint16_t>(converted[1]);
            k_frag[pair * 2] = WmmaBF16(lo);
            k_frag[pair * 2 + 1] = WmmaBF16(hi);
          }
        } else {
          rocwmma::load_matrix_sync(
              k_frag,
              reinterpret_cast<const WmmaBF16*>(k_shared) + d * tile_pitch +
                  wave * 16,
              static_cast<uint32_t>(tile_pitch));
        }
        rocwmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
      }
      if constexpr (ExactRows) {
        const auto score_store =
            rocwmma::apply_data_layout<rocwmma::row_major>(score_frag);
        if (lane < 16) {
#pragma unroll
          for (int row = 0; row < kQPerKV; ++row) {
            score_shared[row * tile_pitch + wave * 16 + lane] =
                score_store[row];
          }
        }
      } else {
        rocwmma::store_matrix_sync(score_shared + wave * 16, score_frag,
                                   tile_pitch,
                                   rocwmma::layout_t::mem_row_major);
      }
    }
    __syncthreads();

    // Issue all packed K reads before the softmax and P*V work.  They remain
    // in registers until publication into the alternate raw LDS tile.
    if constexpr (RawK) {
      if (has_next_k && (!StaggerK || wave < 4)) {
        const int t = tid & (tile_size - 1);
        const int group = tid / tile_size;
        const int meta = (pipe ^ 1) * tile_size + t;
        const int64_t ki = static_cast<int64_t>(physical[meta]) * k_stride0 +
                           kv_head * k_stride1 + group * k_stride2 +
                           page_offset[meta] * k_stride3;
        next_k0 = *reinterpret_cast<const NativeU32x4*>(key8 + ki);
        next_k1 =
            *reinterpret_cast<const NativeU32x4*>(key8 + ki + 4 * k_stride2);
        next_k2 =
            *reinterpret_cast<const NativeU32x4*>(key8 + ki + 8 * k_stride2);
        next_k3 =
            *reinterpret_cast<const NativeU32x4*>(key8 + ki + 12 * k_stride2);
      }
    }

    float alpha = 0.0f;
    if (wave < kQPerKV) {
      float score0 = score_shared[wave * tile_pitch + lane] * 0.0625f;
      float score1 = score_shared[wave * tile_pitch + lane + 32] * 0.0625f;
      if (tile + lane >= end) score0 = -INFINITY;
      if (tile + lane + 32 >= end) score1 = -INFINITY;
      const float tile_max = wave_max(fmaxf(score0, score1));
      const float new_max = fmaxf(running_max, tile_max);
      const float p0 = (tile + lane < end) ? expf(score0 - new_max) : 0.0f;
      const float p1 = (tile + lane + 32 < end) ? expf(score1 - new_max) : 0.0f;
      const float block_sum = wave_sum(p0 + p1);
      alpha = isinf(running_max) ? 0.0f : expf(running_max - new_max);
      probability[wave * tile_pitch + lane] = float_to_bf16(p0);
      probability[wave * tile_pitch + lane + 32] = float_to_bf16(p1);
      running_max = new_max;
      running_sum = running_sum * alpha + block_sum;
      if constexpr (ExactRows) {
        if (lane == 0) {
          row_alpha[wave] = alpha;
          row_inv_sum[wave] = 1.0f / (running_sum + 1.0e-10f);
        }
      }
    }
    if constexpr (RawK) {
      if (has_next_k) {
        const int t = tid & (tile_size - 1);
        const int group = tid / tile_size;
        auto* next_raw =
            raw_k + (TokenHalves ? 0 : (pipe ^ 1) * raw_k_tile_bytes);
        *reinterpret_cast<NativeU32x4*>(next_raw + (group * tile_size + t) *
                                                       pack) = next_k0;
        *reinterpret_cast<NativeU32x4*>(
            next_raw + ((group + 4) * tile_size + t) * pack) = next_k1;
        *reinterpret_cast<NativeU32x4*>(
            next_raw + ((group + 8) * tile_size + t) * pack) = next_k2;
        *reinterpret_cast<NativeU32x4*>(
            next_raw + ((group + 12) * tile_size + t) * pack) = next_k3;
      }
    }
    __syncthreads();

    if constexpr (!FP8) {
#pragma unroll
      for (int vec = 0; vec < vLoadsPerThread; ++vec) {
        const int d = tid;
        const int t0 = vec * values_per_vec;
        const bool valid = tile + t0 < end;
        v_packed[vec] = make_uint4(0, 0, 0, 0);
        if (valid) {
          const int64_t vi = static_cast<int64_t>(physical[t0]) * v_stride0 +
                             kv_head * v_stride1 + d * v_stride2 +
                             page_offset[t0] * v_stride3;
          v_packed[vec] = *reinterpret_cast<const uint4*>(value16 + vi);
        }
      }
    }

    if constexpr (!ExactRows) {
#pragma unroll
      for (int vec = 0; vec < vLoadsPerThread; ++vec) {
        const int d = tid;
        const int t0 = vec * values_per_vec;
        const uint4 packed = v_packed[vec];
#pragma unroll
        for (int j = 0; j < values_per_vec; ++j) {
          const int t = t0 + j;
          const bool element_valid = tile + t < end;
          if constexpr (FP8) {
            if ((j & 1) == 0) {
              const uint32_t word = j < 4    ? packed.x
                                    : j < 8  ? packed.y
                                    : j < 12 ? packed.z
                                             : packed.w;
              const auto converted =
                  (j & 2) ? fp8x2_scaled_to_bf16x2<true>(word, vs)
                          : fp8x2_scaled_to_bf16x2<false>(word, vs);
              hip_bfloat16 lo;
              hip_bfloat16 hi;
              lo.data = static_cast<uint16_t>(converted[0]);
              hi.data = static_cast<uint16_t>(converted[1]);
              v_shared[t * 257 + d] = element_valid ? lo : float_to_bf16(0.0f);
              v_shared[(t + 1) * 257 + d] =
                  (tile + t + 1 < end) ? hi : float_to_bf16(0.0f);
            }
          } else {
            const uint32_t word = j < 2   ? packed.x
                                  : j < 4 ? packed.y
                                  : j < 6 ? packed.z
                                          : packed.w;
            const uint16_t raw = static_cast<uint16_t>(word >> (16 * (j & 1)));
            hip_bfloat16 value;
            value.data = raw;
            v_shared[t * 257 + d] = element_valid ? value : float_to_bf16(0.0f);
          }
        }
      }
      __syncthreads();
    }

    WmmaC out_frag[2];
    rocwmma::fill_fragment(out_frag[0], 0.0f);
    rocwmma::fill_fragment(out_frag[1], 0.0f);
#pragma unroll
    for (int token = 0; token < tile_size; token += 16) {
      WmmaA p_frag;
      if constexpr (ExactRows) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int row = lane & 15;
          const int col = (lane >> 4) * 8 + i;
          p_frag[i] = row < kQPerKV
                          ? reinterpret_cast<const WmmaBF16*>(
                                probability)[row * tile_pitch + token + col]
                          : WmmaBF16(0.0f);
        }
      } else {
        rocwmma::load_matrix_sync(
            p_frag, reinterpret_cast<const WmmaBF16*>(probability) + token,
            static_cast<uint32_t>(tile_pitch));
      }
#pragma unroll
      for (int col = 0; col < 2; ++col) {
        WmmaB v_frag;
        if constexpr (ExactRows) {
          const int direct_idx = (token >> 4) * 2 + col;
          if constexpr (FP8) {
            const auto* v_pairs = reinterpret_cast<const uint2*>(v_packed);
            const uint2 packed = v_pairs[direct_idx];
            const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const uint32_t word = words[pair >> 1];
              const auto converted =
                  (pair & 1) ? fp8x2_scaled_to_bf16x2<true>(word, vs)
                             : fp8x2_scaled_to_bf16x2<false>(word, vs);
              hip_bfloat16 lo;
              hip_bfloat16 hi;
              lo.data = static_cast<uint16_t>(converted[0]);
              hi.data = static_cast<uint16_t>(converted[1]);
              v_frag[pair * 2] = WmmaBF16(lo);
              v_frag[pair * 2 + 1] = WmmaBF16(hi);
            }
          } else {
            v_frag.mStorage = v16_packed[direct_idx].mStorage;
          }
        } else {
          rocwmma::load_matrix_sync(
              v_frag,
              reinterpret_cast<const WmmaBF16*>(v_shared) + token * 257 +
                  wave * 32 + col * 16,
              static_cast<uint32_t>(257));
        }
        rocwmma::mma_sync(out_frag[col], p_frag, v_frag, out_frag[col]);
      }
    }
    if constexpr (ExactRows) {
      if (lane < 16) {
#pragma unroll
        for (int col = 0; col < 2; ++col) {
          const auto output_store =
              rocwmma::apply_data_layout<rocwmma::row_major>(out_frag[col]);
#pragma unroll
          for (int row = 0; row < kQPerKV; ++row) {
            output_acc[col][row] =
                output_acc[col][row] * row_alpha[row] + output_store[row];
          }
        }
      }
    } else if constexpr (ExactRows) {
      if (lane < 16) {
#pragma unroll
        for (int col = 0; col < 2; ++col) {
          const auto output_store =
              rocwmma::apply_data_layout<rocwmma::row_major>(out_frag[col]);
#pragma unroll
          for (int row = 0; row < kQPerKV; ++row) {
            output_shared[row * 257 + wave * 32 + col * 16 + lane] =
                output_store[row];
          }
        }
      }
    } else {
      rocwmma::store_matrix_sync(output_shared + wave * 32, out_frag[0], 257,
                                 rocwmma::layout_t::mem_row_major);
      rocwmma::store_matrix_sync(output_shared + wave * 32 + 16, out_frag[1],
                                 257, rocwmma::layout_t::mem_row_major);
    }
    if constexpr (!ExactRows) {
      __syncthreads();
    }

    if constexpr (!ExactRows) {
      if (wave < kQPerKV) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int d = lane + 32 * j;
          accum[j] = accum[j] * alpha + output_shared[wave * 257 + d];
        }
      }
    }
  }

  if constexpr (ExactRows) {
    const bool has_tokens = end > begin;
    if ((!TokenHalves || half == 0) && lane < 16) {
#pragma unroll
      for (int col = 0; col < 2; ++col) {
        const int d = wave * 32 + col * 16 + lane;
#pragma unroll
        for (int row = 0; row < kQPerKV; ++row) {
          const int qh = kv_head * kQPerKV + row;
          mid_out[seq * mo_stride0 + qh * mo_stride1 + split * mo_stride2 + d] =
              has_tokens ? output_acc[col][row] * row_inv_sum[row] : 0.0f;
        }
      }
    }
    if ((!TokenHalves || half == 0) && wave < kQPerKV && lane == 0) {
      const int qh = kv_head * kQPerKV + wave;
      mid_lse[seq * ml_stride0 + qh * ml_stride1 + split * ml_stride2] =
          has_tokens ? running_max + logf(running_sum) : -INFINITY;
    }
    if constexpr (TokenHalves) {
      // A cooperative launch makes all 64 workgroups resident.  The grid
      // barrier is both the progress guarantee and the device-wide release /
      // acquire point for the first-half scratch publication.
      cooperative_groups::this_grid().sync();
      if (half == 1) {
        if (wave < kQPerKV && lane == 0) {
          const int qh = kv_head * kQPerKV + wave;
          const int64_t lse_index =
              seq * ml_stride0 + qh * ml_stride1 + split * ml_stride2;
          const float first_lse = mid_lse[lse_index];
          const float own_lse =
              has_tokens ? running_max + logf(running_sum) : -INFINITY;
          const float combined_max = fmaxf(first_lse, own_lse);
          const float first_scale =
              isinf(first_lse) ? 0.0f : expf(first_lse - combined_max);
          const float second_scale =
              isinf(own_lse) ? 0.0f : expf(own_lse - combined_max);
          const float scale_sum = first_scale + second_scale;
          const float inv_scale_sum = 1.0f / (scale_sum + 1.0e-10f);
          first_weight[wave] = first_scale * inv_scale_sum;
          second_weight[wave] = second_scale * inv_scale_sum;
          mid_lse[lse_index] =
              scale_sum > 0.0f ? combined_max + logf(scale_sum) : -INFINITY;
        }
        __syncthreads();
        if (lane < 16) {
#pragma unroll
          for (int col = 0; col < 2; ++col) {
            const int d = wave * 32 + col * 16 + lane;
#pragma unroll
            for (int row = 0; row < kQPerKV; ++row) {
              const int qh = kv_head * kQPerKV + row;
              const int64_t out_index =
                  seq * mo_stride0 + qh * mo_stride1 + split * mo_stride2 + d;
              const float own =
                  has_tokens ? output_acc[col][row] * row_inv_sum[row] : 0.0f;
              mid_out[out_index] = mid_out[out_index] * first_weight[row] +
                                   own * second_weight[row];
            }
          }
        }
      }
    }
  } else if (wave < kQPerKV) {
    const int qh = kv_head * kQPerKV + wave;
    const bool has_tokens = end > begin;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = lane + 32 * j;
      mid_out[seq * mo_stride0 + qh * mo_stride1 + split * mo_stride2 + d] =
          has_tokens ? accum[j] / (running_sum + 1.0e-10f) : 0.0f;
    }
    if (lane == 0) {
      mid_lse[seq * ml_stride0 + qh * ml_stride1 + split * ml_stride2] =
          has_tokens ? running_max + logf(running_sum) : -INFINITY;
    }
  }
}

__global__ __launch_bounds__(32) void splitkv_stage1_fp8_wave(
    const hip_bfloat16* __restrict__ query,
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float* __restrict__ mid_out, float* __restrict__ mid_lse, int batch,
    int splits, int table_stride, int64_t q_stride0, int64_t q_stride1,
    int64_t k_stride0, int64_t k_stride1, int64_t k_stride2, int64_t k_stride3,
    int64_t k_stride4, int64_t v_stride0, int64_t v_stride1, int64_t v_stride2,
    int64_t v_stride3, int64_t mo_stride0, int64_t mo_stride1,
    int64_t mo_stride2, int64_t ml_stride0, int64_t ml_stride1,
    int64_t ml_stride2) {
  const int lane = threadIdx.x;
  const int split = blockIdx.x % splits;
  const int head_item = blockIdx.x / splits;
  const int seq = head_item / 12;
  const int qh = head_item % 12;
  if (seq >= batch) return;
  const int kv_head = qh / kQPerKV;

  __shared__ uint8_t value_tile[kHead][kPitch];
  __shared__ hip_bfloat16 probability[kTile];

  const int length = seq_lens[seq];
  const int split_len =
      ((length + splits - 1) / splits + kTile - 1) / kTile * kTile;
  const int begin = split * split_len;
  const int end = min(begin + split_len, length);
  const float ks = k_scale[0];
  const float vs = v_scale[0];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile = begin; tile < end; tile += kTile) {
    const int token = tile + lane;
    const bool valid = token < end;
    int pb = 0;
    int in_page = 0;
    if (valid) {
      pb = block_tables[seq * table_stride + token / kPage];
      in_page = token % kPage;
    }

    float score = 0.0f;
#pragma unroll 8
    for (int d = 0; d < kHead; ++d) {
      const int64_t ki = static_cast<int64_t>(pb) * k_stride0 +
                         kv_head * k_stride1 + (d >> 4) * k_stride2 +
                         in_page * k_stride3 + (d & 15) * k_stride4;
      const float kval = valid ? bf16_to_float(float_to_bf16(
                                     fp8e4m3fn_to_float(key_cache[ki]) * ks))
                               : 0.0f;
      score = fmaf(bf16_to_float(query[seq * q_stride0 + qh * q_stride1 + d]),
                   kval, score);
    }
    score = valid ? score * 0.0625f : -INFINITY;
    const float tile_max = wave_max(score);
    const float new_max = fmaxf(running_max, tile_max);
    const float p = valid ? expf(score - new_max) : 0.0f;
    const float block_sum = wave_sum(p);
    const float alpha = isinf(running_max) ? 0.0f : expf(running_max - new_max);
    probability[lane] = float_to_bf16(p);
    running_max = new_max;
    running_sum = running_sum * alpha + block_sum;

#pragma unroll 4
    for (int d = 0; d < kHead; ++d) {
      const int64_t vi = static_cast<int64_t>(pb) * v_stride0 +
                         kv_head * v_stride1 + d * v_stride2 +
                         in_page * v_stride3;
      value_tile[d][lane] = valid ? value_cache[vi] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = lane + 32 * j;
      float partial = 0.0f;
#pragma unroll
      for (int t = 0; t < kTile; ++t) {
        const float value = bf16_to_float(
            float_to_bf16(fp8e4m3fn_to_float(value_tile[d][t]) * vs));
        partial = fmaf(bf16_to_float(probability[t]), value, partial);
      }
      accum[j] = accum[j] * alpha + partial;
    }
    __syncthreads();
  }

  const bool has_tokens = end > begin;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = lane + 32 * j;
    mid_out[seq * mo_stride0 + qh * mo_stride1 + split * mo_stride2 + d] =
        has_tokens ? accum[j] / (running_sum + 1.0e-10f) : 0.0f;
  }
  if (lane == 0) {
    mid_lse[seq * ml_stride0 + qh * ml_stride1 + split * ml_stride2] =
        has_tokens ? running_max + logf(running_sum) : -INFINITY;
  }
}

template <int Splits, int DimsPerBlock, int HeadsPerBlock>
__global__ __launch_bounds__(32 * HeadsPerBlock) void splitkv_reduce(
    const float* __restrict__ mid_out, const float* __restrict__ mid_lse,
    const int* __restrict__ query_start_loc, hip_bfloat16* __restrict__ output,
    int batch, int64_t out_stride0, int64_t out_stride1, int64_t mo_stride0,
    int64_t mo_stride1, int64_t mo_stride2, int64_t ml_stride0,
    int64_t ml_stride1, int64_t ml_stride2) {
  constexpr int kDimParts = kHead / DimsPerBlock;
  constexpr int kValuesPerLane = DimsPerBlock / 32;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int linear = blockIdx.x * HeadsPerBlock + wave;
  const int item = linear / kDimParts;
  const int dim_part = linear % kDimParts;
  const int seq = item / 12;
  const int qh = item % 12;
  if (linear >= batch * 12 * kDimParts) return;
  const int query_row = query_start_loc[seq];
  if (query_start_loc[seq + 1] - query_row != 1) return;

  if constexpr (Splits == 1) {
#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      const int d = dim_part * DimsPerBlock + lane + 32 * j;
      output[query_row * out_stride0 + qh * out_stride1 + d] =
          float_to_bf16(mid_out[seq * mo_stride0 + qh * mo_stride1 + d]);
    }
    return;
  }

  float lse = -INFINITY;
  if (lane < Splits) {
    lse = mid_lse[seq * ml_stride0 + qh * ml_stride1 + lane * ml_stride2];
  }
  const float maximum = wave_max(lse);
  float weight = (lane < Splits && !isinf(lse)) ? expf(lse - maximum) : 0.0f;
  const float inv_sum = 1.0f / (wave_sum(weight) + 1.0e-10f);
  weight *= inv_sum;

#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    const int d = dim_part * DimsPerBlock + lane + 32 * j;
    float value = 0.0f;
#pragma unroll
    for (int s = 0; s < Splits; ++s) {
      value =
          fmaf(__shfl(weight, s, 32),
               mid_out[seq * mo_stride0 + qh * mo_stride1 + s * mo_stride2 + d],
               value);
    }
    output[query_row * out_stride0 + qh * out_stride1 + d] =
        float_to_bf16(value);
  }
}

template <int Splits, int DimsPerBlock, int HeadsPerBlock>
void launch_splitkv_reduce(const float* mid_out, const float* mid_lse,
                           const int* query_start_loc, hip_bfloat16* output,
                           int batch, int64_t out_stride0, int64_t out_stride1,
                           int64_t mo_stride0, int64_t mo_stride1,
                           int64_t mo_stride2, int64_t ml_stride0,
                           int64_t ml_stride1, int64_t ml_stride2,
                           hipStream_t stream) {
  constexpr int kDimParts = kHead / DimsPerBlock;
  constexpr int kItemsPerBlock = HeadsPerBlock;
  splitkv_reduce<Splits, DimsPerBlock, HeadsPerBlock>
      <<<dim3((batch * 12 * kDimParts + kItemsPerBlock - 1) / kItemsPerBlock),
         dim3(32 * HeadsPerBlock), 0, stream>>>(
          mid_out, mid_lse, query_start_loc, output, batch, out_stride0,
          out_stride1, mo_stride0, mo_stride1, mo_stride2, ml_stride0,
          ml_stride1, ml_stride2);
}

}  // namespace

void rdna4_splitkv_paged_attention(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens, const torch::Tensor& query_start_loc,
    const torch::Tensor& k_scale, const torch::Tensor& v_scale,
    torch::Tensor& output, torch::Tensor& mid_out, torch::Tensor& mid_lse,
    int64_t splits, bool token_halves) {
  TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda() &&
                  block_tables.is_cuda() && seq_lens.is_cuda() &&
                  query_start_loc.is_cuda() && k_scale.is_cuda() &&
                  v_scale.is_cuda() && output.is_cuda() && mid_out.is_cuda() &&
                  mid_lse.is_cuda(),
              "RDNA4 SplitKV inputs must be on the GPU");
  TORCH_CHECK(query.device() == key_cache.device() &&
                  query.device() == value_cache.device() &&
                  query.device() == block_tables.device() &&
                  query.device() == seq_lens.device() &&
                  query.device() == query_start_loc.device() &&
                  query.device() == k_scale.device() &&
                  query.device() == v_scale.device() &&
                  query.device() == output.device() &&
                  query.device() == mid_out.device() &&
                  query.device() == mid_lse.device(),
              "RDNA4 SplitKV inputs must share a device");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 &&
                  output.scalar_type() == at::kBFloat16,
              "RDNA4 SplitKV query and output must use bfloat16");
  const bool fp8 = key_cache.scalar_type() == at::kFloat8_e4m3fn;
  TORCH_CHECK(fp8 || key_cache.scalar_type() == at::kBFloat16,
              "RDNA4 SplitKV cache must use bfloat16 or float8_e4m3fn");
  TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
              "RDNA4 SplitKV key/value cache dtypes must match");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt &&
                  seq_lens.scalar_type() == at::kInt &&
                  query_start_loc.scalar_type() == at::kInt,
              "RDNA4 SplitKV metadata must use int32");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat &&
                  v_scale.scalar_type() == at::kFloat && k_scale.numel() == 1 &&
                  v_scale.numel() == 1,
              "RDNA4 SplitKV scales must be scalar float32 tensors");
  TORCH_CHECK(mid_out.scalar_type() == at::kFloat &&
                  mid_lse.scalar_type() == at::kFloat,
              "RDNA4 SplitKV scratch must use float32");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == kKVHeads * kQPerKV &&
                  query.size(2) == kHead && output.sizes() == query.sizes(),
              "RDNA4 SplitKV requires query/output [tokens, 12, 256]");
  const int cache_groups = fp8 ? 16 : 32;
  const int cache_pack = fp8 ? 16 : 8;
  TORCH_CHECK(key_cache.dim() == 5 && key_cache.size(1) == kKVHeads &&
                  key_cache.size(2) == cache_groups &&
                  key_cache.size(3) == kPage && key_cache.size(4) == cache_pack,
              "RDNA4 SplitKV received an unsupported key-cache shape");
  TORCH_CHECK(value_cache.dim() == 4 &&
                  value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == kKVHeads &&
                  value_cache.size(2) == kHead && value_cache.size(3) == kPage,
              "RDNA4 SplitKV received an unsupported value-cache shape");
  const int batch = static_cast<int>(seq_lens.size(0));
  TORCH_CHECK(batch > 0 && block_tables.dim() == 2 &&
                  block_tables.size(0) == batch && seq_lens.dim() == 1 &&
                  query_start_loc.dim() == 1 &&
                  query_start_loc.size(0) == batch + 1,
              "RDNA4 SplitKV metadata shapes are inconsistent");
  TORCH_CHECK(mid_out.dim() == 4 && mid_out.size(0) >= batch &&
                  mid_out.size(1) == kKVHeads * kQPerKV &&
                  mid_out.size(2) >= splits && mid_out.size(3) == kHead &&
                  mid_lse.dim() == 3 && mid_lse.size(0) >= batch &&
                  mid_lse.size(1) == kKVHeads * kQPerKV &&
                  mid_lse.size(2) >= splits,
              "RDNA4 SplitKV scratch is too small");
  TORCH_CHECK(
      splits == 1 || splits == 2 || splits == 4 || splits == 8 || splits == 16,
      "RDNA4 SplitKV supports 1, 2, 4, 8, or 16 splits");
  TORCH_CHECK(!token_halves || (fp8 && batch == 1 && splits == 16),
              "RDNA4 SplitKV token-halves requires batch=1 FP8 and 16 splits");
  TORCH_CHECK(query.stride(2) == 1 && output.stride(2) == 1 &&
                  key_cache.stride(4) == 1 && value_cache.stride(3) == 1 &&
                  mid_out.stride(3) == 1 && block_tables.stride(1) == 1 &&
                  seq_lens.is_contiguous() && query_start_loc.is_contiguous(),
              "RDNA4 SplitKV received an unsupported inner layout");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const std::string arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  TORCH_CHECK(arch.find("gfx1200") != std::string::npos ||
                  arch.find("gfx1201") != std::string::npos,
              "RDNA4 SplitKV requires gfx1200 or gfx1201, got ", arch);
  hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((token_halves ? 2 : 1) * batch * kKVHeads *
            static_cast<int>(splits));
  dim3 block(256);

  if (fp8) {
    if (batch == 1) {
      if (token_halves) {
        const auto* q = reinterpret_cast<const hip_bfloat16*>(query.data_ptr());
        const void* k = key_cache.data_ptr();
        const void* v = value_cache.data_ptr();
        const int* tables = block_tables.data_ptr<int>();
        const int* lengths = seq_lens.data_ptr<int>();
        const int* query_starts = query_start_loc.data_ptr<int>();
        const float* ks = k_scale.data_ptr<float>();
        const float* vs = v_scale.data_ptr<float>();
        float* mo = mid_out.data_ptr<float>();
        float* ml = mid_lse.data_ptr<float>();
        int launch_batch = batch;
        int launch_splits = static_cast<int>(splits);
        int table_stride = block_tables.stride(0);
        int64_t q_stride0 = query.stride(0);
        int64_t q_stride1 = query.stride(1);
        int64_t k_stride0 = key_cache.stride(0);
        int64_t k_stride1 = key_cache.stride(1);
        int64_t k_stride2 = key_cache.stride(2);
        int64_t k_stride3 = key_cache.stride(3);
        int64_t k_stride4 = key_cache.stride(4);
        int64_t v_stride0 = value_cache.stride(0);
        int64_t v_stride1 = value_cache.stride(1);
        int64_t v_stride2 = value_cache.stride(2);
        int64_t v_stride3 = value_cache.stride(3);
        int64_t mo_stride0 = mid_out.stride(0);
        int64_t mo_stride1 = mid_out.stride(1);
        int64_t mo_stride2 = mid_out.stride(2);
        int64_t ml_stride0 = mid_lse.stride(0);
        int64_t ml_stride1 = mid_lse.stride(1);
        int64_t ml_stride2 = mid_lse.stride(2);
        void* args[] = {&q,
                        &k,
                        &v,
                        &tables,
                        &lengths,
                        &query_starts,
                        &ks,
                        &vs,
                        &mo,
                        &ml,
                        &launch_batch,
                        &launch_splits,
                        &table_stride,
                        &q_stride0,
                        &q_stride1,
                        &k_stride0,
                        &k_stride1,
                        &k_stride2,
                        &k_stride3,
                        &k_stride4,
                        &v_stride0,
                        &v_stride1,
                        &v_stride2,
                        &v_stride3,
                        &mo_stride0,
                        &mo_stride1,
                        &mo_stride2,
                        &ml_stride0,
                        &ml_stride1,
                        &ml_stride2};
        const hipError_t status =
            hipLaunchCooperativeKernel(splitkv_stage1<true, true, true, true>,
                                       grid, block, args, 0, stream);
        TORCH_CHECK(status == hipSuccess,
                    "B=1 token-half cooperative launch failed: ",
                    hipGetErrorString(status));
      } else {
        splitkv_stage1<true, true, true><<<grid, block, 0, stream>>>(
            reinterpret_cast<const hip_bfloat16*>(query.data_ptr()),
            key_cache.data_ptr(), value_cache.data_ptr(),
            block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
            query_start_loc.data_ptr<int>(), k_scale.data_ptr<float>(),
            v_scale.data_ptr<float>(), mid_out.data_ptr<float>(),
            mid_lse.data_ptr<float>(), batch, static_cast<int>(splits),
            block_tables.stride(0), query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),
            value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2));
      }
    } else {
      splitkv_stage1<true, true, false><<<grid, block, 0, stream>>>(
          reinterpret_cast<const hip_bfloat16*>(query.data_ptr()),
          key_cache.data_ptr(), value_cache.data_ptr(),
          block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
          query_start_loc.data_ptr<int>(), k_scale.data_ptr<float>(),
          v_scale.data_ptr<float>(), mid_out.data_ptr<float>(),
          mid_lse.data_ptr<float>(), batch, static_cast<int>(splits),
          block_tables.stride(0), query.stride(0), query.stride(1),
          key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
          key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),
          value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
          mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
          mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2));
    }
  } else {
    splitkv_stage1<false, true, false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const hip_bfloat16*>(query.data_ptr()),
        key_cache.data_ptr(), value_cache.data_ptr(),
        block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(), mid_out.data_ptr<float>(),
        mid_lse.data_ptr<float>(), batch, static_cast<int>(splits),
        block_tables.stride(0), query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),
        value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2));
  }
  auto* reduced = reinterpret_cast<hip_bfloat16*>(output.data_ptr());
  if (splits == 16) {
    if (batch == 1) {
      launch_splitkv_reduce<16, 64, 1>(
          mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
          query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
          output.stride(1), mid_out.stride(0), mid_out.stride(1),
          mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
          mid_lse.stride(2), stream);
    } else {
      launch_splitkv_reduce<16, 256, 1>(
          mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
          query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
          output.stride(1), mid_out.stride(0), mid_out.stride(1),
          mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
          mid_lse.stride(2), stream);
    }
  } else if (splits == 8) {
    launch_splitkv_reduce<8, 256, 4>(
        mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
        output.stride(1), mid_out.stride(0), mid_out.stride(1),
        mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        mid_lse.stride(2), stream);
  } else if (splits == 4) {
    launch_splitkv_reduce<4, 256, 4>(
        mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
        output.stride(1), mid_out.stride(0), mid_out.stride(1),
        mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        mid_lse.stride(2), stream);
  } else if (splits == 2) {
    launch_splitkv_reduce<2, 256, 4>(
        mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
        output.stride(1), mid_out.stride(0), mid_out.stride(1),
        mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        mid_lse.stride(2), stream);
  } else {
    launch_splitkv_reduce<1, 256, 4>(
        mid_out.data_ptr<float>(), mid_lse.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), reduced, batch, output.stride(0),
        output.stride(1), mid_out.stride(0), mid_out.stride(1),
        mid_out.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        mid_lse.stride(2), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
