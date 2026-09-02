#!/usr/bin/env python3
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RDNA4 block-scaled FP8 GEMM for vLLM decode and small prefill shapes.

This is a raw-weight kernel: ``weight`` is the ordinary row-major ``[N, K]``
tensor consumed by vLLM.  No preshuffle or persistent workspace is part of the
interface.  The numerical contract is intentionally identical to
``rdna4_fp8_block_scaled_mm_decode`` in the vLLM fork::

    out[m, n] = sum_kb(
        dot_fp32(a[m, kb * 128 : (kb + 1) * 128], weight[n, kb * 128 : (kb + 1) * 128])
        * a_scale[m, kb]
        * weight_scale[n // 128, kb]
    )

Each K=128 dot product is completed in FP32 before its two scales are applied.
The scaled blocks are accumulated in FP32 and converted to BF16 once.
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gfx12_sync import lds_fence_signal, lds_fence_wait
from .runtime import run_compiled

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
SCALE_K = 128
WAVE_SIZE = 32


class KernelRoute(str, Enum):
    """The micro-routes retained from the tuned native HIP implementation."""

    DECODE_SPLITK_M2 = "decode_splitk_m2"
    DECODE_SPLITK_M1 = "decode_splitk_m1"
    DECODE_PACKED_M4_N64 = "decode_packed_m4_n64"
    PACKED_M16 = "packed_m16"
    TILED_M16 = "tiled_m16"
    PAIRED_N_M64 = "paired_n_m64"
    PAIRED_N_M32 = "paired_n_m32"


@dataclass(frozen=True)
class KernelConfig:
    route: KernelRoute
    block_n: int
    row_tiles: int
    threads: int
    split_k: int = 1


def select_kernel_config(m: int, n: int, k: int) -> KernelConfig:
    """Select the same M1--M64 micro-route as the current vLLM HIP kernel."""
    if not 1 <= m <= 64:
        raise ValueError(f"RDNA4 small-M route requires 1 <= M <= 64, got {m}")
    if n <= 0 or n % 128:
        raise ValueError(
            f"RDNA4 small-M route requires positive N divisible by 128, got {n}"
        )
    if k <= 0 or k % SCALE_K:
        raise ValueError(
            f"RDNA4 small-M route requires positive K divisible by 128, got {k}"
        )

    n_groups = n // 128
    if m == 2 and n_groups <= 64:
        return KernelConfig(
            KernelRoute.DECODE_SPLITK_M2, block_n=64, row_tiles=1, threads=64, split_k=2
        )
    if m == 1:
        if n >= 16384:
            return KernelConfig(
                KernelRoute.PACKED_M16, block_n=64, row_tiles=1, threads=64
            )
        return KernelConfig(
            KernelRoute.DECODE_SPLITK_M1,
            block_n=128,
            row_tiles=1,
            threads=128,
            split_k=2,
        )
    if m == 4 and n_groups <= 64:
        return KernelConfig(
            KernelRoute.DECODE_PACKED_M4_N64, block_n=64, row_tiles=1, threads=64
        )
    if m <= 16:
        return KernelConfig(
            KernelRoute.PACKED_M16, block_n=128, row_tiles=1, threads=128
        )
    if n <= 4096 or (n >= 16384 and 39 <= m <= 48 and k >= 8192):
        return KernelConfig(
            KernelRoute.TILED_M16, block_n=128, row_tiles=1, threads=128
        )
    if n >= 16384 and 33 <= m <= 38:
        return KernelConfig(
            KernelRoute.PAIRED_N_M64, block_n=128, row_tiles=4, threads=128
        )
    return KernelConfig(KernelRoute.PAIRED_N_M32, block_n=128, row_tiles=2, threads=128)


def _make_buffer(tensor, elem_ty, width: int, num_records_bytes, *, bounds_check=False):
    """Build a flat raw-buffer view addressable by an element offset.

    The first mode is the vector width and the second has unit stride, so an
    index names the first element rather than a width-sized record. This keeps
    padded weight rows correct even when their stride is not a multiple of the
    eight-element FP8 load width.
    """
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(
        view,
        max_size=False,
        num_records_bytes=num_records_bytes,
        bounds_check=bounds_check,
    )


def _load_f32(buffer, index):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    fragment = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)
    return Vec(fragment.load())[0]


def _load_fp8_fragment_buffer(buffer, index, fragment):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float8E4M3FN)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)


def _load_fp8_fragment_ptr(ptr, index, fragment):
    packed = fx.ptr_load(ptr + index, result_type=fx.Vector.make_type(8, fx.Uint8))
    fragment.store(Vec(packed.bitcast(fx.Float8E4M3FN)))


def _store_bf16(buffer, index, value):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    fragment = fx.make_rmem_tensor(1, fx.BFloat16)
    fragment.store(Vec.from_elements([value], fx.BFloat16))
    fx.copy(atom, fragment, fx.slice(buffer, (None, index)))


def _f32_to_bf16_rne(value):
    """Round FP32 to BF16 without gfx120x NaN-classification lowering."""
    bits = fx.Float32(value).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> fx.Uint32(16)) & fx.Uint32(1))
    return fx.Uint16(rounded >> fx.Uint32(16)).bitcast(fx.BFloat16)


def _create_packed_module(m: int, n: int, k: int, stride_b: int, config: KernelConfig):
    """Create the shared packed-row implementation for all non-split-K routes."""
    assert config.split_k == 1
    assert config.block_n == config.threads
    assert config.block_n in (64, 128)
    assert config.row_tiles in (1, 2, 4)

    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    scale_blocks = k // SCALE_K
    grid_m = (
        1
        if m <= 16
        else (m + config.row_tiles * WMMA_M - 1) // (config.row_tiles * WMMA_M)
    )

    @flyc.kernel
    def packed_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(WAVE_SIZE)
        wave = tid // fx.Int32(WAVE_SIZE)
        lane_col = lane % fx.Int32(WMMA_N)
        lane_row_base = (lane // fx.Int32(WMMA_N)) * fx.Int32(8)
        block_row = fx.block_idx.y * fx.Int32(config.row_tiles * WMMA_M)
        n0 = fx.block_idx.x * fx.Int32(config.block_n) + wave * fx.Int32(2 * WMMA_N)

        a_buf = _make_buffer(arg_a, fp8, 8, m * k)
        b_buf = _make_buffer(arg_b, fp8, 8, n * stride_b)
        as_buf = _make_buffer(arg_as, f32, 1, m * scale_blocks * 4)
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_buf = _make_buffer(arg_out, bf16, 1, m * n * 2)

        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [
            [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
            for _ in range_constexpr(config.row_tiles)
        ]
        for rt in range_constexpr(config.row_tiles):
            for j in range_constexpr(2):
                totals[rt][j].fill(0)

        for kb in range(0, scale_blocks, 1):
            partials = [
                [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
                for _ in range_constexpr(config.row_tiles)
            ]
            for rt in range_constexpr(config.row_tiles):
                for j in range_constexpr(2):
                    partials[rt][j].fill(0)

            if const_expr(m == 1 and n >= 16384):
                for ks_group in range_constexpr(2):
                    a_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)]
                    b_frags = [
                        [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                        for _ in range_constexpr(4)
                    ]
                    for ks_local in range_constexpr(4):
                        ks = ks_group * 4 + ks_local
                        k_lane = (
                            kb * fx.Int32(SCALE_K)
                            + fx.Int32(ks * WMMA_K)
                            + (lane // fx.Int32(16)) * fx.Int32(8)
                        )
                        for j in range_constexpr(2):
                            b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                            _load_fp8_fragment_buffer(
                                b_buf,
                                b_row * fx.Int32(stride_b) + k_lane,
                                b_frags[ks_local][j],
                            )
                        a_frags[ks_local].fill(0)
                        global_row = block_row + (lane % fx.Int32(WMMA_M))
                        if global_row < fx.Int32(m):
                            _load_fp8_fragment_buffer(
                                a_buf,
                                global_row * fx.Int32(k) + k_lane,
                                a_frags[ks_local],
                            )
                    for ks_local in range_constexpr(4):
                        for j in range_constexpr(2):
                            fx.gemm(
                                mma,
                                partials[0][j],
                                a_frags[ks_local],
                                b_frags[ks_local][j],
                                partials[0][j],
                            )
                    fx.rocdl.sched_vmem(12)
                    fx.rocdl.sched_mfma(8)
                    fx.rocdl.sched_barrier(0)
            else:
                for ks in range_constexpr(SCALE_K // WMMA_K):
                    b_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                    k_lane = (
                        kb * fx.Int32(SCALE_K)
                        + fx.Int32(ks * WMMA_K)
                        + (lane // fx.Int32(16)) * fx.Int32(8)
                    )
                    for j in range_constexpr(2):
                        b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                        _load_fp8_fragment_buffer(
                            b_buf,
                            b_row * fx.Int32(stride_b) + k_lane,
                            b_frags[j],
                        )

                    for rt in range_constexpr(config.row_tiles):
                        a_frag = fx.make_rmem_tensor(8, fp8)
                        a_frag.fill(0)
                        local_row = fx.Int32(rt * WMMA_M) + (lane % fx.Int32(WMMA_M))
                        global_row = block_row + local_row
                        if global_row < fx.Int32(m):
                            _load_fp8_fragment_buffer(
                                a_buf,
                                global_row * fx.Int32(k) + k_lane,
                                a_frag,
                            )
                        for j in range_constexpr(2):
                            fx.gemm(
                                mma,
                                partials[rt][j],
                                a_frag,
                                b_frags[j],
                                partials[rt][j],
                            )

            b_group = n0 // fx.Int32(SCALE_K)
            b_scale = f32(fx.ptr_load(bs_ptr + b_group * fx.Int32(scale_blocks) + kb))
            for rt in range_constexpr(config.row_tiles):
                if const_expr(m == 1):
                    a_scale = _load_f32(as_buf, kb)
                    scales = [
                        (lane < fx.Int32(16)).select(a_scale * b_scale, f32(0.0))
                        if value_idx == 0
                        else f32(0.0)
                        for value_idx in range_constexpr(8)
                    ]
                else:
                    scales = []
                    for value_idx in range_constexpr(8):
                        global_row = (
                            block_row
                            + fx.Int32(rt * WMMA_M)
                            + lane_row_base
                            + fx.Int32(value_idx)
                        )
                        valid = global_row < fx.Int32(m)
                        safe_row = valid.select(global_row, fx.Int32(0))
                        a_scale = _load_f32(
                            as_buf, safe_row * fx.Int32(scale_blocks) + kb
                        )
                        scales.append(valid.select(a_scale * b_scale, f32(0.0)))
                for j in range_constexpr(2):
                    total_v = Vec(totals[rt][j].load())
                    partial_v = Vec(partials[rt][j].load())
                    updated = [
                        total_v[x] + partial_v[x] * scales[x]
                        for x in range_constexpr(8)
                    ]
                    totals[rt][j].store(Vec.from_elements(updated, f32))

        if const_expr(m == 1):
            if lane < fx.Int32(16):
                for j in range_constexpr(2):
                    col = n0 + fx.Int32(j * WMMA_N) + lane_col
                    value = _f32_to_bf16_rne(Vec(totals[0][j].load())[0])
                    _store_bf16(out_buf, col, value)
        else:
            for rt in range_constexpr(config.row_tiles):
                for value_idx in range_constexpr(8):
                    global_row = (
                        block_row
                        + fx.Int32(rt * WMMA_M)
                        + lane_row_base
                        + fx.Int32(value_idx)
                    )
                    if global_row < fx.Int32(m):
                        for j in range_constexpr(2):
                            col = n0 + fx.Int32(j * WMMA_N) + lane_col
                            value = _f32_to_bf16_rne(
                                Vec(totals[rt][j].load())[value_idx]
                            )
                            _store_bf16(out_buf, global_row * fx.Int32(n) + col, value)

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        packed_kernel(arg_a, arg_b, arg_as, arg_bs, arg_out).launch(
            grid=(n // config.block_n, grid_m, 1),
            block=(config.threads, 1, 1),
            stream=stream,
        )

    return launch


def _create_m4_n64_module(n: int, k: int, stride_b: int, config: KernelConfig):
    """Create the cache-resident two-wave M4 specialization."""
    assert config.route is KernelRoute.DECODE_PACKED_M4_N64
    assert (config.block_n, config.threads, config.split_k) == (64, 64, 1)

    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    scale_blocks = k // SCALE_K

    @flyc.kernel
    def m4_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE_SIZE)
        lane = tid % fx.Int32(WAVE_SIZE)
        lane_col = lane % fx.Int32(WMMA_N)
        n0 = fx.block_idx.x * fx.Int32(64) + wave * fx.Int32(32)

        a_ptr = fx.recast_iter(fx.Uint8, fx.get_iter(arg_a))
        b_ptr = fx.recast_iter(fx.Uint8, fx.get_iter(arg_b))
        as_ptr = fx.recast_iter(f32, fx.get_iter(arg_as))
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_buf = _make_buffer(arg_out, bf16, 1, 4 * n * 2)

        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
        for j in range_constexpr(2):
            totals[j].fill(0)

        for kb in range(0, scale_blocks, 1):
            partials = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(2)]
            for j in range_constexpr(2):
                partials[j].fill(0)

            # Match the HIP specialization's two-slice load batch: expose all
            # six 64-bit loads before issuing the four dependent WMMAs.
            for ks_pair in range_constexpr(SCALE_K // (2 * WMMA_K)):
                k_lane = (
                    kb * fx.Int32(SCALE_K)
                    + fx.Int32(ks_pair * 2 * WMMA_K)
                    + (lane // fx.Int32(16)) * fx.Int32(8)
                )
                a_frags = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                for ki in range_constexpr(2):
                    a_frags[ki].fill(0)
                a_row = lane % fx.Int32(WMMA_M)
                if a_row < fx.Int32(4):
                    for ki in range_constexpr(2):
                        _load_fp8_fragment_ptr(
                            a_ptr,
                            a_row * fx.Int32(k) + k_lane + fx.Int32(ki * WMMA_K),
                            a_frags[ki],
                        )

                b_frags = [
                    [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(2)]
                    for _ in range_constexpr(2)
                ]
                for j in range_constexpr(2):
                    b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                    for ki in range_constexpr(2):
                        _load_fp8_fragment_ptr(
                            b_ptr,
                            b_row * fx.Int32(stride_b) + k_lane + fx.Int32(ki * WMMA_K),
                            b_frags[ki][j],
                        )
                for ki in range_constexpr(2):
                    for j in range_constexpr(2):
                        fx.gemm(
                            mma, partials[j], a_frags[ki], b_frags[ki][j], partials[j]
                        )

            b_group = n0 // fx.Int32(SCALE_K)
            b_scale = f32(fx.ptr_load(bs_ptr + b_group * fx.Int32(scale_blocks) + kb))
            scales = [
                f32(fx.ptr_load(as_ptr + fx.Int32(row * scale_blocks) + kb)) * b_scale
                for row in range_constexpr(4)
            ]
            for j in range_constexpr(2):
                total_v = Vec(totals[j].load())
                partial_v = Vec(partials[j].load())
                totals[j].store(
                    Vec.from_elements(
                        [
                            total_v[x] + partial_v[x] * scales[x]
                            if x < 4
                            else total_v[x]
                            for x in range_constexpr(8)
                        ],
                        f32,
                    )
                )

        if lane < fx.Int32(16):
            for row in range_constexpr(4):
                for j in range_constexpr(2):
                    col = n0 + fx.Int32(j * WMMA_N) + lane_col
                    value = _f32_to_bf16_rne(Vec(totals[j].load())[row])
                    _store_bf16(out_buf, fx.Int32(row * n) + col, value)

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        m4_kernel(arg_a, arg_b, arg_as, arg_bs, arg_out).launch(
            grid=(n // 64, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def _create_splitk_module(m: int, n: int, k: int, stride_b: int, config: KernelConfig):
    """Create the M1/M2 two-way K-split decode kernels."""
    assert m in (1, 2)
    assert config.split_k == 2
    assert (m, config.block_n, config.threads) in ((1, 128, 128), (2, 64, 64))

    fp8 = fx.Float8E4M3FN
    f32 = fx.Float32
    bf16 = fx.BFloat16
    scale_blocks = k // SCALE_K
    half_blocks = (scale_blocks + 1) // 2
    waves = config.threads // WAVE_SIZE
    shared_elems = waves * 4 * m * WMMA_N

    @fx.struct
    class SharedStorage:
        partial: fx.Array[f32, shared_elems, 16]

    @flyc.kernel
    def splitk_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE_SIZE)
        lane = tid % fx.Int32(WAVE_SIZE)
        lane_col = lane % fx.Int32(WMMA_N)
        half_id = (wave % fx.Int32(2)) if const_expr(m == 1) else wave
        pair = (wave // fx.Int32(2)) if const_expr(m == 1) else fx.Int32(0)
        n0 = fx.block_idx.x * fx.Int32(config.block_n) + pair * fx.Int32(64)
        kb_begin = half_id * fx.Int32(half_blocks)
        kb_end_unclamped = kb_begin + fx.Int32(half_blocks)
        kb_end = (kb_end_unclamped < fx.Int32(scale_blocks)).select(
            kb_end_unclamped, fx.Int32(scale_blocks)
        )

        a_ptr = fx.recast_iter(fx.Uint8, fx.get_iter(arg_a))
        b_ptr = fx.recast_iter(fx.Uint8, fx.get_iter(arg_b))
        a_buf = _make_buffer(arg_a, fp8, 8, m * k)
        b_buf = _make_buffer(arg_b, fp8, 8, n * stride_b)
        as_ptr = fx.recast_iter(f32, fx.get_iter(arg_as))
        bs_ptr = fx.recast_iter(f32, fx.get_iter(arg_bs))
        out_ptr = fx.recast_iter(bf16, fx.get_iter(arg_out))
        shared = (
            fx.SharedAllocator()
            .allocate(SharedStorage)
            .peek()
            .partial.view(fx.make_layout(shared_elems, 1))
        )

        mma = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fp8, f32))
        totals = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(4)]
        for j in range_constexpr(4):
            totals[j].fill(0)

        for kb in range(kb_begin, kb_end, 1):
            partials = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(4)]
            for j in range_constexpr(4):
                partials[j].fill(0)

            if const_expr(m == 1):
                if const_expr(n >= 16384):
                    # Streaming the very-wide projection benefits more from a
                    # second resident wave than from exposing the full K128
                    # load train. Keep four K16 slices live at a time here.
                    for ks_group in range_constexpr(2):
                        a_frags = [
                            fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)
                        ]
                        b_frags = [
                            [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)]
                            for _ in range_constexpr(4)
                        ]
                        for ks_local in range_constexpr(4):
                            ks = ks_group * 4 + ks_local
                            k_lane = (
                                kb * fx.Int32(SCALE_K)
                                + fx.Int32(ks * WMMA_K)
                                + (lane // fx.Int32(16)) * fx.Int32(8)
                            )
                            _load_fp8_fragment_buffer(a_buf, k_lane, a_frags[ks_local])
                            for j in range_constexpr(4):
                                b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                                _load_fp8_fragment_buffer(
                                    b_buf,
                                    b_row * fx.Int32(stride_b) + k_lane,
                                    b_frags[ks_local][j],
                                )
                        for ks_local in range_constexpr(4):
                            for j in range_constexpr(4):
                                fx.gemm(
                                    mma,
                                    partials[j],
                                    a_frags[ks_local],
                                    b_frags[ks_local][j],
                                    partials[j],
                                )
                        fx.rocdl.sched_vmem(20)
                        fx.rocdl.sched_mfma(16)
                        fx.rocdl.sched_barrier(0)
                else:
                    # Cache-resident projections benefit from exposing the
                    # entire K=128 block's 40 loads before its 32 WMMAs.
                    a_frags = [
                        fx.make_rmem_tensor(8, fp8)
                        for _ in range_constexpr(SCALE_K // WMMA_K)
                    ]
                    b_frags = [
                        [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)]
                        for _ in range_constexpr(SCALE_K // WMMA_K)
                    ]
                    for ks in range_constexpr(SCALE_K // WMMA_K):
                        k_lane = (
                            kb * fx.Int32(SCALE_K)
                            + fx.Int32(ks * WMMA_K)
                            + (lane // fx.Int32(16)) * fx.Int32(8)
                        )
                        _load_fp8_fragment_buffer(a_buf, k_lane, a_frags[ks])
                        for j in range_constexpr(4):
                            b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                            _load_fp8_fragment_buffer(
                                b_buf,
                                b_row * fx.Int32(stride_b) + k_lane,
                                b_frags[ks][j],
                            )
                    for ks in range_constexpr(SCALE_K // WMMA_K):
                        for j in range_constexpr(4):
                            fx.gemm(
                                mma,
                                partials[j],
                                a_frags[ks],
                                b_frags[ks][j],
                                partials[j],
                            )
                    fx.rocdl.sched_vmem(40)
                    fx.rocdl.sched_mfma(32)
                    fx.rocdl.sched_barrier(0)
            else:
                for ks in range_constexpr(SCALE_K // WMMA_K):
                    k_lane = (
                        kb * fx.Int32(SCALE_K)
                        + fx.Int32(ks * WMMA_K)
                        + (lane // fx.Int32(16)) * fx.Int32(8)
                    )
                    a_frag = fx.make_rmem_tensor(8, fp8)
                    a_frag.fill(0)
                    a_row = lane % fx.Int32(WMMA_M)
                    if a_row < fx.Int32(m):
                        _load_fp8_fragment_ptr(
                            a_ptr, a_row * fx.Int32(k) + k_lane, a_frag
                        )

                    b_step = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(4)]
                    for j in range_constexpr(4):
                        b_row = n0 + fx.Int32(j * WMMA_N) + lane_col
                        _load_fp8_fragment_ptr(
                            b_ptr, b_row * fx.Int32(stride_b) + k_lane, b_step[j]
                        )
                        fx.gemm(mma, partials[j], a_frag, b_step[j], partials[j])

            if const_expr(m == 1):
                b_group = fx.block_idx.x
            else:
                b_group = fx.block_idx.x // fx.Int32(2)
            b_scale = f32(fx.ptr_load(bs_ptr + b_group * fx.Int32(scale_blocks) + kb))
            scales = [
                f32(fx.ptr_load(as_ptr + fx.Int32(value_idx * scale_blocks) + kb))
                * b_scale
                for value_idx in range_constexpr(m)
            ]
            for j in range_constexpr(4):
                total_v = Vec(totals[j].load())
                partial_v = Vec(partials[j].load())
                updated = [
                    total_v[x] + partial_v[x] * scales[x] if x < m else total_v[x]
                    for x in range_constexpr(8)
                ]
                totals[j].store(Vec.from_elements(updated, f32))

        if lane < fx.Int32(16):
            for j in range_constexpr(4):
                total_v = Vec(totals[j].load())
                for row in range_constexpr(m):
                    index = (
                        (
                            (wave * fx.Int32(4) + fx.Int32(j)) * fx.Int32(m)
                            + fx.Int32(row)
                        )
                        * fx.Int32(16)
                    ) + lane
                    fx.memref_store(total_v[row], shared, index)
        # Keep the split barrier open while computing the LDS read addresses.
        # This matches the tuned HIP path and avoids the global-memory fence
        # carried by a generic gpu.barrier().
        lds_fence_signal()

        frag = tid // fx.Int32(16)
        col = tid % fx.Int32(16)
        if const_expr(m == 1):
            out_pair = tid // fx.Int32(64)
            local = tid % fx.Int32(64)
            frag = local // fx.Int32(16)
            col = local % fx.Int32(16)
            wave0 = out_pair * fx.Int32(2)
            index0 = (wave0 * fx.Int32(4) + frag) * fx.Int32(16) + col
            index1 = ((wave0 + fx.Int32(1)) * fx.Int32(4) + frag) * fx.Int32(16) + col
            lds_fence_wait()
            value = _f32_to_bf16_rne(
                f32(fx.memref_load(shared, index0) + fx.memref_load(shared, index1))
            )
            fx.ptr_store(
                value, out_ptr + fx.block_idx.x * fx.Int32(config.block_n) + tid
            )
        else:
            lds_fence_wait()
            for row in range_constexpr(2):
                index0 = (frag * fx.Int32(2) + fx.Int32(row)) * fx.Int32(16) + col
                index1 = (
                    (fx.Int32(4) + frag) * fx.Int32(2) + fx.Int32(row)
                ) * fx.Int32(16) + col
                value = _f32_to_bf16_rne(
                    f32(fx.memref_load(shared, index0) + fx.memref_load(shared, index1))
                )
                out_index = (
                    fx.Int32(row * n) + fx.block_idx.x * fx.Int32(config.block_n) + tid
                )
                fx.ptr_store(value, out_ptr + out_index)

    @flyc.jit
    def launch(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_as: fx.Tensor,
        arg_bs: fx.Tensor,
        arg_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel_attrs = {
            "rocdl.waves_per_eu": 1,
            "rocdl.flat_work_group_size": f"{config.threads},{config.threads}",
        }
        splitk_kernel(
            arg_a,
            arg_b,
            arg_as,
            arg_bs,
            arg_out,
            value_attrs=kernel_attrs,
        ).launch(
            grid=(n // config.block_n, 1, 1),
            block=(config.threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=128)
def _get_module(m: int, n: int, k: int, stride_b: int, config: KernelConfig):
    if config.split_k == 2:
        return _create_splitk_module(m, n, k, stride_b, config)
    if config.route is KernelRoute.DECODE_PACKED_M4_N64:
        return _create_m4_n64_module(n, k, stride_b, config)
    return _create_packed_module(m, n, k, stride_b, config)


def _validate_tensors(a, weight, a_scale, weight_scale, out):
    import torch

    tensors = (a, weight, a_scale, weight_scale)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be on the GPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be rank two")
    if a.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("RDNA4 block-FP8 operands must use torch.float8_e4m3fn")
    if a_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("RDNA4 block-FP8 scales must use torch.float32")
    if any(t.device != a.device for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must share a device")
    arch = getattr(torch.cuda.get_device_properties(a.device), "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise ValueError(
            f"RDNA4 block-FP8 route requires gfx1200 or gfx1201, got {arch!r}"
        )

    m, k = a.shape
    n, weight_k = weight.shape
    config = select_kernel_config(m, n, k)
    if weight_k != k:
        raise ValueError(f"weight K mismatch: A has K={k}, weight has K={weight_k}")
    if (
        not a.is_contiguous()
        or not a_scale.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("A and both scale tensors must be contiguous")
    if weight.stride(1) != 1:
        raise ValueError("weight must have contiguous K rows (weight.stride(1) == 1)")
    if tuple(a_scale.shape) != (m, k // SCALE_K):
        raise ValueError(
            f"activation scale shape must be {(m, k // SCALE_K)}, "
            f"got {tuple(a_scale.shape)}"
        )
    if tuple(weight_scale.shape) != (n // SCALE_K, k // SCALE_K):
        raise ValueError(
            "weight scale shape must be "
            f"{(n // SCALE_K, k // SCALE_K)}, got {tuple(weight_scale.shape)}"
        )
    if out is not None and (
        out.device != a.device
        or out.dtype != torch.bfloat16
        or tuple(out.shape) != (m, n)
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous BF16 CUDA tensor with shape [M, N] on A's device"
        )
    return m, n, k, config


def rdna4_fp8_block_scaled_mm_decode(
    a, weight, a_scale, weight_scale, *, out=None, stream=None
):
    """Run the vLLM-compatible M1--M64 RDNA4 block-scaled FP8 GEMM.

    The first four positional arguments intentionally match the vLLM custom op.
    ``out`` and ``stream`` are optional integration conveniences; omitting them
    allocates a fresh BF16 result on PyTorch's current stream.
    """
    import torch

    m, n, k, config = _validate_tensors(a, weight, a_scale, weight_scale, out)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    module = _get_module(m, n, k, weight.stride(0), config)
    run_compiled(module, a, weight, a_scale, weight_scale, out, stream)
    return out


def rdna4_fp8_block_scaled_mm(
    a, weight, a_scale, weight_scale, *, out=None, stream=None
):
    """Run the complete RDNA4 block-scaled FP8 stack for any positive M.

    M1--M64 retains the register-tiled decode implementation. Broad prefill
    shapes use the register-prefetched LDS implementation in the companion
    module. The import is deliberately lazy so that module can reuse the
    common gfx120x buffer and WMMA helpers above without an import cycle.
    """
    if getattr(a, "ndim", None) == 2 and a.shape[0] > 64:
        from .rdna4_fp8_blockscale_prefill import (
            rdna4_fp8_block_scaled_mm_prefill,
        )

        return rdna4_fp8_block_scaled_mm_prefill(
            a, weight, a_scale, weight_scale, out=out, stream=stream
        )
    return rdna4_fp8_block_scaled_mm_decode(
        a, weight, a_scale, weight_scale, out=out, stream=stream
    )


__all__ = [
    "KernelConfig",
    "KernelRoute",
    "rdna4_fp8_block_scaled_mm",
    "rdna4_fp8_block_scaled_mm_decode",
    "select_kernel_config",
]
