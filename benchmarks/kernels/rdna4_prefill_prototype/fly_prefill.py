# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B008, B023 -- FlyDSL traces nested functions and stream defaults.
"""Experimental four-wave BF16 prefill stage, adapted from RDNA4 SplitKV."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from rdna4_prefill_prototype.fly_common import (
    LOG2E,
    _flat_view,
    _wave_reduce,
)
from rdna4_prefill_prototype.fly_runtime import run_compiled

from vllm.v1.attention.ops.segmented_prefill import _segmented_prefill_reduce


@functools.lru_cache(128)
def compile_stage(batch, hq, hk, dim, page, qcap, splits, strides):
    g = hq // hk
    rows = qcap * g
    q0, q1, k0, k1, k2, k3, v0, v1, v2, v3, d0, d1, e0, e1, t0 = strides
    scale = dim**-0.5 * LOG2E
    cols = dim // 64

    @fx.struct
    class Shared:
        scores: fx.Array[fx.Float32, 1024, 16]
        weights: fx.Array[fx.BFloat16, 1024, 16]
        rescale: fx.Array[fx.Float32, 16, 16]
        denom: fx.Array[fx.Float32, 16, 16]

    @flyc.kernel(known_block_size=(128, 1, 1))
    def stage(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        T: fx.Tensor,
        START: fx.Tensor,
        LEN: fx.Tensor,
        P: fx.Tensor,
        L: fx.Tensor,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 32
        wave = tid // 32
        block = fx.Int32(gpu.block_id("x"))
        split = block % splits
        item = block // splits
        kh = item % hk
        seq = (item // hk) % batch
        mt = item // (hk * batch)
        query = _flat_view(Q)
        key = _flat_view(K)
        value = _flat_view(V)
        dk = _flat_view(DK)
        dv = _flat_view(DV)
        table = _flat_view(T)
        starts = _flat_view(START)
        lens = _flat_view(LEN)
        part = _flat_view(P)
        lse = _flat_view(L)
        first = fx.Int32(starts[seq])
        nq = fx.Int32(starts[seq + 1]) - first
        if (nq > 0) & (nq <= 128) & (mt * 16 < nq * g):
            prefix = fx.Int32(lens[seq]) - nq
            seg = ((prefix + splits * 64 - 1) // (splits * 64)) * 64
            begin = fx.min(split * seg, prefix)
            end = fx.min(begin + seg, prefix)
            prefix_tiles = (end - begin + 63) // 64
            tiles = prefix_tiles + (split == splits - 1).select(
                (nq + 63) // 64, fx.Int32(0)
            )
            storage = fx.SharedAllocator().allocate(Shared).peek()
            scores = storage.scores.view(fx.make_layout((16, 64), (64, 1)))
            weights = storage.weights.view(fx.make_layout((16, 64), (64, 1)))
            scales = storage.rescale.view(fx.make_layout(16, 1))
            denoms = storage.denom.view(fx.make_layout(16, 1))
            copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            qdiv = fx.logical_divide(query, fx.make_layout(1, 1))
            kdiv = fx.logical_divide(key, fx.make_layout(1, 1))
            vdiv = fx.logical_divide(value, fx.make_layout(1, 1))
            qr = mt * 16 + lane % 16
            qtoken = first + fx.min(qr // g, nq - 1)
            qh = kh * g + qr % g
            qfrags = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                for _ in range_constexpr(dim // 16)
            ]
            for j in range_constexpr(dim // 16):
                qi = qtoken * q0 + qh * q1 + j * 16 + (lane // 16) * 8
                fx.copy(copy, fx.slice(qdiv, (None, qi)), qfrags[j])
            kfrags = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                for _ in range_constexpr(4)
            ]
            pfrags = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                for _ in range_constexpr(4)
            ]
            vfrags = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                for _ in range_constexpr(4)
            ]
            sf = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
            of = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
                for _ in range_constexpr(cols)
            ]
            mma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.BFloat16, fx.Float32))
            zero = fx.Float32(0.0)
            neg = fx.Float32(float("-inf"))
            init = [neg, zero] * 4 + [zero for _ in range_constexpr(cols * 8)]
            for tile, state in range(fx.Int32(0), tiles, fx.Int32(1), init=init):
                tile = fx.Int32(tile)
                is_prefix = tile < prefix_tiles
                base = is_prefix.select(begin + tile * 64, (tile - prefix_tiles) * 64)
                kt = base + wave * 16 + lane % 16

                @flyc.jit
                def load_keys(
                    f0: fx.Tensor,
                    f1: fx.Tensor,
                    f2: fx.Tensor,
                    f3: fx.Tensor,
                    jbase: fx.Constexpr[int],
                ):
                    if is_prefix:
                        safe = fx.min(kt, end - 1)
                        physical = fx.Int64(table[seq * t0 + safe // page])
                        fs = [f0, f1, f2, f3]
                        for jj in range_constexpr(4):
                            j = jbase + jj
                            ki = (
                                physical * k0
                                + kh * k1
                                + (j * 2 + lane // 16) * k2
                                + (safe % page) * k3
                            )
                            fx.copy(copy, fx.slice(kdiv, (None, ki)), fs[jj])
                    else:
                        safe = fx.min(kt, nq - 1)
                        fs = [f0, f1, f2, f3]
                        for jj in range_constexpr(4):
                            j = jbase + jj
                            for e in range_constexpr(8):
                                fs[jj][e] = dk[
                                    (first + safe) * d0
                                    + kh * d1
                                    + j * 16
                                    + (lane // 16) * 8
                                    + e
                                ]

                sf.fill(0.0)
                for jbase in range_constexpr(dim // 64):
                    load_keys(kfrags[0], kfrags[1], kfrags[2], kfrags[3], jbase * 4)
                    for jj in range_constexpr(4):
                        fx.mma_atom_call(
                            mma, sf, qfrags[jbase * 4 + jj], kfrags[jj], sf
                        )
                sv = fx.Vector(sf.load())
                for r in range_constexpr(8):
                    scores[(lane // 16) * 8 + r, wave * 16 + lane % 16] = sv[r] * scale
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                stats = []
                for rr in range_constexpr(4):
                    row = wave + rr * 4
                    m = mt * 16 + row
                    a = base + lane
                    b = a + 32
                    va = is_prefix.select(a < end, (a < nq) & (a <= m // g))
                    vb = is_prefix.select(b < end, (b < nq) & (b <= m // g))
                    sa = va.select(fx.Float32(scores[row, lane]), neg)
                    sb = vb.select(fx.Float32(scores[row, lane + 32]), neg)
                    mx = _wave_reduce(fx.max(sa, sb), "max")
                    oldmax = fx.Float32(state[rr * 2])
                    oldsum = fx.Float32(state[rr * 2 + 1])
                    newmax = fx.max(oldmax, mx)
                    safe_max = (newmax == neg).select(zero, newmax)
                    alpha = fmath.exp2(oldmax - safe_max)
                    pa = va.select(fmath.exp2(sa - safe_max), zero)
                    pb = vb.select(fmath.exp2(sb - safe_max), zero)
                    new_sum = oldsum * alpha + _wave_reduce(pa + pb, "sum")
                    weights[row, lane] = pa.to(fx.BFloat16)
                    weights[row, lane + 32] = pb.to(fx.BFloat16)
                    if lane == 0:
                        scales[row] = alpha
                    stats.extend([newmax, new_sum])
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                accum = []
                for j in range_constexpr(4):
                    for e in range_constexpr(8):
                        pfrags[j][e] = weights[lane % 16, j * 16 + (lane // 16) * 8 + e]
                rscales = [
                    fx.Float32(scales[(lane // 16) * 8 + r]) for r in range_constexpr(8)
                ]

                @flyc.jit
                def load_values(
                    f0: fx.Tensor,
                    f1: fx.Tensor,
                    f2: fx.Tensor,
                    f3: fx.Tensor,
                    dc: fx.Int32,
                ):
                    if is_prefix:
                        if base + 64 <= end:
                            fs = [f0, f1, f2, f3]
                            for j in range_constexpr(4):
                                vt = base + j * 16 + (lane // 16) * 8
                                physical = fx.Int64(table[seq * t0 + vt // page])
                                vi = (
                                    physical * v0 + kh * v1 + dc * v2 + (vt % page) * v3
                                )
                                fx.copy(copy, fx.slice(vdiv, (None, vi)), fs[j])
                        else:
                            fs = [f0, f1, f2, f3]
                            for j in range_constexpr(4):
                                for e in range_constexpr(8):
                                    safe = fx.min(
                                        base + j * 16 + (lane // 16) * 8 + e, end - 1
                                    )
                                    physical = fx.Int64(table[seq * t0 + safe // page])
                                    vi = (
                                        physical * v0
                                        + kh * v1
                                        + dc * v2
                                        + (safe % page) * v3
                                    )
                                    fs[j][e] = value[vi]
                    else:
                        fs = [f0, f1, f2, f3]
                        for j in range_constexpr(4):
                            for e in range_constexpr(8):
                                safe = fx.min(
                                    base + j * 16 + (lane // 16) * 8 + e, nq - 1
                                )
                                fs[j][e] = dv[(first + safe) * e0 + kh * e1 + dc]

                for c in range_constexpr(cols):
                    dc = wave * (dim // 4) + c * 16 + lane % 16
                    load_values(vfrags[0], vfrags[1], vfrags[2], vfrags[3], dc)
                    for r in range_constexpr(8):
                        of[c][r] = fx.Float32(state[8 + c * 8 + r]) * rscales[r]
                    for j in range_constexpr(4):
                        fx.mma_atom_call(mma, of[c], pfrags[j], vfrags[j], of[c])
                    oval = fx.Vector(of[c].load())
                    accum.extend([oval[r] for r in range_constexpr(8)])
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                results = yield stats + accum
            for rr in range_constexpr(4):
                row = wave + rr * 4
                denominator = fx.Float32(results[rr * 2 + 1])
                if lane == 0:
                    denoms[row] = denominator
                    m = mt * 16 + row
                    if m < nq * g:
                        li = ((split * batch + seq) * hk + kh) * rows + m
                        lse[li] = (denominator > 0.0).select(
                            fx.Float32(results[rr * 2]) + fmath.log2(denominator), neg
                        )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            for c in range_constexpr(cols):
                dc = wave * (dim // 4) + c * 16 + lane % 16
                for r in range_constexpr(8):
                    row = (lane // 16) * 8 + r
                    m = mt * 16 + row
                    if m < nq * g:
                        den = fx.Float32(denoms[row])
                        safe_den = (den > 0.0).select(den, fx.Float32(1.0))
                        pi = (((split * batch + seq) * hk + kh) * rows + m) * dim + dc
                        part[pi] = fx.Float32(results[8 + c * 8 + r]) / safe_den

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        T: fx.Tensor,
        START: fx.Tensor,
        LEN: fx.Tensor,
        P: fx.Tensor,
        L: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        stage(Q, K, V, DK, DV, T, START, LEN, P, L).launch(
            grid=(((rows + 15) // 16) * batch * hk * splits,),
            block=(128,),
            stream=stream,
        )

    return launch


def make_call(data, splits=8):
    q = data["q"]
    kc = data["kc"]
    vc = data["vc"]
    dk = data["k"]
    dv = data["v"]
    assert q.dtype == kc.dtype == torch.bfloat16
    assert q.shape[2] in (128, 256) and kc.shape[3] % 16 == 0
    batch = len(data["queries"])
    hq, dim = q.shape[1:]
    hk = kc.shape[1]
    qcap = max(data["queries"])
    assert 0 < qcap <= 128
    rows = qcap * (hq // hk)
    part = torch.empty(
        (splits, batch, hk, rows, dim), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((splits, batch, hk, rows), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    strides = (
        *q.stride()[:2],
        *kc.stride()[:4],
        *vc.stride(),
        *dk.stride()[:2],
        *dv.stride()[:2],
        data["table"].stride(0),
    )
    stage = compile_stage(batch, hq, hk, dim, kc.shape[3], qcap, splits, strides)

    def run():
        run_compiled(
            stage,
            q,
            kc,
            vc,
            dk,
            dv,
            data["table"],
            data["starts"],
            data["lens"],
            part,
            lse,
            torch.cuda.current_stream(q.device),
        )
        _segmented_prefill_reduce[(rows, batch * hk)](
            part,
            lse,
            out,
            data["starts"],
            *out.stride()[:2],
            batch,
            hq,
            hk,
            dim,
            qcap,
            1,
            128,
            splits,
            num_warps=4,
        )

    return run, out
