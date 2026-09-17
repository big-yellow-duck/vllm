# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit Triton Gluon WMMA layout experiment for the fixed short-prefill shape."""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna4 import wmma

from .page_kernel import merge_attention


@g.jit
def get_scores(
    Q,
    K,
    KD,
    mb,
    kh,
    block,
    start,
    K0: gl.constexpr,
    K1: gl.constexpr,
    K2: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    WL: gl.constexpr,
    QL: gl.constexpr,
    KL: gl.constexpr,
    PAGE: gl.constexpr,
    BASE,
    DENSE: gl.constexpr,
):
    mq = mb * BM + gl.arange(0, BM, layout=SliceLayout(1, QL))
    nk = start + gl.arange(0, BN, layout=SliceLayout(0, KL))
    a = gl.full((BM, BN), 0, gl.float32, WL)
    for ki in range(256 // BK):
        dq = ki * BK + gl.arange(0, BK, layout=SliceLayout(0, QL))
        dk = ki * BK + gl.arange(0, BK, layout=SliceLayout(1, KL))
        q = gl.load(
            Q
            + (mq[:, None] // 6) * 12 * 256
            + (kh * 6 + mq[:, None] % 6) * 256
            + dq[None, :],
            mask=(mq < 192)[:, None],
            other=0,
        )
        if DENSE:
            k = gl.load(
                KD + nk[None, :] * 2 * 256 + kh * 256 + dk[:, None],
                mask=(nk < 32)[None, :],
                other=0,
            )
        else:
            k = gl.load(
                K
                + block * K0
                + kh * K1
                + (dk[:, None] // 8) * K2
                + nk[None, :] * 8
                + dk[:, None] % 8,
                mask=(nk < PAGE)[None, :] & ((BASE + nk) < 8192)[None, :],
                other=0,
            )
        q = gl.convert_layout(q, DotOperandLayout(0, WL, 8))
        k = gl.convert_layout(k, DotOperandLayout(1, WL, 8))
        a = wmma(q, k, a)
    return a * (1.4426950408889634 / 16.0)


@g.jit
def step(
    Q,
    K,
    V,
    KD,
    VD,
    mb,
    kh,
    block,
    start,
    base,
    maximum,
    denom,
    acc,
    K0: gl.constexpr,
    K1: gl.constexpr,
    K2: gl.constexpr,
    V0: gl.constexpr,
    V1: gl.constexpr,
    V2: gl.constexpr,
    PAGE: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    WL: gl.constexpr,
    QL: gl.constexpr,
    KL: gl.constexpr,
    VL: gl.constexpr,
    DENSE: gl.constexpr,
):
    scores = get_scores(
        Q,
        K,
        KD,
        mb,
        kh,
        block,
        start,
        K0,
        K1,
        K2,
        BM,
        BN,
        BK,
        WL,
        QL,
        KL,
        PAGE,
        base,
        DENSE,
    )
    m = mb * BM + gl.arange(0, BM, layout=SliceLayout(1, WL))
    ns = start + gl.arange(0, BN, layout=SliceLayout(0, WL))
    if DENSE:
        valid = (ns < 32)[None, :] & (ns[None, :] <= m[:, None] // 6)
    else:
        valid = (ns < PAGE)[None, :] & ((base + ns) < 8192)[None, :]
    scores = gl.where(valid, scores, -float("inf"))
    newmax = gl.maximum(maximum, gl.max(scores, 1))
    newmax = gl.where(newmax == -float("inf"), 0.0, newmax)
    alpha = gl.exp2(maximum - newmax)
    p = gl.exp2(scores - newmax[:, None])
    acc = acc * alpha[:, None]
    nv = start + gl.arange(0, BN, layout=SliceLayout(1, VL))
    dv = gl.arange(0, 256, layout=SliceLayout(0, VL))
    if DENSE:
        v = gl.load(
            VD + nv[:, None] * 2 * 256 + kh * 256 + dv[None, :],
            mask=(nv < 32)[:, None],
            other=0,
        )
    else:
        v = gl.load(
            V + block * V0 + kh * V1 + dv[None, :] * V2 + nv[:, None],
            mask=(nv < PAGE)[:, None] & ((base + nv) < 8192)[:, None],
            other=0,
        )
    p = gl.convert_layout(p.to(gl.bfloat16), DotOperandLayout(0, WL, 8))
    v = gl.convert_layout(v, DotOperandLayout(1, WL, 8))
    acc = wmma(p, v, acc)
    denom = denom * alpha + gl.sum(gl.exp2(scores - newmax[:, None]), 1)
    return newmax, denom, acc


@g.jit
def partial(
    Q,
    K,
    V,
    KD,
    VD,
    TABLE,
    PART,
    LSE,
    K0: gl.constexpr,
    K1: gl.constexpr,
    K2: gl.constexpr,
    V0: gl.constexpr,
    V1: gl.constexpr,
    V2: gl.constexpr,
    PAGE: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    PPS: gl.constexpr,
    SPLITS: gl.constexpr,
    WL: gl.constexpr,
    QL: gl.constexpr,
    KL: gl.constexpr,
    VL: gl.constexpr,
):
    mb, kh, s = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    page = s // PPS
    block = gl.load(TABLE + page)
    CHUNK: gl.constexpr = triton.cdiv(triton.cdiv(PAGE, PPS), BN) * BN
    start = (s % PPS) * CHUNK
    end = gl.minimum(start + CHUNK, PAGE)
    maximum = gl.full((BM,), -float("inf"), gl.float32, SliceLayout(1, WL))
    denom = gl.full((BM,), 0, gl.float32, SliceLayout(1, WL))
    acc = gl.full((BM, 256), 0, gl.float32, WL)
    for n in range(start, end, BN):
        maximum, denom, acc = step(
            Q,
            K,
            V,
            KD,
            VD,
            mb,
            kh,
            block,
            n,
            page * PAGE,
            maximum,
            denom,
            acc,
            K0,
            K1,
            K2,
            V0,
            V1,
            V2,
            PAGE,
            BM,
            BN,
            BK,
            WL,
            QL,
            KL,
            VL,
            False,
        )
    if s == SPLITS - 1:
        maximum, denom, acc = step(
            Q,
            K,
            V,
            KD,
            VD,
            mb,
            kh,
            block,
            0,
            0,
            maximum,
            denom,
            acc,
            K0,
            K1,
            K2,
            V0,
            V1,
            V2,
            PAGE,
            BM,
            BN,
            BK,
            WL,
            QL,
            KL,
            VL,
            True,
        )
    m = mb * BM + gl.arange(0, BM, layout=SliceLayout(1, WL))
    d = gl.arange(0, 256, layout=SliceLayout(0, WL))
    gl.store(
        PART + ((s * 2 + kh) * 192 + m[:, None]) * 256 + d[None, :],
        acc / gl.where(denom == 0, 1.0, denom)[:, None],
        mask=(m < 192)[:, None],
    )
    gl.store(LSE + (s * 2 + kh) * 192 + m, maximum + gl.log2(denom), mask=m < 192)


def make_call(data, config, context=8192):
    q, k, v = data["q"], data["kc"], data["vc"]
    assert context == 8192 and config["bn"] >= 32
    bm, bn, bk = config["bm"], config["bn"], config["bk"]
    pps = config.get("pps", 1)
    splits = triton.cdiv(context, k.shape[3]) * pps
    warps = config.get("warps", 4)
    bases = config.get("bases", [[1, 0], [2, 0]])
    wl = AMDWMMALayout(version=2, transposed=True, warp_bases=bases)
    ql = BlockedLayout([1, 4], [4, 8], [warps, 1], [1, 0])
    kl = BlockedLayout([8, 1], [4, 8], [1, warps], [0, 1])
    vl = BlockedLayout([4, 1], [8, 4], [1, warps], [0, 1])
    part = torch.empty((splits, 2, 192, 256), device=q.device, dtype=torch.float32)
    lse = torch.empty((splits, 2, 192), device=q.device)
    out = torch.empty_like(q)

    def stage():
        c = partial[(triton.cdiv(192, bm), 2, splits)](
            q,
            k,
            v,
            data["kd"],
            data["vd"],
            data["table"],
            part,
            lse,
            *k.stride()[:3],
            *v.stride()[:3],
            k.shape[3],
            bm,
            bn,
            bk,
            pps,
            splits,
            wl,
            ql,
            kl,
            vl,
            num_warps=warps,
            num_stages=config.get("stages", 1),
            waves_per_eu=config.get("waves", 2),
        )
        run.resources = dict(
            shared=c.metadata.shared, registers=c.n_regs, spills=c.n_spills
        )

    def reduce():
        merge_attention[(192, 2)](part, lse, out, 32, 12, 2, 256, splits, num_warps=4)

    def run():
        stage()
        reduce()

    run.stage, run.reduce = stage, reduce
    return run, out
