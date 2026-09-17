# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental materialized-score attention; all intermediate work is timed."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def qk_kernel(
    Q,
    K,
    KD,
    TABLE,
    SCORES,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    PAGE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NN: tl.constexpr,
):
    mi, ni, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    if ni * BN < 8192:
        block = tl.load(TABLE + n // PAGE, mask=n < 8192, other=0)
        off = n % PAGE
        for di in range(256 // BK):
            d = di * BK + tl.arange(0, BK)
            q = tl.load(
                Q
                + (m[:, None] // 6) * 12 * 256
                + (h * 6 + m[:, None] % 6) * 256
                + d[None, :],
                mask=(m < 192)[:, None],
                other=0,
            )
            k = tl.load(
                K
                + block[None, :] * K0
                + h * K1
                + (d[:, None] // 8) * K2
                + off[None, :] * 8
                + d[:, None] % 8,
                mask=(n < 8192)[None, :],
                other=0,
            )
            acc = tl.dot(q, k, acc)
    else:
        for di in range(256 // BK):
            d = di * BK + tl.arange(0, BK)
            q = tl.load(
                Q
                + (m[:, None] // 6) * 12 * 256
                + (h * 6 + m[:, None] % 6) * 256
                + d[None, :],
                mask=(m < 192)[:, None],
                other=0,
            )
            k = tl.load(
                KD + (n[None, :] - 8192) * 2 * 256 + h * 256 + d[:, None],
                mask=(n < 8224)[None, :],
                other=0,
            )
            acc = tl.dot(q, k, acc)
    acc = tl.where(
        (n[None, :] <= 8192 + m[:, None] // 6) & (n[None, :] < 8224),
        acc * (1.4426950408889634 / 16),
        -float("inf"),
    )
    tl.store(
        SCORES + (h * 192 + m[:, None]) * NN + n[None, :],
        acc,
        mask=(m < 192)[:, None] & (n < NN)[None, :],
    )


@tr.jit
def softmax_kernel(SCORES, P, NN: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0)
    n = tl.arange(0, BN)
    s = tl.load(SCORES + row * NN + n, mask=n < NN, other=-float("inf"))
    p = tl.exp2(s - tl.max(s, 0))
    p = p / tl.sum(p, 0)
    tl.store(P + row * NN + n, p, mask=n < NN)


@tr.jit
def pv_kernel(
    P,
    V,
    VD,
    TABLE,
    PART,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    PAGE: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    NN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    tile, h, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = (tile // (256 // BD)) * BM + tl.arange(0, BM)
    d = (tile % (256 // BD)) * BD + tl.arange(0, BD)
    SLEN: tl.constexpr = 8192 // SPLITS
    acc = tl.zeros((BM, BD), tl.float32)
    for ni in range(SLEN // BN):
        n = split * SLEN + ni * BN + tl.arange(0, BN)
        p = tl.load(
            P + (h * 192 + m[:, None]) * NN + n[None, :],
            mask=(m < 192)[:, None],
            other=0,
        )
        block = tl.load(TABLE + n // PAGE)
        v = tl.load(
            V + block[:, None] * V0 + h * V1 + d[None, :] * V2 + (n % PAGE)[:, None]
        )
        acc = tl.dot(p, v, acc)
    if split == SPLITS - 1:
        n = tl.arange(0, 32)
        p = tl.load(
            P + (h * 192 + m[:, None]) * NN + 8192 + n[None, :],
            mask=(m < 192)[:, None],
            other=0,
        )
        v = tl.load(VD + n[:, None] * 2 * 256 + h * 256 + d[None, :])
        acc = tl.dot(p, v, acc)
    tl.store(
        PART + ((split * 2 + h) * 192 + m[:, None]) * 256 + d[None, :],
        acc,
        mask=(m < 192)[:, None],
    )


@tr.jit
def reduce_kernel(PART, OUT, SPLITS: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 256)
    p = tl.load(PART + ((s[:, None] * 2 + h) * 192 + row) * 256 + d[None, :])
    tl.store(OUT + (row // 6) * 12 * 256 + (h * 6 + row % 6) * 256 + d, tl.sum(p, 0))


def make_call(data, config, context=8192):
    q, k, v = data["q"], data["kc"], data["vc"]
    bm, bn, bk = config.get("qm", 32), config.get("qn", 64), config.get("qk", 64)
    pm, pd, pn = config.get("pm", 32), config.get("pd", 64), config.get("pn", 32)
    splits = config["splits"]
    nn = tr.cdiv(8224, bn) * bn
    scores = torch.empty((2, 192, nn), device=q.device, dtype=torch.float32)
    p = torch.empty_like(scores, dtype=torch.bfloat16)
    part = torch.empty((splits, 2, 192, 256), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)

    def qk():
        c = qk_kernel[(tr.cdiv(192, bm), tr.cdiv(nn, bn), 2)](
            q,
            k,
            data["kd"],
            data["table"],
            scores,
            *k.stride()[:3],
            k.shape[3],
            bm,
            bn,
            bk,
            nn,
            num_warps=config.get("qw", 4),
            num_stages=config.get("qs", 1),
        )
        run.resources["qk"] = dict(
            shared=c.metadata.shared, registers=c.n_regs, spills=c.n_spills
        )

    def softmax():
        softmax_kernel[(384,)](scores, p, nn, tr.next_power_of_2(nn), num_warps=16)

    def pv():
        c = pv_kernel[(tr.cdiv(192, pm) * (256 // pd), 2, splits)](
            p,
            v,
            data["vd"],
            data["table"],
            part,
            *v.stride()[:3],
            k.shape[3],
            pm,
            pd,
            pn,
            nn,
            splits,
            num_warps=config.get("pw", 4),
            num_stages=config.get("ps", 1),
        )
        run.resources["pv"] = dict(
            shared=c.metadata.shared, registers=c.n_regs, spills=c.n_spills
        )

    def stage():
        qk()
        softmax()
        pv()

    def reduce():
        reduce_kernel[(192, 2)](part, out, splits, num_warps=4)

    def run():
        stage()
        reduce()

    run.resources = {}
    run.stage, run.reduce, run.qk, run.softmax, run.pv = stage, reduce, qk, softmax, pv
    return run, out
