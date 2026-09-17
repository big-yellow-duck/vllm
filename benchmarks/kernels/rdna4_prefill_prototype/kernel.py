# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Isolated BF16 paged short-prefill prototype; not a production backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def partial_attention(
    Q,
    K,
    V,
    KD,
    VD,
    TABLE,
    PART,
    LSE,
    OUT,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    PAGE: tl.constexpr,
    C: tl.constexpr,
    NQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
):
    mb, kh, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = NQ * G
    SLEN: tl.constexpr = C // SPLITS
    dslice = mb // tl.cdiv(ROWS, BM)
    mb = mb % tl.cdiv(ROWS, BM)
    od = dslice * DV + tl.arange(0, DV)
    m = mb * BM + tl.arange(0, BM)
    qpos, qh = m // G, kh * G + m % G
    n = tl.arange(0, BN)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DV), tl.float32)
    begin = split * SLEN
    end = begin + SLEN
    for start in tl.range(begin, end, BN):
        ns = start + n
        block = tl.load(TABLE + ns // PAGE)
        inside = ns % PAGE
        scores = tl.zeros((BM, BN), tl.float32)
        for ki in range(D // BK):
            kd = ki * BK + tl.arange(0, BK)
            q = tl.load(
                Q + qpos[:, None] * HQ * D + qh[:, None] * D + kd[None, :],
                mask=(m < ROWS)[:, None],
                other=0,
            )
            k = tl.load(
                K
                + block[None, :] * K0
                + kh * K1
                + (kd[:, None] // 8) * K2
                + inside[None, :] * 8
                + kd[:, None] % 8
            )
            scores = tl.dot(q, k, scores)
        scores *= 1.4426950408889634 / 16.0
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(scores - new_max[:, None])
        acc = acc * alpha[:, None]
        v = tl.load(
            V + block[:, None] * V0 + kh * V1 + od[None, :] * V2 + inside[:, None]
        )
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_max
    if split == SPLITS - 1:
        for tail_start in range(tl.cdiv(NQ, BN)):
            ns = tail_start * BN + n
            scores = tl.zeros((BM, BN), tl.float32)
            for ki in range(D // BK):
                kd = ki * BK + tl.arange(0, BK)
                q = tl.load(
                    Q + qpos[:, None] * HQ * D + qh[:, None] * D + kd[None, :],
                    mask=(m < ROWS)[:, None],
                    other=0,
                )
                k = tl.load(
                    KD + ns[None, :] * HK * D + kh * D + kd[:, None],
                    mask=(ns < NQ)[None, :],
                    other=0,
                )
                scores = tl.dot(q, k, scores)
            scores *= 1.4426950408889634 / 16.0
            scores = tl.where(
                (ns < NQ)[None, :] & (ns[None, :] <= qpos[:, None]),
                scores,
                -float("inf"),
            )
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp2(maximum - new_max)
            p = tl.exp2(scores - new_max[:, None])
            acc = acc * alpha[:, None]
            v = tl.load(
                VD + ns[:, None] * HK * D + kh * D + od[None, :],
                mask=(ns < NQ)[:, None],
                other=0,
            )
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = new_max
    if SPLITS == 1:
        tl.store(
            OUT + qpos[:, None] * HQ * D + qh[:, None] * D + od[None, :],
            acc / denom[:, None],
            mask=(m < ROWS)[:, None],
        )
    else:
        base = ((split * HK + kh) * ROWS + m) * D
        tl.store(
            PART + base[:, None] + od[None, :],
            acc / denom[:, None],
            mask=(m < ROWS)[:, None],
        )
        if dslice == 0:
            tl.store(
                LSE + (split * HK + kh) * ROWS + m,
                maximum + tl.log2(denom),
                mask=m < ROWS,
            )


@triton.jit
def merge_attention(
    PART,
    LSE,
    OUT,
    NQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    m, kh = tl.program_id(0), tl.program_id(1)
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = NQ * G
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    lse = tl.load(LSE + (s * HK + kh) * ROWS + m)
    weights = tl.exp2(lse - tl.max(lse, 0))
    part = tl.load(PART + ((s * HK + kh) * ROWS + m)[:, None] * D + d[None, :])
    value = tl.sum(part * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + (m // G) * HQ * D + (kh * G + m % G) * D + d, value)


def make_call(data, config, context=8192):
    q, k, v = data["q"], data["kc"], data["vc"]
    nq, hq, dim = q.shape
    hk = k.shape[1]
    splits = config["splits"]
    assert nq > 0 and hq == 12 and hk == 2 and dim == 256
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert context % (splits * config["bn"]) == 0
    assert splits > 0 and splits & (splits - 1) == 0
    assert (
        q.is_contiguous() and data["kd"].is_contiguous() and data["vd"].is_contiguous()
    )
    assert data["table"].is_contiguous() and data["table"].shape[0] == 1
    assert k.stride(-1) == 1 and k.stride(-2) == 8 and v.stride(-1) == 1
    assert data["kd"].shape == data["vd"].shape == (nq, hk, dim)
    assert data["kd"].dtype == data["vd"].dtype == q.dtype
    assert context % config["bn"] == 0
    assert splits * config["bn"] <= context
    out = torch.empty_like(q)
    partial = torch.empty(
        (splits, hk, nq * hq // hk, dim),
        device=q.device,
        dtype=torch.bfloat16 if config.get("partial_bf16") else torch.float32,
    )
    lse = torch.empty(partial.shape[:-1], device=q.device, dtype=torch.float32)

    def stage():
        compiled = partial_attention[
            (
                triton.cdiv(nq * hq // hk, config["bm"])
                * (dim // config.get("dv", dim)),
                hk,
                splits,
            )
        ](
            q,
            k,
            v,
            data["kd"],
            data["vd"],
            data["table"],
            partial,
            lse,
            out,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k.shape[3],
            context,
            nq,
            hq,
            hk,
            dim,
            config["bm"],
            config["bn"],
            splits,
            config.get("dv", dim),
            config.get("bk", dim),
            num_warps=config["warps"],
            num_stages=config["stages"],
            waves_per_eu=config.get("waves", 2),
        )
        run.compiled = compiled
        run.resources = dict(
            shared=compiled.metadata.shared,
            registers=compiled.n_regs,
            spills=compiled.n_spills,
        )

    def reduce():
        if splits > 1:
            run.reduce_compiled = merge_attention[(nq * hq // hk, hk)](
                partial,
                lse,
                out,
                nq,
                hq,
                hk,
                dim,
                splits,
                num_warps=config.get("reduce_warps", 4),
            )

    def run():
        stage()
        reduce()

    run.stage, run.reduce = stage, reduce
    return run, out


def prepare_attention(query, key_cache, value_cache, key, value, block_table, context):
    """Allocate reusable workspace for one of the two validated BF16 shapes.

    Returns (run, output). Warm up run before graph capture. Tensor addresses
    remain fixed; tensor contents may change between calls or graph replays.
    HQ=12, HKV=2, D=256, one request, causal attention, scale=1/16.
    Cache: packed K [pages,2,32,784,8], V [pages,2,256,784].
    Current K/V are fresh dense tensors [Q,2,256].
    """
    shape = (query.shape[0], context)
    configs = {
        (2, 65536): dict(bm=16, bn=32, bk=256, splits=32, warps=4, stages=1),
        (32, 8192): dict(bm=32, bn=64, bk=64, splits=8, warps=4, stages=2),
    }
    if shape not in configs or key_cache.shape[3] != 784:
        raise ValueError(
            "This prototype supports Q2/C65536 and Q32/C8192 with page 784"
        )
    return make_call(
        dict(
            q=query, kc=key_cache, vc=value_cache, kd=key, vd=value, table=block_table
        ),
        configs[shape],
        context,
    )
