# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Paged, grouped-query short prefill with bounded segmented workspace."""

from __future__ import annotations

from functools import lru_cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

MAX_QUERY_LEN = 128
MAX_SPLITS = 32


@triton.jit
def _segmented_prefill_stage(
    Q,
    K,
    V,
    KD,
    VD,
    TABLE,
    STARTS,
    LENS,
    KS,
    VS,
    PART,
    LSE,
    OUT,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    KD0: tl.constexpr,
    KD1: tl.constexpr,
    VD0: tl.constexpr,
    VD1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    V3: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    PAGE: tl.constexpr,
    PACK: tl.constexpr,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QCAP: tl.constexpr,
    MIN_Q: tl.constexpr,
    LIMIT_Q: tl.constexpr,
    SCALE: tl.constexpr,
    FP8: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    mb, item, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, kh = item // HK, item % HK
    first = tl.load(STARTS + seq)
    nq = tl.load(STARTS + seq + 1) - first
    if nq < MIN_Q or nq > LIMIT_Q:
        return
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = QCAP * G
    if mb * BM >= nq * G:
        return
    context = tl.load(LENS + seq) - nq
    m = mb * BM + tl.arange(0, BM)
    qpos, qh = m // G, kh * G + m % G
    n = tl.arange(0, BN)
    d = tl.arange(0, D)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    segment = tl.cdiv(context, SPLITS * BN) * BN
    begin = split * segment
    end = tl.minimum(begin + segment, context)
    k_scale = 1.0
    v_scale = 1.0
    if FP8:
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
    for tail in tl.static_range(2):
        if tail:
            loop_begin = tl.maximum(begin, (end // BN) * BN)
            loop_end = end
        else:
            loop_begin = begin
            loop_end = (end // BN) * BN
        for start in tl.range(loop_begin, loop_end, BN):
            ns = start + n
            valid = tl.full((BN,), True, tl.int1)
            if tail:
                valid = ns < end
            block = tl.load(
                TABLE + seq * T0 + (ns // PAGE) * T1, mask=valid, other=0
            ).to(tl.int64)
            inside = ns % PAGE
            scores = tl.zeros((BM, BN), tl.float32)
            for ki in range(D // BK):
                kd = ki * BK + tl.arange(0, BK)
                q = tl.load(
                    Q + (first + qpos[:, None]) * Q0 + qh[:, None] * Q1 + kd[None, :],
                    mask=(m < nq * G)[:, None],
                    other=0.0,
                )
                k = tl.load(
                    K
                    + block[None, :] * K0
                    + kh * K1
                    + (kd[:, None] // PACK) * K2
                    + inside[None, :] * K3
                    + kd[:, None] % PACK,
                    mask=valid[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, k.to(q.dtype), scores)
            scores *= SCALE * 1.4426950408889634 * k_scale
            scores = tl.where(valid[None, :], scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp2(maximum - new_max)
            p = tl.exp2(scores - new_max[:, None])
            acc *= alpha[:, None]
            v = tl.load(
                V
                + block[:, None] * V0
                + kh * V1
                + d[None, :] * V2
                + inside[:, None] * V3,
                mask=valid[:, None],
                other=0.0,
            )
            if FP8:
                acc += (
                    tl.dot(p.to(Q.dtype.element_ty), v.to(Q.dtype.element_ty)) * v_scale
                )
            else:
                acc = tl.dot(p.to(Q.dtype.element_ty), v, acc)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = new_max
    if split == SPLITS - 1:
        for start in range(tl.cdiv(nq, BN)):
            ns = start * BN + n
            scores = tl.zeros((BM, BN), tl.float32)
            for ki in range(D // BK):
                kd = ki * BK + tl.arange(0, BK)
                q = tl.load(
                    Q + (first + qpos[:, None]) * Q0 + qh[:, None] * Q1 + kd[None, :],
                    mask=(m < nq * G)[:, None],
                    other=0.0,
                )
                k = tl.load(
                    KD + (first + ns[None, :]) * KD0 + kh * KD1 + kd[:, None],
                    mask=(ns < nq)[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, k, scores)
            scores *= SCALE * 1.4426950408889634
            scores = tl.where(
                (ns < nq)[None, :] & (ns[None, :] <= qpos[:, None]),
                scores,
                -float("inf"),
            )
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            # Later current-chunk tiles may be entirely masked for an early row.
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            p = tl.exp2(scores - safe_max[:, None])
            acc *= alpha[:, None]
            v = tl.load(
                VD + (first + ns[:, None]) * VD0 + kh * VD1 + d[None, :],
                mask=(ns < nq)[:, None],
                other=0.0,
            )
            acc = tl.dot(p.to(Q.dtype.element_ty), v, acc)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = safe_max
    norm = tl.where(denom > 0.0, denom, 1.0)
    if SPLITS == 1:
        tl.store(
            OUT + (first + qpos[:, None]) * O0 + qh[:, None] * O1 + d[None, :],
            acc / norm[:, None],
            mask=(m < nq * G)[:, None],
        )
    else:
        base = (((split * BATCH + seq) * HK + kh) * ROWS + m) * D
        tl.store(
            PART + base[:, None] + d[None, :],
            acc / norm[:, None],
            mask=(m < nq * G)[:, None],
        )
        tl.store(
            LSE + ((split * BATCH + seq) * HK + kh) * ROWS + m,
            maximum + tl.log2(denom),
            mask=m < nq * G,
        )


@triton.jit
def _segmented_prefill_reduce(
    PART,
    LSE,
    OUT,
    STARTS,
    O0: tl.constexpr,
    O1: tl.constexpr,
    BATCH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QCAP: tl.constexpr,
    MIN_Q: tl.constexpr,
    LIMIT_Q: tl.constexpr,
    SPLITS: tl.constexpr,
):
    m, item = tl.program_id(0), tl.program_id(1)
    seq, kh = item // HK, item % HK
    first = tl.load(STARTS + seq)
    nq = tl.load(STARTS + seq + 1) - first
    G: tl.constexpr = HQ // HK
    ROWS: tl.constexpr = QCAP * G
    if nq < MIN_Q or nq > LIMIT_Q or m >= nq * G:
        return
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    lse = tl.load(LSE + ((s * BATCH + seq) * HK + kh) * ROWS + m)
    weights = tl.exp2(lse - tl.max(lse, 0))
    part = tl.load(
        PART + (((s * BATCH + seq) * HK + kh) * ROWS + m)[:, None] * D + d[None, :]
    )
    result = tl.sum(part * weights[:, None], 0) / tl.sum(weights, 0)
    tl.store(OUT + (first + m // G) * O0 + (kh * G + m % G) * O1 + d, result)


@lru_cache(maxsize=512)
def select_segmented_config(batch, max_query_len, max_seq_len, hq, hk, dim, fp8):
    """Choose bounded initial launch ranges; benchmark overrides remain explicit."""
    qcap = min(max_query_len, MAX_QUERY_LEN)
    if batch < 1 or qcap < 1:
        raise ValueError("Batch and query capacity must be positive")
    bm = 16 if qcap <= 2 else 32
    bn = 32 if qcap <= 2 else 64
    bk = dim if qcap <= 2 or dim == 128 else 64
    groups = batch * hk * triton.cdiv(qcap * (hq // hk), bm)
    target = 64 if qcap <= 2 else 96
    splits = min(MAX_SPLITS, triton.next_power_of_2(triton.cdiv(target, groups)))
    while splits > 1 and max_seq_len < splits * 128:
        splits //= 2
    return dict(
        bm=bm,
        bn=bn,
        bk=bk,
        splits=splits,
        warps=4,
        stages=2 if dim == 256 and qcap > 2 and not fp8 else 1,
    )


def segmented_workspace_shapes(batch, query_cap, hq, hk, dim, splits):
    if splits == 1:
        return None
    partial = (splits, batch, hk, query_cap * (hq // hk), dim)
    return partial, partial[:-1]


def reserve_segmented_prefill_workspace(max_batch, hq, hk, dim, max_seq_len):
    """Reserve the largest selected short-prefill workspace before graph capture."""
    if not is_workspace_manager_initialized():
        return
    largest = 0
    for batch in range(1, max_batch + 1):
        for qcap in range(1, MAX_QUERY_LEN + 1):
            cfg = select_segmented_config(batch, qcap, max_seq_len, hq, hk, dim, False)
            if cfg["splits"] > 1:
                largest = max(largest, cfg["splits"] * batch * qcap * hq)
    if largest:
        current_workspace_manager()._reserve_simultaneous(
            ((largest * dim,), torch.float32), ((largest,), torch.float32)
        )


def can_use_segmented_prefill(
    q, k, v, out, kc, vc, table, starts, lengths, k_scale, v_scale
):
    """Check tensor metadata; unsupported feature checks live in the caller."""
    if (
        k is None
        or v is None
        or q.ndim != 3
        or kc.ndim != 5
        or vc.ndim != 4
        or q.dtype not in (torch.bfloat16, torch.float16)
        or out.dtype != q.dtype
        or q.shape != out.shape
        or q.shape[-1] not in (128, 256)
        or kc.shape[1] < 1
        or q.shape[1] % kc.shape[1]
        or not 1 <= q.shape[1] // kc.shape[1] <= 16
        or kc.dtype != vc.dtype
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or kc.dtype not in (q.dtype, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or table.ndim != 2
        or starts.ndim != 1
        or lengths.ndim != 1
        or table.shape[0] != lengths.numel()
        or starts.numel() != lengths.numel() + 1
        or table.dtype != torch.int32
        or starts.dtype != torch.int32
        or lengths.dtype != torch.int32
        or starts.stride(0) != 1
        or lengths.stride(0) != 1
        or not q.is_cuda
    ):
        return False
    hkv, dim, page = kc.shape[1], q.shape[-1], kc.shape[3]
    pack = 16 // kc.element_size()
    if (
        k.shape != (q.shape[0], hkv, dim)
        or v.shape != k.shape
        or kc.shape[2:] != (dim // pack, page, pack)
        or vc.shape != (kc.shape[0], hkv, dim, page)
        or any(
            t.device != q.device for t in (k, v, out, kc, vc, table, starts, lengths)
        )
        or any(t.stride(-1) != 1 for t in (q, k, v, out, kc, vc))
    ):
        return False
    if kc.element_size() == 1:
        return all(
            isinstance(s, torch.Tensor)
            and s.numel() == 1
            and s.dtype == torch.float32
            and s.device == q.device
            for s in (k_scale, v_scale)
        )
    return True


def segmented_prefill_attention(
    q,
    k,
    v,
    out,
    kc,
    vc,
    table,
    starts,
    lengths,
    max_query_len,
    max_seq_len,
    k_scale,
    v_scale,
    scale,
    *,
    skip_decode=True,
    config=None,
    workspace=None,
):
    """Write eligible short-prefill rows; leave other requests untouched."""
    batch, hq, dim = lengths.numel(), q.shape[1], q.shape[2]
    hk = kc.shape[1]
    if batch == 0 or max_query_len == 0 or q.numel() == 0:
        return out
    qcap = min(max_query_len, MAX_QUERY_LEN)
    fp8 = kc.element_size() == 1
    cfg = config or select_segmented_config(batch, qcap, max_seq_len, hq, hk, dim, fp8)
    splits = cfg["splits"]
    shapes = segmented_workspace_shapes(batch, qcap, hq, hk, dim, splits)
    if shapes is None:
        partial = lse = out
    elif workspace is not None:
        partial, lse = workspace
    elif is_workspace_manager_initialized():
        partial, lse = current_workspace_manager().get_simultaneous(
            (shapes[0], torch.float32), (shapes[1], torch.float32)
        )
    else:
        partial = torch.empty(shapes[0], device=q.device, dtype=torch.float32)
        lse = torch.empty(shapes[1], device=q.device, dtype=torch.float32)
    minimum = 2 if skip_decode else 1
    _segmented_prefill_stage[
        (triton.cdiv(qcap * (hq // hk), cfg["bm"]), batch * hk, splits)
    ](
        q,
        kc,
        vc,
        k,
        v,
        table,
        starts,
        lengths,
        k_scale,
        v_scale,
        partial,
        lse,
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        *kc.stride()[:4],
        *vc.stride(),
        *table.stride(),
        kc.shape[3],
        kc.shape[4],
        batch,
        hq,
        hk,
        dim,
        qcap,
        minimum,
        MAX_QUERY_LEN,
        scale,
        fp8,
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        splits,
        num_warps=cfg["warps"],
        num_stages=cfg["stages"],
        waves_per_eu=2,
    )
    if splits > 1:
        _segmented_prefill_reduce[(qcap * (hq // hk), batch * hk)](
            partial,
            lse,
            out,
            starts,
            out.stride(0),
            out.stride(1),
            batch,
            hq,
            hk,
            dim,
            qcap,
            minimum,
            MAX_QUERY_LEN,
            splits,
            num_warps=4,
        )
    return out
