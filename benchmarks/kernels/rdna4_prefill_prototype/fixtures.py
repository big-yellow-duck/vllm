# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic, matched BF16 fixtures with interleaved paged-cache backing."""

import torch


def padded_cache_views(key, value):
    blocks, page_elements = key.shape[0], key[0].numel()
    backing = torch.empty(
        blocks * 2 * page_elements, dtype=key.dtype, device=key.device
    )
    k = torch.as_strided(backing, key.shape, (2 * page_elements, *key.stride()[1:]))
    v = torch.as_strided(
        backing, value.shape, (2 * page_elements, *value.stride()[1:]), page_elements
    )
    k.copy_(key)
    v.copy_(value)
    return k, v


def inputs(queries, contexts, dtype="bf16", page=784):
    assert dtype == "bf16"
    torch.manual_seed(20260915)
    batch = len(queries)
    lengths = [q + c for q, c in zip(queries, contexts)]
    blocks = (max(lengths) + page - 1) // page

    def rand(*shape):
        return (torch.randn(*shape, device="cuda") * 0.25).to(torch.bfloat16)

    q = rand(sum(queries), 12, 256)
    kn, vn = [rand(batch, blocks * page, 2, 256) for _ in range(2)]
    ks = torch.tensor(1.0, device="cuda")
    vs = torch.tensor(1.0, device="cuda")
    kd = torch.cat(
        [kn[b, c : c + n] for b, (n, c) in enumerate(zip(queries, contexts))]
    )
    vd = torch.cat(
        [vn[b, c : c + n] for b, (n, c) in enumerate(zip(queries, contexts))]
    )
    kn = kn.reshape(batch * blocks, page, 2, 256)
    vn = vn.reshape_as(kn)
    kc = kn.reshape(batch * blocks, page, 2, 32, 8).permute(0, 2, 3, 1, 4).contiguous()
    vc = vn.permute(0, 2, 3, 1).contiguous()
    kc, vc = padded_cache_views(kc, vc)
    kn, vn = padded_cache_views(kn, vn)
    table = torch.arange(batch * blocks, device="cuda", dtype=torch.int32).reshape(
        batch, blocks
    )
    starts = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        kn=kn,
        vn=vn,
        kc=kc,
        vc=vc,
        kd=kd,
        vd=vd,
        ks=ks,
        vs=vs,
        table=table,
        starts=starts,
        lens=lens,
    )
