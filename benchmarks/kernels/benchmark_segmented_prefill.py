# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched RDNA4 short-prefill shape-range benchmark and reference fixtures."""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from benchmark_rocm_splitkv_paged_decode import _padded_cache_views
from rdna4_prefill_prototype.eviction import ReadEviction

from vllm import envs
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd
from vllm.v1.attention.ops.segmented_prefill import (
    MAX_QUERY_LEN,
    segmented_prefill_attention,
    segmented_query_capacity,
    segmented_workspace_shapes,
    select_segmented_config,
)


def measure(graph, flush, cold, samples=15):
    pairs = []
    for _ in range(samples):
        if cold:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        pairs.append((start, end))
    pairs[-1][1].synchronize()
    return statistics.median(a.elapsed_time(b) * 1000 for a, b in pairs)


def make_inputs(
    queries,
    contexts,
    *,
    heads=12,
    kv_heads=2,
    dim=256,
    page=784,
    fp8=False,
    dtype=torch.bfloat16,
    seed=419,
    matched_current=True,
    legacy_layout=True,
):
    torch.manual_seed(seed)
    batch, total = len(queries), sum(queries)
    blocks = max(1, math.ceil(max(q + c for q, c in zip(queries, contexts)) / page))
    raw = (torch.randn(total, heads + 2 * kv_heads, dim, device="cuda") * 0.25).to(
        dtype
    )
    q, k, v = (
        raw[:, :heads],
        raw[:, heads : heads + kv_heads],
        raw[:, heads + kv_heads :],
    )
    cache_dtype = torch.float8_e4m3fn if fp8 else dtype
    ks = torch.tensor(0.125 if fp8 else 1.0, device="cuda")
    vs = torch.tensor(0.25 if fp8 else 1.0, device="cuda")
    cache_shape = (batch * blocks, page, kv_heads, dim)

    def fill_cache(cache, scale):
        block_elements = math.prod(cache_shape[1:])
        chunk_blocks = max(1, (64 * 1024**2) // block_elements)
        for start in range(0, cache_shape[0], chunk_blocks):
            chunk = cache[start : start + chunk_blocks]
            values = torch.randn(chunk.shape, device="cuda") * 0.25 / scale
            chunk.copy_(values)

    if legacy_layout:
        kn = torch.empty(cache_shape, device="cuda", dtype=cache_dtype)
        vn = torch.empty_like(kn)
    else:
        page_elements = math.prod(cache_shape[1:])
        backing = torch.empty(
            cache_shape[0] * 2 * page_elements,
            device="cuda",
            dtype=cache_dtype,
        )
        strides = (2 * page_elements, kv_heads * dim, dim, 1)
        kn = torch.as_strided(backing, cache_shape, strides)
        vn = torch.as_strided(backing, cache_shape, strides, page_elements)
    fill_cache(kn, ks)
    fill_cache(vn, vs)
    table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int32).view(
        batch, blocks
    )
    first = 0
    for seq, (n, c) in enumerate(zip(queries, contexts)):
        positions = c + torch.arange(n, device="cuda")
        physical = table[seq, positions // page].long()
        offsets = positions % page
        kn[physical, offsets] = (k[first : first + n].float() / ks).to(cache_dtype)
        vn[physical, offsets] = (v[first : first + n].float() / vs).to(cache_dtype)
        if matched_current:
            k[first : first + n].copy_((kn[physical, offsets].float() * ks).to(dtype))
            v[first : first + n].copy_((vn[physical, offsets].float() * vs).to(dtype))
        first += n
    pack = 16 // kn.element_size()
    if legacy_layout:
        kc = (
            kn.reshape(batch * blocks, page, kv_heads, dim // pack, pack)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        vc = vn.permute(0, 2, 3, 1).contiguous()
        kc, vc = _padded_cache_views(kc, vc)
    else:
        kc = vc = None
    if legacy_layout:
        kn, vn = _padded_cache_views(kn, vn)
    starts = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    lens = torch.tensor(
        [q + c for q, c in zip(queries, contexts)], device="cuda", dtype=torch.int32
    )
    return dict(
        q=q,
        k=k,
        v=v,
        kc=kc,
        vc=vc,
        kn=kn,
        vn=vn,
        table=table,
        starts=starts,
        lens=lens,
        ks=ks,
        vs=vs,
        queries=queries,
        contexts=contexts,
    )


def reference(data):
    refs = []
    first = 0
    hq, dim = data["q"].shape[1:]
    hk = data["kn"].shape[2]
    for seq, (n, c) in enumerate(zip(data["queries"], data["contexts"])):
        ids = data["table"][seq].long()
        k = data["kn"][ids].reshape(-1, hk, dim)[:c].float() * data["ks"]
        v = data["vn"][ids].reshape(-1, hk, dim)[:c].float() * data["vs"]
        k = torch.cat((k, data["k"][first : first + n].float())).repeat_interleave(
            hq // hk, 1
        )
        v = torch.cat((v, data["v"][first : first + n].float())).repeat_interleave(
            hq // hk, 1
        )
        scores = torch.einsum(
            "qhd,khd->hqk", data["q"][first : first + n].float(), k
        ) / math.sqrt(dim)
        mask = (
            torch.arange(c + n, device="cuda")[None, :]
            > c + torch.arange(n, device="cuda")[:, None]
        )
        scores.masked_fill_(mask[None], -float("inf"))
        refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        first += n
    return torch.cat(refs)


def make_call(data, backend, config=None):
    q, k, v = data["q"], data["k"], data["v"]
    out = torch.empty_like(q)
    maxq, maxs = (
        max(data["queries"]),
        max(q + c for q, c in zip(data["queries"], data["contexts"])),
    )
    hq, dim = q.shape[1:]
    hk = data["kn"].shape[2]
    batch = len(data["queries"])
    scale = dim**-0.5
    fp8 = data["kn"].element_size() == 1
    selected = {}
    if backend == "segmented":
        cfg = config or select_segmented_config(
            batch, min(maxq, MAX_QUERY_LEN), maxs, hq, hk, dim, fp8
        )
        query_capacity = segmented_query_capacity(min(maxq, MAX_QUERY_LEN))
        shapes = segmented_workspace_shapes(
            batch, query_capacity, hq, hk, dim, cfg["splits"]
        )
        workspace = (
            None
            if shapes is None
            else tuple(
                torch.empty(s, device=q.device, dtype=torch.float32) for s in shapes
            )
        )

        def run():
            segmented_prefill_attention(
                q,
                out,
                data["kn"],
                data["vn"],
                data["table"],
                data["starts"],
                data["lens"],
                maxq,
                maxs,
                data["ks"],
                data["vs"],
                scale,
                skip_decode=False,
                config=cfg,
                workspace=workspace,
            )

        selected = cfg
    elif backend in ("fly", "flywide", "flystream"):
        if backend == "flystream":
            from rdna4_prefill_prototype.fly_prefill_streamq import (
                make_call as fly_call,
            )
        elif backend == "flywide":
            from rdna4_prefill_prototype.fly_prefill_wide import make_call as fly_call
        else:
            from rdna4_prefill_prototype.fly_prefill import make_call as fly_call
        run, out = fly_call(data, (config or {}).get("splits", 8))
    elif backend.startswith("aiter"):
        from aiter.ops.triton.attention import unified_attention as aiter

        original3d = aiter.select_3d_config
        original2d = aiter.select_2d_config
        qs = torch.tensor(0.125, device=q.device) if backend == "aiter_fp8q" else None
        aq = (q.float() / qs).to(torch.float8_e4m3fn) if qs is not None else q

        def choose3d(*args, **kw):
            attn, reduce = original3d(*args, **kw)
            attn, reduce = dict(attn), dict(reduce)
            if config:
                for target in (attn, reduce):
                    if "splits" in config:
                        target["NUM_SEGMENTS_PER_SEQ"] = config["splits"]
                for key in ("TILE_SIZE", "num_warps", "num_stages"):
                    if key in config:
                        attn[key] = config[key]
            selected.update(kind="3d", attention=attn, reduce=reduce)
            return attn, reduce

        def choose2d(*args, **kw):
            result = dict(original2d(*args, **kw))
            if config:
                for key in ("TILE_SIZE", "num_warps", "num_stages"):
                    if key in config:
                        result[key] = config[key]
            selected.update(kind="2d", attention=result)
            return result

        def run():
            with ExitStack() as stack:
                stack.enter_context(patch.object(aiter, "select_3d_config", choose3d))
                stack.enter_context(patch.object(aiter, "select_2d_config", choose2d))
                if config and "force3d" in config:
                    stack.enter_context(
                        patch.object(
                            aiter,
                            "use_2d_kernel",
                            lambda *a, **k: not config["force3d"],
                        )
                    )
                aiter.unified_attention(
                    aq,
                    data["kn"],
                    data["vn"],
                    out,
                    data["starts"],
                    maxq,
                    data["lens"],
                    maxs,
                    scale,
                    True,
                    (-1, -1),
                    data["table"],
                    0,
                    qs,
                    data["ks"],
                    data["vs"],
                )
    else:
        if backend == "context" and config is None:
            from vllm.v1.attention.ops import prefix_prefill_tuning as tuning

            cache_dir = Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention"
            expected_sha = None
            if data.get("context_source"):
                expected_sha = hashlib.sha256(
                    (
                        Path(data["context_source"])
                        / "vllm/v1/attention/ops/prefix_prefill.py"
                    ).read_bytes()
                ).hexdigest()
            for cache in sorted(cache_dir.glob("*.json")):
                saved = json.loads(cache.read_text())
                identity = saved["identity"]
                if expected_sha and identity["kernel_sha256"] != expected_sha:
                    continue
                if all(
                    identity[key] == value
                    for key, value in {
                        "heads": hq,
                        "kv_heads": hk,
                        "dim": dim,
                        "page": data["kc"].shape[3],
                        "kv_dtype": str(data["kc"].dtype),
                    }.items()
                ):
                    tuning._TABLES[
                        tuning._key(
                            q.device,
                            hq,
                            hk,
                            dim,
                            data["kc"].shape[3],
                            scale,
                            data["kc"].dtype,
                        )
                    ] = saved
                    config = tuning.get_context_attention_config(
                        q.device,
                        hq,
                        hk,
                        dim,
                        data["kc"].shape[3],
                        batch,
                        maxq,
                        maxs,
                        scale,
                        data["kc"].dtype,
                    )
                    selected.update(
                        cache_file=str(cache), cache_identity=identity, launch=config
                    )
                    break
            if config is None:
                config = {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "num_unroll_cache": 4,
                    "num_unroll_request": 1,
                    "num_warps": 4,
                    "num_stages": 1,
                }
                selected.update(launch=config)

        def run():
            context_attention_fwd(
                q,
                k,
                v,
                out,
                "fp8" if fp8 else "auto",
                data["kc"],
                data["vc"],
                data["table"],
                data["starts"],
                data["lens"],
                maxs,
                maxq,
                data["ks"],
                data["vs"],
                sm_scale=scale,
                skip_decode=False,
                _launch_config=config,
            )

    return run, out, selected


def run_case(case, backends, samples=15, configs=None, context_source=None):
    data = make_inputs(**case)
    data["context_source"] = context_source
    ref = reference(data)
    read = ReadEviction()
    write = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    records = []
    for backend in backends:
        for cfg in configs or [None]:
            row = dict(case=case, backend=backend, config=cfg)
            from vllm.v1.attention.ops.segmented_prefill import _segmented_prefill_stage

            row["segmented_source_sha256"] = hashlib.sha256(
                _segmented_prefill_stage.src.encode()
            ).hexdigest()
            try:
                run, out, selected = make_call(data, backend, cfg)
                run()
                torch.cuda.synchronize()
                err = (
                    (
                        (out.float() - ref).norm(dim=-1)
                        / ref.norm(dim=-1).clamp_min(1e-8)
                    )
                    .max()
                    .item()
                )
                assert torch.isfinite(out).all(), err
                row["passes_bf16_reference_gate"] = err < 0.01
                if backend != "aiter_fp8q":
                    assert err < 0.01, err
                for _ in range(10):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for _ in range(10):
                    graph.replay()
                row.update(
                    selected=selected,
                    error=err,
                    read_us=measure(graph, read, True, samples),
                    write_us=measure(graph, write, True, samples),
                    reuse_us=measure(graph, read, False, samples),
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                row["failure"] = str(exc)
            records.append(row)
            print(json.dumps(row), flush=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=Path)
    p.add_argument("--configs", type=Path)
    p.add_argument("--backends", default="segmented,aiter")
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--context-source", type=str)
    args = p.parse_args()
    cases = (
        json.loads(args.cases.read_text())
        if args.cases
        else [
            dict(queries=[q] * b, contexts=[c] * b, fp8=fp8, page=1568 if fp8 else 784)
            for q, c, b, fp8 in itertools.product(
                (2, 8, 32, 64, 128), (0, 512, 8192), (1, 4), (False, True)
            )
        ]
    )
    configs = json.loads(args.configs.read_text()) if args.configs else None
    records = []
    for case in cases:
        records.extend(
            run_case(
                case,
                args.backends.split(","),
                args.samples,
                configs,
                args.context_source,
            )
        )
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
