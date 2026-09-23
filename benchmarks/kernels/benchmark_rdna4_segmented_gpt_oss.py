# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare segmented GPT-OSS attention with AITER unified on RDNA4.

Run from the repository root with ``PYTHONPATH=$PWD``. Each backend receives
the same KV values in its native cache layout. Times include the complete
attention launch sequence inside a CUDA graph, but not KV-cache updates.
The optional tuning pass uses the production candidate search for sink layers.
"""

import argparse
import gc
import statistics

import torch

from vllm.v1.attention.ops.segmented_attention import segmented_attention
from vllm.v1.attention.ops.segmented_attention_tuning import (
    _TABLES,
    _key,
    _tune_workload,
)


def _median_graph_us(fn, calls: int, replays: int) -> float:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(elapsed)


def _measure(
    query_len: int,
    seq_len: int,
    window: int,
    fp8: bool,
    calls: int,
    replays: int,
    autotune: bool,
) -> None:
    from aiter.ops.triton.unified_attention import (
        unified_attention as aiter_unified_attention,
    )

    heads, kv_heads, dim, page = 16, 2, 64, 64
    blocks = (seq_len + page - 1) // page
    cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    query = torch.randn(
        query_len, heads, dim, device="cuda", dtype=torch.bfloat16
    ).mul_(0.25)
    key = torch.randn(blocks, page, kv_heads, dim, device="cuda").mul_(0.25)
    value = torch.randn_like(key).mul_(0.25)
    key = key.to(cache_dtype)
    value = value.to(cache_dtype)

    segmented_cache = torch.empty(
        blocks, 2, page, kv_heads * dim, device="cuda", dtype=cache_dtype
    )
    segmented_cache[:, 0].copy_(key.reshape(blocks, page, kv_heads * dim))
    segmented_cache[:, 1].copy_(value.reshape(blocks, page, kv_heads * dim))
    segmented_key = segmented_cache[:, 0].unflatten(-1, (kv_heads, dim))
    segmented_value = segmented_cache[:, 1].unflatten(-1, (kv_heads, dim))

    aiter_cache = torch.empty(
        blocks, kv_heads, page, 2 * dim, device="cuda", dtype=cache_dtype
    )
    aiter_cache[..., :dim].copy_(key.permute(0, 2, 1, 3))
    aiter_cache[..., dim:].copy_(value.permute(0, 2, 1, 3))
    aiter_key, aiter_value = aiter_cache.transpose(1, 2).split(dim, dim=-1)

    starts = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32)[None, :]
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    sinks = torch.linspace(-1, 1, heads, device="cuda", dtype=torch.float32)
    segmented_out = torch.empty_like(query)
    aiter_out = torch.empty_like(query)
    left = window - 1 if window else -1
    if autotune:
        span = min(seq_len, left + query_len) if window else seq_len
        record = _tune_workload(
            query.device,
            query.dtype,
            cache_dtype,
            heads,
            kv_heads,
            dim,
            page,
            dim**-0.5,
            query_len,
            (1, query_len, span),
            sliding_window=left,
            has_sinks=True,
            physical_seq_len=seq_len,
        )
        config = record["best"]
        table_key = _key(
            query.device,
            query.dtype,
            cache_dtype,
            heads,
            kv_heads,
            dim,
            page,
            dim**-0.5,
            left,
            True,
            True,
        )
        _TABLES[table_key] = {"records": [record]}
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"tuned_config={config},workload=(1,{query_len},{span})",
            flush=True,
        )

    def run_selected() -> None:
        segmented_attention(
            query,
            query[:, :kv_heads],
            query[:, :kv_heads],
            segmented_out,
            "fp8" if fp8 else "auto",
            segmented_key,
            segmented_value,
            table,
            starts,
            lengths,
            seq_len,
            query_len,
            scale,
            scale,
            dim**-0.5,
            sliding_window=left,
            sinks=sinks,
        )

    def run_aiter() -> None:
        aiter_unified_attention(
            q=query,
            k=aiter_key,
            v=aiter_value,
            out=aiter_out,
            cu_seqlens_q=starts,
            max_seqlen_q=query_len,
            seqused_k=lengths,
            max_seqlen_k=seq_len,
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(left, 0) if window else (-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None,
            k_descale=scale,
            v_descale=scale,
            sinks=sinks,
        )

    run_selected()
    run_aiter()
    torch.cuda.synchronize()
    difference = (segmented_out.float() - aiter_out.float()).abs()
    relative = difference.norm(dim=-1) / aiter_out.float().norm(dim=-1).clamp_min(1e-6)
    max_abs = difference.max().item()
    max_rel = relative.max().item()
    if max_abs > 0.02 or max_rel > 0.02:
        raise AssertionError(f"attention mismatch: abs={max_abs}, rel={max_rel}")

    selected_us = _median_graph_us(run_selected, calls, replays)
    aiter_us = _median_graph_us(run_aiter, calls, replays)
    print(
        f"{query_len},{seq_len},{window},{cache_dtype},{selected_us:.3f},"
        f"{aiter_us:.3f},{aiter_us / selected_us:.3f},{max_abs:.6f},"
        f"{max_rel:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-len", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--replays", type=int, default=7)
    args = parser.parse_args()
    if args.calls < 1 or args.replays < 1:
        parser.error("--calls and --replays must be positive")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise RuntimeError(f"benchmark requires gfx1200 or gfx1201, got {arch!r}")
    print(f"device={properties.name},arch={arch}")
    print("query,sequence,window,kv_dtype,selected_us,aiter_us,speedup,max_abs,max_rel")
    if args.suite:
        shapes = (
            (1, 8192, 0),
            (8, 8192, 0),
            (1, 131072, 0),
            (8, 131072, 0),
            (1, 131072, 128),
            (8, 131072, 128),
            (128, 8192, 0),
            (128, 131072, 0),
            (128, 8192, 128),
            (512, 8192, 128),
            (4096, 8192, 128),
        )
        for fp8 in (False, True):
            for query_len, seq_len, window in shapes:
                _measure(
                    query_len,
                    seq_len,
                    window,
                    fp8,
                    args.calls,
                    args.replays,
                    args.autotune,
                )
    else:
        _measure(
            args.query_len,
            args.seq_len,
            args.window,
            args.fp8,
            args.calls,
            args.replays,
            args.autotune,
        )


if __name__ == "__main__":
    main()
