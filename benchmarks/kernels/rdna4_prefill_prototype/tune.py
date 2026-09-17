# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-shape search; compile/correctness/eviction are outside GPU timing."""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
from fixtures import inputs
from kernel import make_call


def reference(data):
    q = data["q"].float()
    context = data.get("context", 8192)
    nq, hq, dim = q.shape
    table = data["table"][0].long()
    kc, vc = data["kc"], data["vc"]
    hk = kc.shape[1]
    k = (
        kc.permute(0, 3, 1, 2, 4)
        .reshape(kc.shape[0], kc.shape[3], hk, dim)[table]
        .reshape(-1, hk, dim)[:context]
        .float()
    )
    v = vc.permute(0, 3, 1, 2)[table].reshape(-1, hk, dim)[:context].float()
    k = torch.cat([k, data["kd"].float()]).repeat_interleave(hq // hk, dim=1)
    v = torch.cat([v, data["vd"].float()]).repeat_interleave(hq // hk, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q, k) / dim**0.5
    mask = (
        torch.arange(context + nq, device=q.device)[None, :]
        > context + torch.arange(nq, device=q.device)[:, None]
    )
    logits.masked_fill_(mask[None], -float("inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)


def error(out, ref):
    return ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()


def measure(graph, flush, cold, samples=12, repeat=1):
    pairs = []
    for _ in range(samples):
        if cold:
            flush.zero_()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(repeat):
            graph.replay()
        end.record()
        pairs.append((start, end))
    pairs[-1][1].synchronize()
    return statistics.median(a.elapsed_time(b) * 1000 / repeat for a, b in pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=32)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=json.loads)
    p.add_argument("--eviction", choices=("write", "read"), default="write")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--config-file", type=Path)
    args = p.parse_args()
    data = inputs([args.queries], [args.context], "bf16", 784)
    data["context"] = args.context
    useful_flops = (
        4
        * 12
        * 256
        * (args.queries * args.context + args.queries * (args.queries + 1) // 2)
    )
    minimum_bytes = (
        args.queries * 12 * 256 * 4 + 2 * (args.context + args.queries) * 2 * 256 * 2
    )
    ideal_us = max(useful_flops / 191e12, minimum_bytes / 640e9) * 1e6
    ref = reference(data)
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    if args.eviction == "read":
        from eviction import ReadEviction

        flush = ReadEviction()
    configs = (
        [args.config]
        if args.config
        else [
            dict(bm=m, bn=n, splits=s, warps=w, stages=st)
            for m, n, s, w, st in itertools.product(
                (16, 32, 64), (32, 64), (4, 8, 16), (4, 8), (1,)
            )
        ]
    )
    if args.limit:
        configs = configs[: args.limit]
    if args.config_file:
        configs = json.loads(args.config_file.read_text())
    records = []
    for config in configs:
        start = time.monotonic()
        record = dict(
            config=config,
            eviction=args.eviction,
            queries=args.queries,
            context=args.context,
            ideal_us=ideal_us,
        )
        try:
            maker = make_call
            if config.get("impl") == "decomposed":
                from experiments.decomposed import make_call as maker
            if config.get("impl") == "page":
                from experiments.page_kernel import make_call as maker
            if config.get("impl") == "gluon":
                from experiments.gluon_kernel import make_call as maker
            run, out = maker(data, config, context=args.context)
            run()
            torch.cuda.synchronize()
            record["error"] = error(out, ref)
            assert record["error"] < 0.01 and torch.isfinite(out).all(), record["error"]
            for _ in range(8):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            record["cold_us"] = measure(graph, flush, True)
            record["reuse_us"] = measure(graph, flush, False)
            record["resources"] = run.resources
            if args.config:
                for name in (
                    ("qk", "softmax", "pv", "reduce")
                    if config.get("impl") == "decomposed"
                    else ("stage", "reduce")
                ):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        getattr(run, name)()
                    record[name + "_reuse_us"] = measure(g, flush, False)
            record["roof_pct"] = ideal_us / record["cold_us"] * 100
        except Exception as exc:
            record["failure"] = str(exc)
        record["wall_seconds"] = time.monotonic() - start
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    good = [r for r in records if "cold_us" in r]
    print("BEST", json.dumps(sorted(good, key=lambda r: r["cold_us"])[:5]), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
