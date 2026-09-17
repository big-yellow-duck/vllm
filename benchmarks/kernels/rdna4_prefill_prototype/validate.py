# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Held-out correctness and repeated matched timing for the selected prototypes."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from eviction import ReadEviction
from fixtures import inputs
from kernel import make_call
from tune import error, measure, reference

Q2_CONFIG = dict(bm=16, bn=32, bk=256, splits=32, warps=4, stages=1)
Q32_CONFIG = dict(bm=32, bn=64, bk=64, splits=8, warps=4, stages=2)


def correctness(nq, context, config):
    data = inputs([nq], [context], "bf16", 784)
    data["context"] = context
    run, out = make_call(data, config, context)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    results = []
    for case, (seed, scale, permute, zeroq) in enumerate(
        [
            (31, 0.25, False, False),
            (79, 1.0, False, False),
            (103, 3.0, True, False),
            (211, 1.0, True, True),
            (433, 0.25, True, False),
        ]
    ):
        torch.manual_seed(seed)
        for name in ("q", "kc", "vc", "kd", "vd"):
            data[name].copy_(torch.randn_like(data[name]) * scale)
        if zeroq:
            data["q"].zero_()
        if permute:
            data["table"][0].copy_(torch.randperm(data["table"].numel(), device="cuda"))
        else:
            data["table"][0].copy_(torch.arange(data["table"].numel(), device="cuda"))
        # Fresh current values deliberately differ substantially from the cache.
        data["kd"].mul_(2.0)
        data["vd"].add_(2.0)
        # Poison the cached current chunk and padding after the prefix.
        for logical in range(context, data["table"].numel() * 784):
            page, offset = divmod(logical, 784)
            if offset == context % 784 or offset == 0:
                physical = int(data["table"][0, page].item())
                start = offset
                data["kc"][physical, :, :, start:, :].fill_(float("nan"))
                data["vc"][physical, :, :, start:].fill_(float("nan"))
                break
        graph.replay()
        torch.cuda.synchronize()
        ref = reference(data)
        e = error(out, ref)
        assert torch.isfinite(out).all() and e < 0.01, (case, e)
        initial = out.clone()
        # Replay the same graph after changing dense current values and Q.
        data["q"].mul_(0.875)
        data["vd"].add_(3.0)
        graph.replay()
        torch.cuda.synchronize()
        e2 = error(out, reference(data))
        assert torch.isfinite(out).all() and e2 < 0.01, (case, e2)
        assert not torch.equal(initial, out), "Graph did not consume changed inputs"
        row = dict(
            seed=seed,
            scale=scale,
            permuted_table=permute,
            zero_query=zeroq,
            poisoned_cached_tail=True,
            initial_max_head_l2=e,
            replay_max_head_l2=e2,
        )
        results.append(row)
        print("VALID", json.dumps(row), flush=True)
    return results


def baseline(data, nq, context, backend):
    from aiter.ops.triton.attention import unified_attention as aiter

    from vllm import envs
    from vllm.v1.attention.ops import prefix_prefill_tuning as tuning
    from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

    out = torch.empty_like(data["q"])
    selected = {}
    if backend.startswith("aiter"):

        def invoke():
            aiter.unified_attention(
                data["q"],
                data["kn"],
                data["vn"],
                out,
                data["starts"],
                nq,
                data["lens"],
                context + nq,
                0.0625,
                True,
                (-1, -1),
                data["table"],
                0,
                None,
                data["ks"],
                data["vs"],
            )

        if backend == "aiter_32":
            from unittest.mock import patch

            original = aiter.select_3d_config

            def choose(*args, **kwargs):
                return tuple(
                    dict(c, NUM_SEGMENTS_PER_SEQ=32) for c in original(*args, **kwargs)
                )

            def run():
                with patch.object(aiter, "select_3d_config", choose):
                    invoke()

            selected = dict(isolated_segments=32)
        else:
            run = invoke
    else:
        identity = tuning._identity(
            torch.device("cuda:0"), 12, 2, 256, 784, 0.0625, data["kc"].dtype
        )
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        cache = (
            Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention" / (digest + ".json")
        )
        if cache.exists():
            saved = json.loads(cache.read_text())
            assert saved["identity"] == identity
            tuning._TABLES[
                tuning._key(
                    torch.device("cuda:0"), 12, 2, 256, 784, 0.0625, data["kc"].dtype
                )
            ] = saved
        config = tuning.get_context_attention_config(
            torch.device("cuda:0"),
            12,
            2,
            256,
            784,
            1,
            nq,
            context + nq,
            0.0625,
            data["kc"].dtype,
        )
        selected = dict(
            config=config, cache_file=str(cache), cache_hit=config is not None
        )

        if backend == "rocm_attn_tuned":
            config = dict(
                BLOCK_M=16,
                BLOCK_N=64,
                num_unroll_cache=1,
                num_unroll_request=1,
                num_warps=4,
                num_stages=1,
            )
            selected.update(config=config, isolated_launch_override=True)

        def run():
            context_attention_fwd(
                data["q"],
                data["kd"],
                data["vd"],
                out,
                "auto",
                data["kc"],
                data["vc"],
                data["table"],
                data["starts"],
                data["lens"],
                context + nq,
                nq,
                data["ks"],
                data["vs"],
                sm_scale=0.0625,
                skip_decode=True,
                _launch_config=config or tuning._DEFAULT,
            )

    return run, out, selected


def compare(nq, context, config, rounds, samples):
    data = inputs([nq], [context], "bf16", 784)
    data["context"] = context
    ref = reference(data)
    read = ReadEviction()
    write = torch.zeros(256 * 1024**2, device="cuda", dtype=torch.int8)
    entries = []
    graphs = []
    keepers = []
    backends = ("prototype", "aiter", "rocm_attn")
    if nq == 2:
        backends += ("aiter_32", "rocm_attn_tuned")
    for name in backends:
        if name == "prototype":
            run, out = make_call(data, config, context)
            selected = dict(config=config)
        else:
            run, out, selected = baseline(data, nq, context, name)
        run()
        torch.cuda.synchronize()
        e = error(out, ref)
        assert e < 0.01 and torch.isfinite(out).all(), (name, e)
        for _ in range(20):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(20):
            graph.replay()
        if name == "prototype":
            selected["resources"] = run.resources
        entries.append(dict(backend=name, selected=selected, error=e, rounds=[]))
        graphs.append(graph)
        keepers.append((run, out))
    for rnd in range(rounds):
        order = (
            list(range(len(graphs)))
            if rnd % 2 == 0
            else list(reversed(range(len(graphs))))
        )
        for ix in order:
            graph = graphs[ix]
            row = dict(
                round=rnd,
                read_eviction_us=measure(graph, read, True, samples),
                write_eviction_us=measure(graph, write, True, samples),
                reuse_us=measure(graph, read, False, samples),
            )
            graph.replay()
            torch.cuda.synchronize()
            row["post_timing_error"] = error(keepers[ix][1], ref)
            assert (
                row["post_timing_error"] < 0.01 and torch.isfinite(keepers[ix][1]).all()
            ), row
            entries[ix]["rounds"].append(row)
            print("TIMING", entries[ix]["backend"], json.dumps(row), flush=True)
    ideal = (
        max(
            4 * 12 * 256 * (nq * context + nq * (nq + 1) // 2) / 191e12,
            (nq * 12 * 256 * 4 + 2 * (context + nq) * 2 * 256 * 2) / 640e9,
        )
        * 1e6
    )
    for entry in entries:
        for key in ("read_eviction_us", "write_eviction_us", "reuse_us"):
            entry[key] = statistics.median(r[key] for r in entry["rounds"])
        entry["ideal_roof_pct"] = ideal / entry["read_eviction_us"] * 100
    return dict(
        queries=nq,
        context=context,
        ideal_us=ideal,
        target_80_us=ideal / 0.8,
        results=entries,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queries", type=int, default=2)
    p.add_argument("--context", type=int, default=65536)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--skip-correctness", action="store_true")
    args = p.parse_args()
    config = Q2_CONFIG if args.queries == 2 else Q32_CONFIG
    report = dict(queries=args.queries, context=args.context, config=config)
    if not args.skip_correctness:
        report["validation"] = correctness(args.queries, args.context, config)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["comparison"] = compare(
        args.queries, args.context, config, args.rounds, args.samples
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
