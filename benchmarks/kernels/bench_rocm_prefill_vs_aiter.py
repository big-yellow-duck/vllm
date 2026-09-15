# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare saved ROCm prefill workloads with TPS AITER using identical KV data."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from aiter.ops.triton.attention import unified_attention as aiter_attention
from aiter.ops.triton.utils.types import e4m3_dtype

import vllm.envs as envs
from vllm.v1.attention.ops import prefix_prefill_tuning as tuning
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def make_inputs(device, batch, query, context, dtype):
    heads, kv_heads, dim, page = 12, 2, 256, 784
    total = query + context
    blocks = (total + page - 1) // page
    generator = torch.Generator(device=device).manual_seed(20260915)

    def random(*shape):
        return (torch.randn(*shape, device=device, generator=generator) * 0.25).to(
            torch.bfloat16
        )

    q = random(batch * query, heads, dim)
    k_nhd = random(batch, blocks, page, kv_heads, dim)
    v_nhd = random(batch, blocks, page, kv_heads, dim)
    ks = torch.full((), 0.125 if dtype == "fp8" else 1.0, device=device)
    vs = torch.full((), 0.25 if dtype == "fp8" else 1.0, device=device)
    if dtype == "fp8":
        k_nhd = (k_nhd.float() / ks).to(e4m3_dtype)
        v_nhd = (v_nhd.float() / vs).to(e4m3_dtype)
    k_full = k_nhd.reshape(batch, blocks * page, kv_heads, dim)
    v_full = v_nhd.reshape(batch, blocks * page, kv_heads, dim)
    # Both algorithms see the same effective KV, including current-chunk FP8.
    k_dense = (
        (k_full[:, context:total].float() * ks)
        .to(torch.bfloat16)
        .reshape(batch * query, kv_heads, dim)
    )
    v_dense = (
        (v_full[:, context:total].float() * vs)
        .to(torch.bfloat16)
        .reshape(batch * query, kv_heads, dim)
    )
    x = 16 if dtype == "fp8" else 8
    kc = (
        k_nhd.reshape(batch * blocks, page, kv_heads, dim // x, x)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    vc = (
        v_nhd.reshape(batch * blocks, page, kv_heads, dim)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    table = torch.arange(batch * blocks, device=device, dtype=torch.int32).view(
        batch, blocks
    )
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * query
    lengths = torch.full((batch,), total, device=device, dtype=torch.int32)
    return {
        "q": q,
        "k_dense": k_dense,
        "v_dense": v_dense,
        "kc": kc,
        "vc": vc,
        "k_nhd": k_nhd.reshape(batch * blocks, page, kv_heads, dim),
        "v_nhd": v_nhd.reshape(batch * blocks, page, kv_heads, dim),
        "table": table,
        "starts": starts,
        "lengths": lengths,
        "ks": ks,
        "vs": vs,
    }


def calls(data, query, context, dtype, baseline, ours, aiter_output):
    def rocm(output, config=None):
        context_attention_fwd(
            data["q"],
            data["k_dense"],
            data["v_dense"],
            output,
            "fp8_e4m3" if dtype == "fp8" else "auto",
            data["kc"],
            data["vc"],
            data["table"],
            data["starts"],
            data["lengths"],
            query + context,
            query,
            data["ks"],
            data["vs"],
            sm_scale=0.0625,
            skip_decode=True,
            _launch_config=config,
        )

    def aiter_call():
        aiter_attention.unified_attention(
            q=data["q"],
            k=data["k_nhd"],
            v=data["v_nhd"],
            out=aiter_output,
            cu_seqlens_q=data["starts"],
            max_seqlen_q=query,
            seqused_k=data["lengths"],
            max_seqlen_k=query + context,
            softmax_scale=0.0625,
            causal=True,
            window_size=(-1, -1),
            block_table=data["table"],
            softcap=0,
            q_descale=None,
            k_descale=data["ks"],
            v_descale=data["vs"],
        )

    return {
        "baseline": lambda: rocm(baseline, tuning._DEFAULT),
        "rocm_autotune": lambda: rocm(ours),
        "aiter_unified": aiter_call,
    }


@torch.inference_mode()
def reference_errors(data, outputs, batch, query, context):
    rows = sorted({0, min(31, query - 1), min(32, query - 1), query // 2, query - 1})
    total = query + context
    errors = {name: 0.0 for name in outputs}
    for b in range(batch):
        first = int(data["table"][b, 0].item())
        count = data["table"].shape[1]
        k = (
            data["k_nhd"][first : first + count].reshape(-1, 2, 256)[:total].float()
            * data["ks"]
        ).repeat_interleave(6, dim=1)
        v = (
            data["v_nhd"][first : first + count].reshape(-1, 2, 256)[:total].float()
            * data["vs"]
        ).repeat_interleave(6, dim=1)
        indices = [b * query + r for r in rows]
        logits = (
            torch.bmm(data["q"][indices].transpose(0, 1).float(), k.permute(1, 2, 0))
            * 0.0625
        )
        mask = torch.arange(total, device=k.device)[None, :] > (
            torch.tensor(rows, device=k.device)[:, None] + context
        )
        logits.masked_fill_(mask[None], float("-inf"))
        ref = torch.bmm(logits.softmax(-1), v.transpose(0, 1)).transpose(0, 1)
        for name, out in outputs.items():
            error = (
                (
                    (out[indices].float() - ref).flatten(1).norm(dim=1)
                    / ref.flatten(1).norm(dim=1)
                )
                .max()
                .item()
            )
            errors[name] = max(errors[name], error)
            assert error <= 0.01, (name, b, error)
    return errors


def measure(graph, cache, repetitions):
    events = []
    for _ in range(repetitions):
        cache.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(end) * 1000 for start, end in events)


@torch.inference_mode()
def probe(device, workload, dtype, rounds):
    batch, query, total = workload
    context = total - query
    data = make_inputs(device, batch, query, context, dtype)
    outputs = {
        name: torch.empty_like(data["q"])
        for name in ("baseline", "rocm_autotune", "aiter_unified")
    }
    run = calls(data, query, context, dtype, *outputs.values())
    run["baseline"]()
    with patch.object(
        tuning,
        "get_context_attention_config",
        wraps=tuning.get_context_attention_config,
    ) as lookup:
        run["rocm_autotune"]()
        assert lookup.call_count == (dtype == "bf16")
        if dtype == "bf16":
            assert lookup.call_args.args[5:8] == (batch, query, total)
    config = tuning.get_context_attention_config(
        device, 12, 2, 256, 784, batch, query, total, 0.0625
    )
    if dtype == "bf16":
        assert config is not None
    selected = {}

    def record_2d(*args, **kwargs):
        selected["route"] = "2d_prefill"
        selected["config"] = select_2d(*args, **kwargs)
        return selected["config"]

    def record_3d(*args, **kwargs):
        selected["route"] = "3d_multiquery"
        selected["config"] = select_3d(*args, **kwargs)
        return selected["config"]

    select_2d = aiter_attention.select_2d_config
    select_3d = aiter_attention.select_3d_config
    with (
        patch.object(aiter_attention, "select_2d_config", side_effect=record_2d),
        patch.object(aiter_attention, "select_3d_config", side_effect=record_3d),
    ):
        run["aiter_unified"]()
        assert "route" in selected
    torch.accelerator.synchronize()
    for name in ("rocm_autotune", "aiter_unified"):
        torch.testing.assert_close(
            outputs[name], outputs["baseline"], atol=0.01, rtol=0.01
        )
    pair_l2 = (
        (outputs["rocm_autotune"].float() - outputs["aiter_unified"].float()).norm()
        / outputs["rocm_autotune"].float().norm()
    ).item()
    assert pair_l2 <= 0.01, pair_l2
    errors = reference_errors(data, outputs, batch, query, context)
    graphs = {}
    for name in ("rocm_autotune", "aiter_unified"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run[name]()
        graphs[name] = graph
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            outputs[name], outputs["baseline"], atol=0.01, rtol=0.01
        )
    # Change Q after capture: graph replay must consume fresh input data.
    data["q"].mul_(0.875)
    run["baseline"]()
    for graph in graphs.values():
        graph.replay()
    torch.accelerator.synchronize()
    for name in graphs:
        torch.testing.assert_close(
            outputs[name], outputs["baseline"], atol=0.01, rtol=0.01
        )
    changed_errors = reference_errors(data, outputs, batch, query, context)
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    pilot = {name: measure(graph, cache, 1) for name, graph in graphs.items()}
    warm_repetitions = min(1000, max(1, int(100000 / sum(pilot.values()))))
    for _ in range(warm_repetitions):
        for graph in graphs.values():
            graph.replay()
    torch.accelerator.synchronize()
    repetitions = min(20, max(1, int(5000 / max(pilot.values()))))
    samples = {name: [] for name in graphs}
    for r in range(rounds):
        order = list(graphs) if r % 2 == 0 else list(reversed(graphs))
        for name in order:
            samples[name].append(measure(graphs[name], cache, repetitions))
    times = {name: statistics.median(values) for name, values in samples.items()}
    flops = 4 * batch * 12 * 256 * (query * context + query * (query + 1) / 2)
    return {
        "workload": workload,
        "context": context,
        "dtype": dtype,
        "rocm_route": "autotuned" if dtype == "bf16" else "default_fp8_fallback",
        "rocm_config": config if dtype == "bf16" else tuning._DEFAULT,
        "aiter_route": selected["route"],
        "aiter_config": selected["config"],
        "tps_bf16_patch_active": dtype == "bf16"
        and selected["route"] == "3d_multiquery",
        "us": times,
        "samples_us": samples,
        "repetitions": repetitions,
        "rocm_speedup_vs_aiter": times["aiter_unified"] / times["rocm_autotune"],
        "logical_tflops": {name: flops / us / 1e6 for name, us in times.items()},
        "pair_relative_l2": pair_l2,
        "fp32_sampled_row_l2": errors,
        "changed_q_fp32_sampled_row_l2": changed_errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-shape", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    assert envs.VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE
    assert str(aiter_attention.__file__).startswith("/app/tps-rdna4-qwen38-tp2/aiter/")
    if not args.manifest.exists():
        shapes = set()
        sources = []
        for path in sorted(
            (Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention").glob("*.json")
        ):
            data = json.loads(path.read_text())
            identity = data["identity"]
            if tuple(identity[k] for k in ("heads", "kv_heads", "dim", "page")) != (
                12,
                2,
                256,
                784,
            ):
                continue
            shapes.update(tuple(r["workload"]) for r in data["records"])
            sources.append(
                {
                    "path": str(path),
                    "sha256": fingerprint(path),
                    "records": len(data["records"]),
                }
            )
        save(args.manifest, {"workloads": sorted(shapes), "source_caches": sources})
    manifest = json.loads(args.manifest.read_text())
    workloads = [tuple(w) for w in manifest["workloads"]]
    if args.only_shape:
        selected = {tuple(map(int, s.split(","))) for s in args.only_shape}
        workloads = [w for w in workloads if (w[0], w[1], w[2] - w[1]) in selected]
        assert len(workloads) == len(selected)
    # Retune historical source fingerprints before using automatic dispatch.
    # Bound offline warmup to the exact saved-shape union, including old large Q.
    with patch.object(
        tuning, "_workloads", side_effect=lambda *a, **k: iter(workloads)
    ):
        tuning.warmup_context_attention(
            device,
            torch.bfloat16,
            12,
            2,
            256,
            784,
            0.0625,
            65536,
            524288,
            32,
            memory_budget_bytes=8 * 1024**3,
        )
    tuning._tune_workload = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("Runtime retuning")
    )
    identity = tuning._identity(device, 12, 2, 256, 784, 0.0625)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = (
        Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention" / f"{digest}.json"
    )
    cache_snapshot = (fingerprint(cache_path), cache_path.stat().st_mtime_ns)
    result = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(device),
            "arch": torch.cuda.get_device_properties(device).gcnArchName,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": tuning.triton.__version__,
            "heads": 12,
            "kv_heads": 2,
            "dim": 256,
            "page": 784,
            "manifest": str(args.manifest),
            "manifest_sha256": fingerprint(args.manifest),
            "rounds": args.rounds,
            "aiter_source": str(aiter_attention.__file__),
            "aiter_source_sha256": fingerprint(aiter_attention.__file__),
            "rocm_source_sha256": fingerprint(tuning.__file__),
            "rocm_kernel_sha256": fingerprint(
                Path(tuning.__file__).with_name("prefix_prefill.py")
            ),
            "aiter_kernel_sha256": fingerprint(
                Path(aiter_attention.__file__).parents[1]
                / "_triton_kernels/attention/unified_attention.py"
            ),
            "tuning_cache": str(cache_path),
            "tuning_cache_sha256": cache_snapshot[0],
            "harness_sha256": fingerprint(__file__),
            "timing": "cold-L2 CUDA graph GPU events; alternating order",
        },
        "rows": [],
        "complete": False,
    }
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text())
        assert saved["metadata"] == result["metadata"]
        result["rows"] = saved["rows"]
    completed = {(tuple(r["workload"]), r["dtype"]) for r in result["rows"]}
    start = time.monotonic()
    for workload in workloads:
        for dtype in ("bf16", "fp8"):
            if (workload, dtype) in completed:
                continue
            row = probe(device, workload, dtype, args.rounds)
            result["rows"].append(row)
            save(args.output, result)
            print("RESULT " + json.dumps(row), flush=True)
            gc.collect()
            torch.accelerator.empty_cache()
    result["complete"] = True
    assert cache_snapshot == (fingerprint(cache_path), cache_path.stat().st_mtime_ns)
    result["inference_cache_unchanged"] = True
    result["elapsed_s"] = time.monotonic() - start
    save(args.output, result)
    print(
        f"COMPLETE rows={len(result['rows'])} seconds={result['elapsed_s']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
