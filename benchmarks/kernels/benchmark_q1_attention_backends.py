# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the three ROCm attention backends over the SplitKV Q=1 range."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
from benchmark_segmented_prefill import make_call, make_inputs, reference
from rdna4_prefill_prototype.eviction import ReadEviction

from vllm import envs
from vllm.platforms import current_platform
from vllm.v1.attention.ops import segmented_prefill
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _get_num_splits,
    _paged_attention_2d_splitkv_decode,
)
from vllm.v1.attention.ops.rdna4_splitkv import (
    get_rdna4_flydsl_splitkv_config,
)

# Representative rank-local head topologies spanning the production D/GQA gate.
TOPOLOGIES = (
    ("d128-g1", 8, 8, 128),
    ("d128-g2", 8, 4, 128),
    ("d128-g4", 16, 4, 128),
    ("d128-g8", 32, 4, 128),
    ("d128-g16", 16, 1, 128),
    ("d256-g1", 4, 4, 256),
    ("d256-g2", 8, 4, 256),
    ("d256-g4", 8, 2, 256),
    ("d256-g8", 16, 2, 256),
    ("d256-g16", 16, 1, 256),
)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
SEQUENCE_LENGTHS = (128, 512, 2048, 8192, 32768, 131072, 262144)
REGIMES = ("read", "write", "reuse")
BACKENDS = ("rocm_attn", "rocm_segmented_attn", "rocm_aiter_unified_attn")


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_name(case: dict) -> str:
    dtype = "fp8" if case["fp8"] else "bf16"
    return f"{case['topology']}-q1-b{case['batch']}-s{case['sequence']}-{dtype}"


def _cases(num_sms: int) -> list[dict]:
    cases = []
    for topology, batch, sequence, fp8 in itertools.product(
        TOPOLOGIES, BATCH_SIZES, SEQUENCE_LENGTHS, (False, True)
    ):
        topology_name, heads, kv_heads, dim = topology
        page = 1568 if fp8 else 784
        splits = _get_num_splits(
            batch,
            kv_heads,
            dim,
            page,
            sequence,
            num_sms=num_sms,
            allow_short_context=fp8,
        )
        if splits == 1:
            continue
        cases.append(
            {
                "topology": topology_name,
                "heads": heads,
                "kv_heads": kv_heads,
                "dim": dim,
                "gqa": heads // kv_heads,
                "batch": batch,
                "sequence": sequence,
                "fp8": fp8,
                "page": page,
                "splits": splits,
            }
        )
    return cases


def _estimated_bytes(case: dict) -> int:
    blocks = case["batch"] * math.ceil(case["sequence"] / case["page"])
    element_size = 1 if case["fp8"] else 2
    one_cache = blocks * case["page"] * case["kv_heads"] * case["dim"] * element_size
    # make_inputs first materializes K/V in both layouts, then each layout is
    # copied into a padded backing allocation.  The caching allocator can keep
    # all four superseded tensors resident while the four padded tensors are
    # live, so size for the observed eight-cache peak rather than steady state.
    cache_bytes = 8 * one_cache
    legacy_scratch = (
        case["batch"] * case["heads"] * case["splits"] * (case["dim"] + 1) * 4
    )
    config = segmented_prefill.select_segmented_config(
        case["batch"],
        1,
        case["sequence"],
        case["heads"],
        case["kv_heads"],
        case["dim"],
        case["fp8"],
    )
    shapes = segmented_prefill.segmented_workspace_shapes(
        case["batch"],
        segmented_prefill.segmented_query_capacity(1),
        case["heads"],
        case["kv_heads"],
        case["dim"],
        config["splits"],
    )
    segmented_scratch = (
        0 if shapes is None else sum(math.prod(shape) * 4 for shape in shapes)
    )
    # Eviction buffers, graph pools, AITER partials, compiler, and allocator margin.
    return cache_bytes + legacy_scratch + segmented_scratch + 2 * 1024**3


def _measure(graph, eviction, samples: int) -> float:
    pairs = []
    for _ in range(samples):
        if eviction is not None:
            eviction.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        pairs.append((start, end))
    pairs[-1][1].synchronize()
    return statistics.median(start.elapsed_time(end) * 1000 for start, end in pairs)


def _capture(run):
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _max_row_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float())
        .norm(dim=-1)
        .div(expected.float().norm(dim=-1).clamp_min(1e-8))
        .max()
        .item()
    )


def _logical_rates(case: dict, times: dict) -> dict:
    element_size = 1 if case["fp8"] else 2
    logical_bytes = (
        2
        * case["batch"]
        * case["sequence"]
        * case["kv_heads"]
        * case["dim"]
        * element_size
    )
    logical_flops = 4 * case["batch"] * case["heads"] * case["dim"] * case["sequence"]
    return {
        backend: {
            regime: {
                "logical_gbps": logical_bytes / value / 1000,
                "logical_tflops": logical_flops / value / 1e6,
            }
            for regime, value in backend_times.items()
        }
        for backend, backend_times in times.items()
    }


@torch.inference_mode()
def _probe(case: dict, samples: int, rounds: int) -> dict:
    batch = case["batch"]
    data = make_inputs(
        queries=[1] * batch,
        contexts=[case["sequence"] - 1] * batch,
        heads=case["heads"],
        kv_heads=case["kv_heads"],
        dim=case["dim"],
        page=case["page"],
        fp8=case["fp8"],
        legacy_layout=True,
    )
    expected = reference(data)
    scale = case["dim"] ** -0.5
    outputs = {}
    calls = {}
    selected = {}

    segmented_call, segmented_output, segmented_selected = make_call(data, "segmented")
    calls["rocm_segmented_attn"] = segmented_call
    outputs["rocm_segmented_attn"] = segmented_output
    selected["rocm_segmented_attn"] = segmented_selected

    aiter_call, aiter_output, aiter_selected = make_call(data, "aiter")
    calls["rocm_aiter_unified_attn"] = aiter_call
    outputs["rocm_aiter_unified_attn"] = aiter_output
    selected["rocm_aiter_unified_attn"] = aiter_selected

    rocm_output = torch.empty_like(data["q"])
    mid_out = torch.empty(
        batch,
        case["heads"],
        case["splits"],
        case["dim"],
        device=data["q"].device,
        dtype=torch.float32,
    )
    mid_lse = torch.empty(
        batch,
        case["heads"],
        case["splits"],
        device=data["q"].device,
        dtype=torch.float32,
    )
    flydsl_config = get_rdna4_flydsl_splitkv_config(
        query=data["q"],
        key_cache=data["kc"],
        value_cache=data["vc"],
        output=rocm_output,
        block_tables=data["table"],
        seq_lens=data["lens"],
        query_start_loc=data["starts"],
        k_scale=data["ks"],
        v_scale=data["vs"],
        scale=scale,
        actual_max_splits=case["splits"],
        max_seq_len=case["sequence"],
        filter_by_query_len=True,
    )

    def rocm_call():
        _paged_attention_2d_splitkv_decode(
            query=data["q"],
            key_cache=data["kc"],
            value_cache=data["vc"],
            block_tables=data["table"],
            seq_lens=data["lens"],
            scale=scale,
            k_scale=data["ks"],
            v_scale=data["vs"],
            output=rocm_output,
            actual_max_splits=case["splits"],
            max_seq_len=case["sequence"],
            mid_out=mid_out,
            mid_lse=mid_lse,
            query_start_loc=data["starts"],
            filter_by_query_len=True,
        )

    calls["rocm_attn"] = rocm_call
    outputs["rocm_attn"] = rocm_output
    selected["rocm_attn"] = {
        "kind": "flydsl" if flydsl_config is not None else "triton_splitkv",
        "route": None if flydsl_config is None else flydsl_config.route.value,
        "splits": case["splits"],
    }

    errors = {}
    for backend in BACKENDS:
        calls[backend]()
        torch.cuda.synchronize()
        error = _max_row_l2(outputs[backend], expected)
        if not torch.isfinite(outputs[backend]).all() or error >= 0.01:
            raise AssertionError(f"{backend} max row L2={error}")
        errors[backend] = error

    pair_errors = {
        "segmented_vs_rocm": _max_row_l2(
            outputs["rocm_segmented_attn"], outputs["rocm_attn"]
        ),
        "segmented_vs_aiter": _max_row_l2(
            outputs["rocm_segmented_attn"], outputs["rocm_aiter_unified_attn"]
        ),
        "rocm_vs_aiter": _max_row_l2(
            outputs["rocm_attn"], outputs["rocm_aiter_unified_attn"]
        ),
    }
    if max(pair_errors.values()) >= 0.01:
        raise AssertionError(f"pairwise max row L2={pair_errors}")

    graphs = {backend: _capture(calls[backend]) for backend in BACKENDS}
    read = ReadEviction()
    write = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    samples_by_backend = {
        backend: {regime: [] for regime in REGIMES} for backend in BACKENDS
    }
    for regime, eviction in (("read", read), ("write", write), ("reuse", None)):
        for round_index in range(rounds):
            order = BACKENDS if round_index % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                samples_by_backend[backend][regime].append(
                    _measure(graphs[backend], eviction, samples)
                )
    times = {
        backend: {
            regime: statistics.median(values)
            for regime, values in backend_samples.items()
        }
        for backend, backend_samples in samples_by_backend.items()
    }
    speedups = {
        regime: {
            "segmented_vs_rocm": times["rocm_attn"][regime]
            / times["rocm_segmented_attn"][regime],
            "segmented_vs_aiter": times["rocm_aiter_unified_attn"][regime]
            / times["rocm_segmented_attn"][regime],
            "rocm_vs_aiter": times["rocm_aiter_unified_attn"][regime]
            / times["rocm_attn"][regime],
        }
        for regime in REGIMES
    }
    winners = {
        regime: min(BACKENDS, key=lambda backend: times[backend][regime])
        for regime in REGIMES
    }
    return {
        "name": _case_name(case),
        "case": case,
        "estimated_bytes": _estimated_bytes(case),
        "selected": selected,
        "max_row_l2_vs_reference": errors,
        "pairwise_max_row_l2": pair_errors,
        "times_us": times,
        "samples_us": samples_by_backend,
        "speedups": speedups,
        "winners": winners,
        "logical_rates": _logical_rates(case, times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--memory-budget-gib", type=int, default=24)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        raise ValueError("Samples and rounds must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if not current_platform.is_rocm():
        raise RuntimeError("This benchmark requires ROCm")

    envs.VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL = True
    properties = torch.cuda.get_device_properties(0)
    candidates = _cases(properties.multi_processor_count)
    if args.case:
        requested = set(args.case)
        candidates = [case for case in candidates if _case_name(case) in requested]
        if len(candidates) != len(requested):
            raise ValueError("Unknown or duplicate case name")
    candidates = [
        case
        for index, case in enumerate(candidates)
        if index % args.num_shards == args.shard_index
    ]
    limit = args.memory_budget_gib * 1024**3
    planned = [case for case in candidates if _estimated_bytes(case) <= limit]
    skipped = [case for case in candidates if _estimated_bytes(case) > limit]
    root = Path(__file__).resolve().parents[2]
    result = {
        "metadata": {
            "gpu": properties.name,
            "gcn_arch": properties.gcnArchName,
            "num_sms": properties.multi_processor_count,
            "samples_per_round": args.samples,
            "rounds_per_regime": args.rounds,
            "timing": "alternating CUDA-graph GPU events",
            "regimes": list(REGIMES),
            "backends": list(BACKENDS),
            "memory_budget_gib": args.memory_budget_gib,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "query_length": 1,
            "batch_sizes": list(BATCH_SIZES),
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "topologies": [list(topology) for topology in TOPOLOGIES],
            "page_sizes": {"bf16": 784, "fp8": 1568},
            "candidate_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=root, text=True
            ).strip(),
            "segmented_kernel_sha256": _sha256(
                root / "vllm/v1/attention/ops/segmented_prefill.py"
            ),
            "splitkv_kernel_sha256": _sha256(
                root / "vllm/v1/attention/ops/chunked_prefill_paged_decode.py"
            ),
            "flydsl_router_sha256": _sha256(
                root / "vllm/v1/attention/ops/flydsl_kernels/rdna4_splitkv.py"
            ),
            "harness_sha256": _sha256(Path(__file__)),
        },
        "plan": {
            "candidates": len(candidates),
            "planned": len(planned),
            "skipped": [
                {"name": _case_name(case), "estimated_bytes": _estimated_bytes(case)}
                for case in skipped
            ],
        },
        "rows": [],
        "failures": [],
        "complete": False,
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous["metadata"] != result["metadata"]
            or previous["plan"] != result["plan"]
        ):
            raise ValueError("Existing output does not match this benchmark plan")
        result["rows"] = previous["rows"]
        result["failures"] = previous["failures"]
    _save(args.output, result)
    if args.plan_only:
        print(json.dumps(result["plan"], indent=2))
        return

    done = {row["name"] for row in result["rows"] + result["failures"]}
    for case in planned:
        name = _case_name(case)
        if name in done:
            continue
        try:
            row = _probe(case, args.samples, args.rounds)
        except Exception as error:
            row = {"name": name, "case": case, "error": repr(error)}
            result["failures"].append(row)
            print("FAILED " + json.dumps(row), flush=True)
        else:
            result["rows"].append(row)
            print("RESULT " + json.dumps(row), flush=True)
        _save(args.output, result)
        gc.collect()
        torch.accelerator.empty_cache()
    result["complete"] = True
    _save(args.output, result)
    print(
        f"COMPLETE rows={len(result['rows'])} failures={len(result['failures'])} "
        f"skipped={len(skipped)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
