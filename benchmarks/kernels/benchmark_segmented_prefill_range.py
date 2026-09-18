# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare segmented prefill with AITER over the established RDNA4 range."""

import argparse
import gc
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import torch
from benchmark_segmented_prefill import make_call, make_inputs
from rdna4_prefill_prototype.eviction import ReadEviction

from vllm.v1.attention.ops import segmented_prefill

QUERY_LENGTHS = (2, 8, 32, 128)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
SEQUENCE_LENGTHS = (128, 512, 2048, 8192, 32768, 131072, 262144)
FAMILIES = (("tp1", 24, 4), ("tp2", 12, 2), ("tp4", 6, 1))


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_name(case: dict) -> str:
    query = case["queries"][0]
    batch = len(case["queries"])
    sequence = query + case["contexts"][0]
    dtype = "fp8" if case["fp8"] else "bf16"
    family = next(
        name
        for name, heads, kv_heads in FAMILIES
        if (heads, kv_heads) == (case["heads"], case["kv_heads"])
    )
    return f"{family}-q{query}-b{batch}-s{sequence}-{dtype}"


def _cases() -> list[dict]:
    result = []
    for (_family, heads, kv_heads), query, batch, sequence, fp8 in itertools.product(
        FAMILIES, QUERY_LENGTHS, BATCH_SIZES, SEQUENCE_LENGTHS, (False, True)
    ):
        if query > sequence:
            continue
        result.append(
            {
                "queries": [query] * batch,
                "contexts": [sequence - query] * batch,
                "fp8": fp8,
                "page": 1568 if fp8 else 784,
                "heads": heads,
                "kv_heads": kv_heads,
            }
        )
    return result


def _estimated_bytes(case: dict) -> int:
    batch = len(case["queries"])
    query = case["queries"][0]
    sequence = query + case["contexts"][0]
    page = case["page"]
    heads, kv_heads, dim = case["heads"], case["kv_heads"], 256
    cache_elements = batch * math.ceil(sequence / page) * page * kv_heads * dim
    cache_bytes = 2 * cache_elements * (1 if case["fp8"] else 2)
    dense_bytes = batch * query * (heads + 2 * kv_heads) * dim * 2
    config = segmented_prefill.select_segmented_unified_config(
        batch, query, sequence, heads, kv_heads, dim, case["fp8"]
    )
    workspace_rows = config["splits"] * batch * query * heads
    workspace_bytes = workspace_rows * (dim + 1) * 4
    # Includes eviction buffers, graph pools, AITER partials, and allocator headroom.
    return cache_bytes + dense_bytes + workspace_bytes + 1024**3


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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return (
        (left.float() - right.float())
        .norm(dim=-1)
        .div(right.float().norm(dim=-1).clamp_min(1e-8))
        .max()
        .item()
    )


@torch.inference_mode()
def _probe(
    case: dict, samples: int, rounds: int, segmented_config: dict | None
) -> dict:
    data = make_inputs(**case, legacy_layout=False)
    calls = {}
    outputs = {}
    selected = {}
    graphs = {}
    for backend in ("segmented", "aiter"):
        if backend == "segmented":
            config = segmented_config or {"auto_unified": True}
        else:
            config = None
        calls[backend], outputs[backend], selected[backend] = make_call(
            data, backend, config
        )
        calls[backend]()
    torch.cuda.synchronize()
    error = _relative_l2(outputs["segmented"], outputs["aiter"])
    if not torch.isfinite(outputs["segmented"]).all() or error >= 0.01:
        raise AssertionError(f"Triton/AITER max row L2={error}")
    for backend in ("segmented", "aiter"):
        graphs[backend] = _capture(calls[backend])
    read = ReadEviction()
    write = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    times = {backend: {} for backend in graphs}
    for regime, eviction in (("read", read), ("write", write), ("reuse", None)):
        for round_index in range(rounds):
            order = (
                ("segmented", "aiter")
                if round_index % 2 == 0
                else ("aiter", "segmented")
            )
            for backend in order:
                value = _measure(graphs[backend], eviction, samples)
                times[backend].setdefault(regime, []).append(value)
    medians = {
        backend: {
            regime: statistics.median(rounds)
            for regime, rounds in backend_times.items()
        }
        for backend, backend_times in times.items()
    }
    return {
        "case": case,
        "name": _case_name(case),
        "estimated_bytes": _estimated_bytes(case),
        "selected": selected,
        "max_row_l2": error,
        "times_us": medians,
        "samples_us": times,
        "speedup_vs_aiter": {
            regime: medians["aiter"][regime] / medians["segmented"][regime]
            for regime in ("read", "write", "reuse")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--segmented-config", type=json.loads)
    parser.add_argument("--memory-budget-gib", type=int, default=8)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        raise ValueError("Samples and rounds must be positive")

    candidates = _cases()
    if args.case:
        requested = set(args.case)
        candidates = [case for case in candidates if _case_name(case) in requested]
        if len(candidates) != len(requested):
            raise ValueError("Unknown or duplicate case name")
    limit = args.memory_budget_gib * 1024**3
    planned = [case for case in candidates if _estimated_bytes(case) <= limit]
    skipped = [case for case in candidates if _estimated_bytes(case) > limit]
    root = Path(__file__).resolve().parents[2]
    result = {
        "metadata": {
            "samples_per_round": args.samples,
            "rounds_per_regime": args.rounds,
            "timing": "alternating CUDA-graph GPU events",
            "memory_budget_gib": args.memory_budget_gib,
            "query_lengths": list(QUERY_LENGTHS),
            "batch_sizes": list(BATCH_SIZES),
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "families": [list(family) for family in FAMILIES],
            "segmented_config": args.segmented_config,
            "kernel_sha256": _sha256(
                root / "vllm/v1/attention/ops/segmented_prefill.py"
            ),
            "harness_sha256": _sha256(Path(__file__)),
            "fixture_sha256": _sha256(
                Path(__file__).with_name("benchmark_segmented_prefill.py")
            ),
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
            row = _probe(case, args.samples, args.rounds, args.segmented_config)
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
