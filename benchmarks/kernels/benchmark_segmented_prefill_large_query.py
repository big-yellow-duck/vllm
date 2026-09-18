# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sweep segmented prefill against AITER over the large-query gate domain."""

import argparse
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import benchmark_segmented_prefill as benchmark
import torch
from benchmark_segmented_prefill import make_call, make_inputs
from rdna4_prefill_prototype.eviction import ReadEviction

from vllm.v1.attention.ops import segmented_prefill

EXPERIMENTAL_QUERY_LIMIT = 8192
DTYPES = (False, True)
QWEN38_TP = (("tp1", 24, 4), ("tp2", 12, 2), ("tp4", 6, 1))
BASE_QUERIES = (256, 512, 1024, 2048, 4096, 8192)
LONG_SEQUENCES = (8192, 32768, 131072, 262144)
BOUNDARY_QUERIES = (
    129,
    192,
    255,
    257,
    384,
    511,
    513,
    768,
    1023,
    1025,
    1536,
    2047,
    2049,
    3072,
    4095,
    4097,
    6144,
    8191,
)
GQA_QUERIES = (128, 129, 256, 512, 1024, 2048, 4096)
GQA_LONG_QUERIES = (256, 1024, 4096)
GQA_RATIOS = tuple(range(1, 17))
GQA_DIMS = (128, 256)
GQA_KV_HEAD_SCALES = (1, 4)
MODEL_FAMILIES = {
    "qwen35_122b": (256, ((32, 2), (16, 1), (8, 1))),
    "qwen35_35b": (256, ((16, 2), (8, 1), (4, 1))),
    "glm45_air": (128, ((96, 8), (48, 4), (24, 2))),
    "llama31_70b": (128, ((64, 8), (32, 4), (16, 2))),
    "mistral_small31": (128, ((32, 8), (16, 4), (8, 2))),
    "gemma3_27b": (128, ((32, 16), (16, 8), (8, 4))),
    "llama4_scout": (128, ((40, 8), (20, 4), (10, 2))),
}


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(
    suite: str,
    family: str,
    queries: list[int],
    contexts: list[int],
    heads: int,
    kv_heads: int,
    dim: int,
    fp8: bool,
    variant: str,
) -> dict:
    return {
        "name": (
            f"{suite}-{family}-{variant}-d{dim}-h{heads}-{kv_heads}-"
            f"{'fp8' if fp8 else 'bf16'}"
        ),
        "suite": suite,
        "family": family,
        "variant": variant,
        "inputs": {
            "queries": queries,
            "contexts": contexts,
            "fp8": fp8,
            "page": 1568 if fp8 else 784,
            "heads": heads,
            "kv_heads": kv_heads,
            "dim": dim,
        },
    }


def _dense_cases() -> list[dict]:
    cases = []
    for family, heads, kv_heads in QWEN38_TP:
        for query in BASE_QUERIES:
            for multiple in (1, 2, 4):
                for fp8 in DTYPES:
                    cases.append(
                        _case(
                            "dense",
                            family,
                            [query],
                            [query * (multiple - 1)],
                            heads,
                            kv_heads,
                            256,
                            fp8,
                            f"q{query}-s{query * multiple}",
                        )
                    )
    return cases


def _batch_cases() -> list[dict]:
    patterns = (
        ("b2-q4096-dense", [4096, 4096], [0, 0]),
        ("b2-q4096-prefix", [4096, 4096], [12288, 12288]),
        ("b4-q2048-dense", [2048] * 4, [0] * 4),
        ("b4-q2048-prefix", [2048] * 4, [6144] * 4),
        ("b8-q1024-dense", [1024] * 8, [0] * 8),
        ("b8-q1024-prefix", [1024] * 8, [7168] * 8),
        ("ragged-dense", [4096, 2048, 1024, 512], [0] * 4),
        (
            "ragged-mixed-prefix",
            [4096, 2048, 1024, 512],
            [4096, 6144, 31744, 130560],
        ),
    )
    cases = []
    for family, heads, kv_heads in QWEN38_TP:
        for variant, queries, contexts in patterns:
            for fp8 in DTYPES:
                cases.append(
                    _case(
                        "batch",
                        family,
                        queries,
                        contexts,
                        heads,
                        kv_heads,
                        256,
                        fp8,
                        variant,
                    )
                )
    return cases


def _model_cases() -> list[dict]:
    cases = []
    for family, (dim, tp_shapes) in MODEL_FAMILIES.items():
        for tp_index, (heads, kv_heads) in enumerate(tp_shapes, 1):
            tp = 1 << (tp_index - 1)
            for query in (256, 2048, 8192):
                for sequence in (query, 131072):
                    for fp8 in DTYPES:
                        cases.append(
                            _case(
                                "models",
                                family,
                                [query],
                                [sequence - query],
                                heads,
                                kv_heads,
                                dim,
                                fp8,
                                f"tp{tp}-q{query}-s{sequence}",
                            )
                        )
    return cases


def _boundary_cases() -> list[dict]:
    cases = []
    for family, heads, kv_heads in QWEN38_TP:
        for query in BOUNDARY_QUERIES:
            for fp8 in DTYPES:
                cases.append(
                    _case(
                        "boundaries",
                        family,
                        [query],
                        [0],
                        heads,
                        kv_heads,
                        256,
                        fp8,
                        f"q{query}-s{query}",
                    )
                )
    return cases


def _regression_cases() -> list[dict]:
    cases = []
    for family, heads, kv_heads in QWEN38_TP:
        for query in BASE_QUERIES:
            for sequence in LONG_SEQUENCES:
                if query > sequence:
                    continue
                for fp8 in DTYPES:
                    cases.append(
                        _case(
                            "regression",
                            family,
                            [query],
                            [sequence - query],
                            heads,
                            kv_heads,
                            256,
                            fp8,
                            f"q{query}-s{sequence}",
                        )
                    )
    return cases


def _gqa_dense_cases() -> list[dict]:
    cases = []
    for dim in GQA_DIMS:
        for gqa in GQA_RATIOS:
            for kv_heads in GQA_KV_HEAD_SCALES:
                heads = gqa * kv_heads
                for query in GQA_QUERIES:
                    for fp8 in DTYPES:
                        cases.append(
                            _case(
                                "gqa_dense",
                                f"hk{kv_heads}-gqa{gqa}",
                                [query],
                                [0],
                                heads,
                                kv_heads,
                                dim,
                                fp8,
                                f"q{query}-s{query}",
                            )
                        )
    return cases


def _gqa_long_cases() -> list[dict]:
    cases = []
    for dim in GQA_DIMS:
        for gqa in GQA_RATIOS:
            for query in GQA_LONG_QUERIES:
                for fp8 in DTYPES:
                    cases.append(
                        _case(
                            "gqa_long",
                            f"hk1-gqa{gqa}",
                            [query],
                            [131072 - query],
                            gqa,
                            1,
                            dim,
                            fp8,
                            f"q{query}-s131072",
                        )
                    )
    return cases


def _gqa_batch_cases() -> list[dict]:
    patterns = (
        ("b4-q1024-dense", [1024] * 4, [0] * 4),
        ("b4-q1024-prefix", [1024] * 4, [7168] * 4),
        ("ragged-dense", [2048, 1024, 512, 256], [0] * 4),
    )
    cases = []
    for dim in GQA_DIMS:
        for gqa in GQA_RATIOS:
            for variant, queries, contexts in patterns:
                for fp8 in DTYPES:
                    cases.append(
                        _case(
                            "gqa_batch",
                            f"hk1-gqa{gqa}",
                            queries,
                            contexts,
                            gqa,
                            1,
                            dim,
                            fp8,
                            variant,
                        )
                    )
    return cases


SUITES = {
    "dense": _dense_cases,
    "batch": _batch_cases,
    "models": _model_cases,
    "boundaries": _boundary_cases,
    "regression": _regression_cases,
    "gqa_dense": _gqa_dense_cases,
    "gqa_long": _gqa_long_cases,
    "gqa_batch": _gqa_batch_cases,
}


def _capture(run):
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    return graph


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


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return (
        (left.float() - right.float())
        .norm(dim=-1)
        .div(right.float().norm(dim=-1).clamp_min(1e-8))
        .max()
        .item()
    )


@torch.inference_mode()
def _probe(case: dict, samples: int, rounds: int, config_overrides: dict) -> dict:
    data = make_inputs(**case["inputs"], legacy_layout=False)
    inputs = case["inputs"]
    config = dict(
        segmented_prefill.select_segmented_config(
            len(inputs["queries"]),
            max(inputs["queries"]),
            max(q + c for q, c in zip(inputs["queries"], inputs["contexts"])),
            inputs["heads"],
            inputs["kv_heads"],
            inputs["dim"],
            inputs["fp8"],
        )
    )
    config.update(config_overrides)
    segmented_run, segmented_out, segmented_cfg = make_call(data, "segmented", config)
    aiter_run, aiter_out, aiter_cfg = make_call(data, "aiter")
    segmented_run()
    aiter_run()
    torch.cuda.synchronize()
    error = _relative_l2(segmented_out, aiter_out)
    if not segmented_out.isfinite().all() or error >= 0.01:
        raise AssertionError(f"Triton/AITER max row L2={error}")
    graphs = {
        "segmented": _capture(segmented_run),
        "aiter": _capture(aiter_run),
    }
    read = ReadEviction()
    write = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    times = {backend: {} for backend in graphs}
    for regime, eviction in (("read", read), ("write", write), ("reuse", None)):
        for round_index in range(rounds):
            order = tuple(graphs) if round_index % 2 == 0 else tuple(reversed(graphs))
            for backend in order:
                value = _measure(graphs[backend], eviction, samples)
                times[backend].setdefault(regime, []).append(value)
    medians = {
        backend: {
            regime: statistics.median(values)
            for regime, values in backend_times.items()
        }
        for backend, backend_times in times.items()
    }
    return {
        **case,
        "selected": {"segmented": segmented_cfg, "aiter": aiter_cfg},
        "max_row_l2": error,
        "times_us": medians,
        "samples_us": times,
        "speedup_vs_aiter": {
            regime: medians["aiter"][regime] / medians["segmented"][regime]
            for regime in ("read", "write", "reuse")
        },
    }


def _estimated_bytes(case: dict) -> int:
    inputs = case["inputs"]
    batch = len(inputs["queries"])
    sequence = max(q + c for q, c in zip(inputs["queries"], inputs["contexts"]))
    blocks = math.ceil(sequence / inputs["page"])
    cache = (
        2
        * batch
        * blocks
        * inputs["page"]
        * inputs["kv_heads"]
        * inputs["dim"]
        * (1 if inputs["fp8"] else 2)
    )
    dense = (
        sum(inputs["queries"])
        * (inputs["heads"] + 2 * inputs["kv_heads"])
        * inputs["dim"]
        * 2
    )
    return cache + dense + 2 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--memory-budget-gib", type=int, default=28)
    parser.add_argument("--segmented-config", type=json.loads, default={})
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        raise ValueError("Samples and rounds must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard selection")

    segmented_prefill.MAX_QUERY_LEN = EXPERIMENTAL_QUERY_LIMIT
    benchmark.MAX_QUERY_LEN = EXPERIMENTAL_QUERY_LIMIT
    all_cases = SUITES[args.suite]()
    if args.case:
        requested = set(args.case)
        all_cases = [case for case in all_cases if case["name"] in requested]
        if len(all_cases) != len(requested):
            raise ValueError("Unknown or duplicate case name")
    cases = [
        case
        for index, case in enumerate(all_cases)
        if index % args.num_shards == args.shard_index
    ]
    limit = args.memory_budget_gib * 1024**3
    selected_cases = [case for case in cases if _estimated_bytes(case) <= limit]
    skipped = [case for case in cases if _estimated_bytes(case) > limit]
    root = Path(__file__).resolve().parents[2]
    result = {
        "metadata": {
            "suite": args.suite,
            "samples_per_round": args.samples,
            "rounds_per_regime": args.rounds,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "experimental_query_limit": EXPERIMENTAL_QUERY_LIMIT,
            "memory_budget_gib": args.memory_budget_gib,
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
            "all_cases": len(all_cases),
            "shard_cases": len(cases),
            "selected_cases": len(selected_cases),
            "skipped": [case["name"] for case in skipped],
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
    done = {row["name"] for row in result["rows"] + result["failures"]}
    for case in selected_cases:
        if case["name"] in done:
            continue
        try:
            row = _probe(case, args.samples, args.rounds, args.segmented_config)
        except Exception as error:
            row = {**case, "error": repr(error)}
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
