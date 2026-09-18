# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Probe larger chunked-prefill queries across the long-prompt TTFT region."""

import argparse
import gc
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import benchmark_segmented_prefill as benchmark
import torch
from benchmark_segmented_prefill import make_call, make_inputs
from rdna4_prefill_prototype.eviction import ReadEviction

from vllm.v1.attention.ops import segmented_prefill
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

QUERY_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SEQUENCE_LENGTHS = (8192, 32768, 131072, 262144)
FAMILIES = (("tp1", 24, 4), ("tp2", 12, 2), ("tp4", 6, 1))
EXPERIMENTAL_QUERY_LIMIT = max(QUERY_LENGTHS)


def _save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_name(case: dict) -> str:
    dtype = "fp8" if case["fp8"] else "bf16"
    family = next(
        name
        for name, heads, kv_heads in FAMILIES
        if (heads, kv_heads) == (case["heads"], case["kv_heads"])
    )
    return f"{family}-q{case['queries'][0]}-s{case['sequence_length']}-{dtype}"


def _cases() -> list[dict]:
    cases = []
    for (_family, heads, kv_heads), query, sequence, fp8 in itertools.product(
        FAMILIES, QUERY_LENGTHS, SEQUENCE_LENGTHS, (False, True)
    ):
        if query > sequence:
            continue
        cases.append(
            {
                "queries": [query],
                "contexts": [sequence - query],
                "sequence_length": sequence,
                "fp8": fp8,
                "page": 1568 if fp8 else 784,
                "heads": heads,
                "kv_heads": kv_heads,
            }
        )
    return cases


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
def _probe(case: dict, samples: int, rounds: int) -> dict:
    query = case["queries"][0]
    sequence = case["sequence_length"]
    inputs = {key: value for key, value in case.items() if key != "sequence_length"}
    data = make_inputs(**inputs, legacy_layout=False)

    calls = {}
    outputs = {}
    selected = {}
    calls["segmented"], outputs["segmented"], selected["segmented"] = make_call(
        data, "segmented", {"auto_unified": True}
    )
    calls["aiter"], outputs["aiter"], selected["aiter"] = make_call(data, "aiter")
    outputs["context"] = torch.empty_like(data["q"])

    def context_call():
        context_attention_fwd(
            q=data["q"],
            k=data["k"],
            v=data["v"],
            o=outputs["context"],
            kv_cache_dtype="fp8" if case["fp8"] else "auto",
            k_cache=data["kn"],
            v_cache=data["vn"],
            b_loc=data["table"],
            b_start_loc=data["starts"],
            b_seq_len=data["lens"],
            max_seq_len=sequence,
            max_input_len=query,
            k_scale=data["ks"],
            v_scale=data["vs"],
            sm_scale=256**-0.5,
            skip_decode=False,
        )

    calls["context"] = context_call
    selected["context"] = {"kind": "context_attention_fwd_2d"}
    for run in calls.values():
        run()
    torch.cuda.synchronize()
    errors = {
        name: _relative_l2(output, outputs["aiter"])
        for name, output in outputs.items()
        if name != "aiter"
    }
    if any(error >= 0.01 for error in errors.values()) or any(
        not output.isfinite().all() for output in outputs.values()
    ):
        raise AssertionError(f"max row relative L2 versus AITER: {errors}")

    graphs = {name: _capture(run) for name, run in calls.items()}
    read = ReadEviction()
    write = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    timings = {name: {} for name in graphs}
    for regime, eviction in (("read", read), ("write", write), ("reuse", None)):
        for round_index in range(rounds):
            names = list(graphs)
            if round_index % 2:
                names.reverse()
            for name in names:
                value = _measure(graphs[name], eviction, samples)
                timings[name].setdefault(regime, []).append(value)
    medians = {
        name: {
            regime: statistics.median(values)
            for regime, values in backend_times.items()
        }
        for name, backend_times in timings.items()
    }
    return {
        "name": _case_name(case),
        "case": case,
        "selected": selected,
        "max_row_l2_vs_aiter": errors,
        "times_us": medians,
        "samples_us": timings,
        "speedup_vs_context": {
            regime: medians["context"][regime] / medians["segmented"][regime]
            for regime in ("read", "write", "reuse")
        },
        "speedup_vs_aiter": {
            regime: medians["aiter"][regime] / medians["segmented"][regime]
            for regime in ("read", "write", "reuse")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        raise ValueError("Samples and rounds must be positive")

    segmented_prefill.MAX_QUERY_LEN = EXPERIMENTAL_QUERY_LIMIT
    benchmark.MAX_QUERY_LEN = EXPERIMENTAL_QUERY_LIMIT
    cases = _cases()
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if _case_name(case) in requested]
        if len(cases) != len(requested):
            raise ValueError("Unknown or duplicate case name")
    root = Path(__file__).resolve().parents[2]
    result = {
        "metadata": {
            "query_lengths": list(QUERY_LENGTHS),
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "families": [list(family) for family in FAMILIES],
            "samples_per_round": args.samples,
            "rounds_per_regime": args.rounds,
            "experimental_query_limit": EXPERIMENTAL_QUERY_LIMIT,
            "production_query_limit": 128,
            "kernel_sha256": _sha256(
                root / "vllm/v1/attention/ops/segmented_prefill.py"
            ),
            "harness_sha256": _sha256(Path(__file__)),
        },
        "rows": [],
        "failures": [],
        "complete": False,
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["metadata"] != result["metadata"]:
            raise ValueError("Existing output metadata does not match")
        result["rows"] = previous["rows"]
        result["failures"] = previous["failures"]
    _save(args.output, result)
    done = {row["name"] for row in result["rows"] + result["failures"]}
    for case in cases:
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
        f"COMPLETE rows={len(result['rows'])} failures={len(result['failures'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
