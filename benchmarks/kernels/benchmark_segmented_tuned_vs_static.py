# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-benchmark a persisted segmented-attention table against static configs."""

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch

from vllm.v1.attention.ops import segmented_prefill, segmented_prefill_tuning


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."))
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unsupported dtype {name}")
    return dtype


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _measure(run, eviction: torch.Tensor, samples: int) -> float:
    stream = torch.cuda.current_stream()
    values = []
    for _ in range(samples):
        eviction.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        run()
        end.record(stream)
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000)
    return statistics.median(values)


@torch.inference_mode()
def _probe(
    identity: dict,
    record: dict,
    device: torch.device,
    eviction: torch.Tensor,
    samples: int,
    rounds: int,
) -> dict:
    workload = tuple(record["workload"])
    batch, query_len, seq_len = workload
    dtype = _torch_dtype(identity["dtype"])
    kv_dtype = _torch_dtype(identity["kv_dtype"])
    heads = identity["heads"]
    kv_heads = identity["kv_heads"]
    dim = identity["dim"]
    q, _k, _v, kc, vc, table, starts, lengths, kv_scale = (
        segmented_prefill_tuning._make_inputs(
            device,
            dtype,
            kv_dtype,
            heads,
            kv_heads,
            dim,
            identity["page"],
            identity["max_tokens"],
            workload,
        )
    )
    static = segmented_prefill.select_segmented_config(
        batch,
        query_len,
        seq_len,
        heads,
        kv_heads,
        dim,
        kv_dtype.itemsize == 1,
    )
    if static != record["default"]:
        raise AssertionError("Persisted static config no longer matches the selector")
    tuned = record["best"]
    shapes = segmented_prefill.segmented_workspace_shapes(
        batch,
        segmented_prefill.segmented_query_capacity(query_len),
        heads,
        kv_heads,
        dim,
        static["splits"],
    )
    workspace = (
        None
        if shapes is None
        else tuple(
            torch.empty(shape, dtype=torch.float32, device=device) for shape in shapes
        )
    )
    outputs = {name: torch.empty_like(q) for name in ("static", "tuned")}

    def run(name: str) -> None:
        segmented_prefill.segmented_prefill_attention(
            q,
            outputs[name],
            kc,
            vc,
            table,
            starts,
            lengths,
            query_len,
            seq_len,
            kv_scale,
            kv_scale,
            identity["scale"],
            skip_decode=False,
            config=static if name == "static" else tuned,
            workspace=workspace,
        )

    for name in outputs:
        outputs[name].fill_(float("nan"))
        run(name)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(
        outputs["tuned"], outputs["static"], atol=0.01, rtol=0.01
    )
    difference = outputs["tuned"].float() - outputs["static"].float()
    relative_l2 = (
        difference.norm() / outputs["static"].float().norm().clamp_min(1e-6)
    ).item()

    timings = {"static": [], "tuned": []}
    for round_index in range(rounds):
        order = ("static", "tuned") if round_index % 2 == 0 else ("tuned", "static")
        for name in order:
            timings[name].append(
                _measure(lambda name=name: run(name), eviction, samples)
            )
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "workload": list(workload),
        "query_lengths": record["query_lengths"],
        "static": static,
        "tuned": tuned,
        "relative_l2": relative_l2,
        "times_us": medians,
        "round_medians_us": timings,
        "speedup": medians["static"] / medians["tuned"],
    }


def _summary(rows: list[dict]) -> dict:
    speedups = [row["speedup"] for row in rows]
    return {
        "rows": len(rows),
        "wins_over_1pct": sum(value > 1.01 for value in speedups),
        "within_1pct": sum(0.99 <= value <= 1.01 for value in speedups),
        "losses_over_1pct": sum(value < 0.99 for value in speedups),
        "geomean_speedup": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--workload",
        action="append",
        default=[],
        help="Restrict to B,Q,S (repeatable)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        raise ValueError("Samples and rounds must be positive")

    table = json.loads(args.cache.read_text())
    requested = {
        tuple(int(value) for value in workload.split(",")) for workload in args.workload
    }
    if any(len(workload) != 3 for workload in requested):
        raise ValueError("Each workload must contain B,Q,S")
    result = {
        "metadata": {
            "cache": str(args.cache),
            "identity": table["identity"],
            "samples": args.samples,
            "rounds": args.rounds,
            "workloads": sorted(requested),
            "timing": "alternating HIP events with 256 MiB write eviction",
        },
        "rows": [],
        "failures": [],
        "summary": None,
        "complete": False,
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["metadata"] != result["metadata"]:
            raise ValueError("Existing output metadata does not match")
        result = previous

    done = {tuple(row["workload"]) for row in result["rows"] + result["failures"]}
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    eviction = torch.empty(256 * 1024**2, dtype=torch.int8, device=device)
    records = sorted(table["records"], key=lambda record: record["workload"])
    if requested:
        records = [
            record for record in records if tuple(record["workload"]) in requested
        ]
        if len(records) != len(requested):
            raise ValueError("Unknown or duplicate requested workload")
    for record in records:
        workload = tuple(record["workload"])
        if workload in done:
            continue
        try:
            row = _probe(
                table["identity"],
                record,
                device,
                eviction,
                args.samples,
                args.rounds,
            )
        except Exception as error:
            row = {"workload": list(workload), "error": repr(error)}
            result["failures"].append(row)
            print("FAILED " + json.dumps(row), flush=True)
        else:
            result["rows"].append(row)
            print("RESULT " + json.dumps(row), flush=True)
        result["summary"] = _summary(result["rows"]) if result["rows"] else None
        _save(args.output, result)
        gc.collect()
        torch.accelerator.empty_cache()
    result["summary"] = _summary(result["rows"]) if result["rows"] else None
    result["complete"] = len(result["rows"]) == len(records) and not result["failures"]
    _save(args.output, result)
    print("COMPLETE " + json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
