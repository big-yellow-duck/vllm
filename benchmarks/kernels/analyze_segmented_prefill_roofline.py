# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build an optimistic HBM/compute roofline from segmented-prefill results.

The byte model counts unique, compulsory KV reads, one Q read, one output write,
and both sides of the split workspace. Repeated KV reads by separate query tiles
are intentionally not charged to HBM: they can be served by the on-chip cache
hierarchy and require hardware counters for an exact attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _model(row: dict, bandwidth_gbps: float, matrix_tflops: float) -> dict:
    case = row["case"]
    config = row["selected"]["segmented"]
    heads = case["heads"]
    kv_heads = case["kv_heads"]
    dim = 256
    cache_element_bytes = 1 if case["fp8"] else 2
    flops = 0
    kv_bytes = 0
    q_output_bytes = 0
    workspace_bytes = 0
    for query, context in zip(case["queries"], case["contexts"]):
        sequence = query + context
        attended_pairs = query * context + query * (query + 1) // 2
        # QK and PV each perform one multiply and one add.
        flops += 4 * heads * dim * attended_pairs
        kv_bytes += 2 * kv_heads * sequence * dim * cache_element_bytes
        q_output_bytes += 2 * query * heads * dim * 2
        if config["splits"] > 1:
            # The stage writes FP32 partials/LSE and the reducer reads them.
            workspace_bytes += 2 * query * heads * config["splits"] * (dim + 1) * 4

    compulsory_bytes = kv_bytes + q_output_bytes + workspace_bytes
    intensity = flops / compulsory_bytes
    memory_roof_tflops = intensity * bandwidth_gbps / 1000
    roof_tflops = min(matrix_tflops, memory_roof_tflops)
    time_us = row["times_us"]["segmented"]["read"]
    achieved_tflops = flops / (time_us * 1e-6) / 1e12
    efficiency = achieved_tflops / roof_tflops
    return {
        "name": row["name"],
        "dtype": "fp8" if case["fp8"] else "bf16",
        "tp": row["name"].split("-", 1)[0],
        "query": case["queries"][0],
        "batch": len(case["queries"]),
        "sequence": case["queries"][0] + case["contexts"][0],
        "heads": heads,
        "kv_heads": kv_heads,
        "bm": config["bm"],
        "bn": config["bn"],
        "bk": config["bk"],
        "splits": config["splits"],
        "useful_tflop": flops / 1e12,
        "compulsory_gb": compulsory_bytes / 1e9,
        "workspace_mb": workspace_bytes / 1e6,
        "arithmetic_intensity": intensity,
        "bound": "memory" if memory_roof_tflops < matrix_tflops else "compute",
        "roof_tflops": roof_tflops,
        "time_us": time_us,
        "achieved_tflops": achieved_tflops,
        "equivalent_hbm_gbps": compulsory_bytes / (time_us * 1e-6) / 1e9,
        "roof_efficiency_pct": 100 * efficiency,
        "roof_headroom_x": 1 / efficiency,
        "roof_gap_pct": 100 * (1 - efficiency),
    }


def _group_summary(rows: list[dict]) -> dict:
    efficiency = [row["roof_efficiency_pct"] for row in rows]
    achieved = [row["achieved_tflops"] for row in rows]
    intensity = [row["arithmetic_intensity"] for row in rows]
    return {
        "cases": len(rows),
        "memory_bound": sum(row["bound"] == "memory" for row in rows),
        "compute_bound": sum(row["bound"] == "compute" for row in rows),
        "efficiency_pct": {
            "min": min(efficiency),
            "p25": _percentile(efficiency, 0.25),
            "median": statistics.median(efficiency),
            "p75": _percentile(efficiency, 0.75),
            "max": max(efficiency),
        },
        "achieved_tflops_median": statistics.median(achieved),
        "arithmetic_intensity_median": statistics.median(intensity),
    }


def _plot(rows: list[dict], path: Path, bandwidth_gbps: float, matrix_tflops: float):
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    colors = {2: "#4C78A8", 8: "#54A24B", 32: "#F58518", 128: "#E45756"}
    x_min = min(row["arithmetic_intensity"] for row in rows) * 0.75
    x_max = max(row["arithmetic_intensity"] for row in rows) * 1.35
    intensity = np.geomspace(x_min, x_max, 512)
    roof = np.minimum(matrix_tflops, intensity * bandwidth_gbps / 1000)
    for axis, dtype in zip(axes, ("bf16", "fp8")):
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        axis.plot(intensity, roof, color="#222222", linewidth=2.0, label="roofline")
        for query in (2, 8, 32, 128):
            query_rows = [row for row in dtype_rows if row["query"] == query]
            axis.scatter(
                [row["arithmetic_intensity"] for row in query_rows],
                [row["achieved_tflops"] for row in query_rows],
                color=colors[query],
                edgecolor="white",
                linewidth=0.5,
                s=45,
                alpha=0.9,
                label=f"Q={query}",
            )
        axis.axvline(
            matrix_tflops * 1000 / bandwidth_gbps,
            color="#777777",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.2)
        axis.set_title(dtype.upper())
        axis.set_xlabel("Arithmetic intensity (useful FLOP / compulsory byte)")
    axes[0].set_ylabel("Useful throughput (TFLOP/s)")
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "R9700 Triton segmented-prefill roofline\n"
        "read-evicted latency; 640 GB/s HBM and 191 TFLOP/s BF16 matrix peaks"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--bandwidth-gbps", type=float, default=640.0)
    parser.add_argument("--matrix-tflops", type=float, default=191.0)
    args = parser.parse_args()

    source_rows = []
    source_metadata = []
    for path in args.inputs:
        result = json.loads(path.read_text())
        if result["failures"]:
            raise ValueError(f"{path} contains benchmark failures")
        source_rows.extend(result["rows"])
        source_metadata.append(
            {
                "path": str(path),
                "kernel_sha256": result["metadata"]["kernel_sha256"],
                "cases": len(result["rows"]),
            }
        )
    if not source_rows:
        raise ValueError("No benchmark rows found")
    hashes = {item["kernel_sha256"] for item in source_metadata}
    if len(hashes) != 1:
        raise ValueError(f"Input files have different kernel hashes: {hashes}")

    rows = [_model(row, args.bandwidth_gbps, args.matrix_tflops) for row in source_rows]
    rows.sort(
        key=lambda row: (
            row["dtype"],
            row["query"],
            row["tp"],
            row["batch"],
            row["sequence"],
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": {
            "timing_regime": "read-evicted CUDA-graph median",
            "byte_model": "unique compulsory HBM bytes plus split workspace",
            "flop_model": "useful QK and PV multiply-add operations",
            "bandwidth_gbps": args.bandwidth_gbps,
            "matrix_tflops": args.matrix_tflops,
            "fp8_note": "FP8 KV is converted before BF16 matrix dot; BF16 peak applies",
        },
        "sources": source_metadata,
        "all": _group_summary(rows),
        "by_dtype": {},
    }
    for dtype in ("bf16", "fp8"):
        dtype_rows = [row for row in rows if row["dtype"] == dtype]
        summary["by_dtype"][dtype] = {
            **_group_summary(dtype_rows),
            "by_query": {
                str(query): _group_summary(
                    [row for row in dtype_rows if row["query"] == query]
                )
                for query in (2, 8, 32, 128)
            },
            "lowest_efficiency": sorted(
                dtype_rows, key=lambda row: row["roof_efficiency_pct"]
            )[:5],
            "highest_efficiency": sorted(
                dtype_rows,
                key=lambda row: row["roof_efficiency_pct"],
                reverse=True,
            )[:5],
        }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_plot:
        _plot(rows, args.output_plot, args.bandwidth_gbps, args.matrix_tflops)


if __name__ == "__main__":
    main()
