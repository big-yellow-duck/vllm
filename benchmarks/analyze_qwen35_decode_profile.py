# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze Qwen3.5 forward kernel time from a torch profiler trace.

The profile produced by ``benchmarks/profile_qwen35_gdn.py`` contains a GPU
annotation for each model forward. This script filters CUDA kernels/memcpy
events to one forward range and reports mutually exclusive time buckets.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


CUDA_EVENT_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
PHASE_RANGE_PREFIX = {
    "decode": "execute_context_0(0)_generation_",
    "prefill": "execute_context_1(",
}


def _load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _find_trace(profile_path: Path) -> Path:
    if profile_path.is_file():
        return profile_path

    trace_dir = profile_path / "torch_profiler"
    if not trace_dir.exists():
        trace_dir = profile_path

    traces = sorted(trace_dir.glob("*.pt.trace.json*"))
    if not traces:
        raise FileNotFoundError(f"No torch profiler trace found under {trace_dir}")
    if len(traces) > 1:
        raise ValueError(
            f"Multiple torch profiler traces found under {trace_dir}; pass one file"
        )
    return traces[0]


def _classify_event(name: str) -> str:
    if "flash_fwd_splitkv" in name or "_vllm_fa2_C::varlen_fwd" in name:
        return "full_attention"
    if (
        "gated_delta_rule" in name
        or "recompute_w_u_fwd_kernel" in name
        or "chunk_fwd_kernel_o" in name
        or "chunk_scaled_dot_kkt_fwd_kernel" in name
        or "chunk_local_cumsum_scalar_kernel" in name
        or "causal_conv1d_update" in name
        or "_causal_conv1d_fwd_kernel" in name
        or "_fused_post_conv_kernel" in name
        or "merge_16x16_to_64x64_inverse_kernel" in name
        or "qwen_gdn_attention_core" in name
        or "gdn_decode" in name
    ):
        return "gdn"
    return "other"


def _event_is_inside(event: dict[str, Any], start_us: float, end_us: float) -> bool:
    event_start = float(event.get("ts", 0.0))
    event_end = event_start + float(event.get("dur", 0.0))
    return event_start >= start_us and event_end <= end_us


def _find_forward_ranges(
    events: list[dict[str, Any]],
    phase: str,
    range_name: str | None,
) -> list[dict[str, Any]]:
    range_prefix = PHASE_RANGE_PREFIX[phase]
    ranges = [
        event
        for event in events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("name", "").startswith(range_prefix)
    ]
    if range_name is not None:
        ranges = [event for event in ranges if event.get("name") == range_name]
    if not ranges:
        raise ValueError(f"No matching {phase} GPU annotation found in trace")
    return sorted(ranges, key=lambda event: event["ts"])


def _analyze_range(
    events: list[dict[str, Any]],
    selected: dict[str, Any],
    trace_path: Path,
    phase: str,
) -> dict[str, Any]:
    start_us = float(selected["ts"])
    end_us = start_us + float(selected["dur"])

    cuda_events = [
        event
        for event in events
        if event.get("cat") in CUDA_EVENT_CATEGORIES
        and _event_is_inside(event, start_us, end_us)
    ]

    bucket_us: Counter[str] = Counter()
    event_us: Counter[str] = Counter()
    event_calls: Counter[str] = Counter()
    bucket_calls: Counter[str] = Counter()
    for event in cuda_events:
        name = event["name"]
        bucket = _classify_event(name)
        dur_us = float(event["dur"])
        bucket_us[bucket] += dur_us
        bucket_calls[bucket] += 1
        event_us[name] += dur_us
        event_calls[name] += 1

    forward_us = float(selected["dur"])
    residual_us = max(0.0, forward_us - sum(bucket_us.values()))
    buckets = []
    for name in ("full_attention", "gdn", "other"):
        total_us = float(bucket_us[name])
        buckets.append(
            {
                "name": name,
                "time_us": total_us,
                "time_ms": total_us / 1000.0,
                "pct_of_forward_gpu_range": (total_us / forward_us * 100.0)
                if forward_us
                else 0.0,
                "kernel_or_memcpy_calls": bucket_calls[name],
            }
        )
    other_including_gap_us = float(bucket_us["other"]) + residual_us

    top_events = [
        {
            "name": name,
            "category": _classify_event(name),
            "time_us": float(total_us),
            "time_ms": float(total_us) / 1000.0,
            "pct_of_forward_gpu_range": float(total_us) / forward_us * 100.0,
            "calls": event_calls[name],
        }
        for name, total_us in event_us.most_common()
    ]

    return {
        "trace_path": str(trace_path),
        "phase": phase,
        "range_mode": "single",
        "range_name": selected["name"],
        "forward_time_us": forward_us,
        "forward_time_ms": forward_us / 1000.0,
        "cuda_event_time_sum_us": sum(bucket_us.values()),
        "cuda_event_time_sum_ms": sum(bucket_us.values()) / 1000.0,
        "unattributed_gap_us": residual_us,
        "unattributed_gap_ms": residual_us / 1000.0,
        "cuda_events_count": len(cuda_events),
        "buckets": buckets,
        "other_including_unattributed_gap": {
            "time_us": other_including_gap_us,
            "time_ms": other_including_gap_us / 1000.0,
            "pct_of_forward_gpu_range": other_including_gap_us / forward_us * 100.0,
        },
        "top_events": top_events,
        "classification": {
            "full_attention": [
                "kernel names containing flash_fwd_splitkv",
                "operator names containing _vllm_fa2_C::varlen_fwd",
            ],
            "gdn": [
                "kernel names containing gated_delta_rule",
                "known GDN prefill/decode helper kernels",
                "operator/range names containing qwen_gdn_attention_core or gdn_decode",
            ],
            "other": ["all remaining CUDA kernel/memcpy/memset events"],
        },
    }


def _aggregate_results(results: list[dict[str, Any]], trace_path: Path, phase: str) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate an empty result list")

    num_ranges = len(results)
    bucket_total_us: Counter[str] = Counter()
    bucket_calls: Counter[str] = Counter()
    event_total_us: Counter[str] = Counter()
    event_calls: Counter[str] = Counter()
    forward_times_us = [float(result["forward_time_us"]) for result in results]

    for result in results:
        for bucket in result["buckets"]:
            bucket_total_us[bucket["name"]] += float(bucket["time_us"])
            bucket_calls[bucket["name"]] += int(bucket["kernel_or_memcpy_calls"])
        for event in result["top_events"]:
            event_total_us[event["name"]] += float(event["time_us"])
            event_calls[event["name"]] += int(event["calls"])

    total_forward_us = sum(forward_times_us)
    cuda_event_time_sum_us = sum(
        float(result["cuda_event_time_sum_us"]) for result in results
    )
    unattributed_gap_us = sum(float(result["unattributed_gap_us"]) for result in results)

    buckets = []
    for name in ("full_attention", "gdn", "other"):
        total_us = float(bucket_total_us[name])
        buckets.append(
            {
                "name": name,
                "total_time_us": total_us,
                "total_time_ms": total_us / 1000.0,
                "avg_time_us": total_us / num_ranges,
                "avg_time_ms": total_us / num_ranges / 1000.0,
                "pct_of_total_forward_gpu_range": total_us
                / total_forward_us
                * 100.0,
                "kernel_or_memcpy_calls": bucket_calls[name],
                "avg_kernel_or_memcpy_calls": bucket_calls[name] / num_ranges,
            }
        )

    other_including_gap_us = float(bucket_total_us["other"]) + unattributed_gap_us
    top_events = [
        {
            "name": name,
            "category": _classify_event(name),
            "total_time_us": float(total_us),
            "total_time_ms": float(total_us) / 1000.0,
            "avg_time_us": float(total_us) / num_ranges,
            "avg_time_ms": float(total_us) / num_ranges / 1000.0,
            "pct_of_total_forward_gpu_range": float(total_us)
            / total_forward_us
            * 100.0,
            "calls": event_calls[name],
            "avg_calls": event_calls[name] / num_ranges,
        }
        for name, total_us in event_total_us.most_common()
    ]

    return {
        "trace_path": str(trace_path),
        "phase": phase,
        "range_mode": "all",
        "range_count": num_ranges,
        "first_range_name": results[0]["range_name"],
        "last_range_name": results[-1]["range_name"],
        "forward_time_total_us": total_forward_us,
        "forward_time_total_ms": total_forward_us / 1000.0,
        "forward_time_avg_us": total_forward_us / num_ranges,
        "forward_time_avg_ms": total_forward_us / num_ranges / 1000.0,
        "forward_time_min_ms": min(forward_times_us) / 1000.0,
        "forward_time_max_ms": max(forward_times_us) / 1000.0,
        "cuda_event_time_sum_us": cuda_event_time_sum_us,
        "cuda_event_time_sum_ms": cuda_event_time_sum_us / 1000.0,
        "cuda_event_time_avg_ms": cuda_event_time_sum_us / num_ranges / 1000.0,
        "unattributed_gap_us": unattributed_gap_us,
        "unattributed_gap_ms": unattributed_gap_us / 1000.0,
        "unattributed_gap_avg_ms": unattributed_gap_us / num_ranges / 1000.0,
        "buckets": buckets,
        "other_including_unattributed_gap": {
            "total_time_us": other_including_gap_us,
            "total_time_ms": other_including_gap_us / 1000.0,
            "avg_time_us": other_including_gap_us / num_ranges,
            "avg_time_ms": other_including_gap_us / num_ranges / 1000.0,
            "pct_of_total_forward_gpu_range": other_including_gap_us
            / total_forward_us
            * 100.0,
        },
        "top_events": top_events,
        "per_range": [
            {
                "range_name": result["range_name"],
                "forward_time_ms": result["forward_time_ms"],
                "cuda_event_time_sum_ms": result["cuda_event_time_sum_ms"],
                "unattributed_gap_ms": result["unattributed_gap_ms"],
                "buckets": result["buckets"],
            }
            for result in results
        ],
        "classification": results[0]["classification"],
    }


def analyze_forward_trace(
    trace_path: Path,
    phase: str,
    range_name: str | None,
    range_mode: str,
) -> dict[str, Any]:
    trace = _load_trace(trace_path)
    events = trace["traceEvents"]
    ranges = _find_forward_ranges(events, phase, range_name)
    if range_mode == "last":
        return _analyze_range(events, ranges[-1], trace_path, phase)

    results = [_analyze_range(events, selected, trace_path, phase) for selected in ranges]
    return _aggregate_results(results, trace_path, phase)


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    phase_title = result["phase"].capitalize()
    if result["range_mode"] == "all":
        lines = [
            f"# Qwen3.5 {phase_title} Profile Breakdown",
            "",
            f"- trace: `{result['trace_path']}`",
            f"- range mode: `all`",
            f"- range count: `{result['range_count']}`",
            f"- first range: `{result['first_range_name']}`",
            f"- last range: `{result['last_range_name']}`",
            f"- average forward GPU range: `{result['forward_time_avg_ms']:.6f} ms`",
            f"- min/max forward GPU range: `{result['forward_time_min_ms']:.6f} ms` / `{result['forward_time_max_ms']:.6f} ms`",
            f"- total forward GPU range: `{result['forward_time_total_ms']:.6f} ms`",
            f"- average CUDA kernel/memcpy/memset sum: `{result['cuda_event_time_avg_ms']:.6f} ms`",
            f"- average unattributed gap inside range: `{result['unattributed_gap_avg_ms']:.6f} ms`",
            "",
            "## Average Buckets",
            "",
            "| bucket | avg time (ms) | pct of total forward ranges | total time (ms) | total calls | avg calls |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for bucket in result["buckets"]:
            lines.append(
                "| {name} | {avg_time_ms:.6f} | "
                "{pct_of_total_forward_gpu_range:.2f}% | "
                "{total_time_ms:.6f} | {kernel_or_memcpy_calls} | "
                "{avg_kernel_or_memcpy_calls:.2f} |".format(**bucket)
            )
        other_with_gap = result["other_including_unattributed_gap"]
        lines.append(
            "| other_including_unattributed_gap | {avg_time_ms:.6f} | "
            "{pct_of_total_forward_gpu_range:.2f}% | {total_time_ms:.6f} | "
            "- | - |".format(**other_with_gap)
        )
        lines.extend(
            [
                "",
                "## Top Events",
                "",
                "| category | avg time (ms) | pct of total forward ranges | total calls | avg calls | event |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for event in result["top_events"][:20]:
            lines.append(
                "| {category} | {avg_time_ms:.6f} | "
                "{pct_of_total_forward_gpu_range:.2f}% | {calls} | "
                "{avg_calls:.2f} | `{name}` |".format(**event)
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = [
        f"# Qwen3.5 {phase_title} Profile Breakdown",
        "",
        f"- trace: `{result['trace_path']}`",
        f"- range: `{result['range_name']}`",
        f"- forward GPU range: `{result['forward_time_ms']:.6f} ms`",
        f"- CUDA kernel/memcpy/memset sum: `{result['cuda_event_time_sum_ms']:.6f} ms`",
        f"- unattributed gap inside range: `{result['unattributed_gap_ms']:.6f} ms`",
        "",
        "## Buckets",
        "",
        "| bucket | time (ms) | pct of forward range | calls |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in result["buckets"]:
        lines.append(
            "| {name} | {time_ms:.6f} | {pct_of_forward_gpu_range:.2f}% | "
            "{kernel_or_memcpy_calls} |".format(**bucket)
        )
    other_with_gap = result["other_including_unattributed_gap"]
    lines.append(
        "| other_including_unattributed_gap | {time_ms:.6f} | "
        "{pct_of_forward_gpu_range:.2f}% | - |".format(**other_with_gap)
    )

    lines.extend(
        [
            "",
            "## Top Events",
            "",
            "| category | time (ms) | pct of forward range | calls | event |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for event in result["top_events"][:20]:
        lines.append(
            "| {category} | {time_ms:.6f} | {pct_of_forward_gpu_range:.2f}% | "
            "{calls} | `{name}` |".format(**event)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile_path",
        type=Path,
        help="Profile directory, torch_profiler directory, or trace file.",
    )
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill"),
        default="decode",
        help="Forward phase to analyze.",
    )
    parser.add_argument(
        "--range-name",
        default=None,
        help="Exact GPU annotation name. Defaults to the last matching phase range.",
    )
    parser.add_argument(
        "--range-mode",
        choices=("last", "all"),
        default="last",
        help="Analyze only the last matching range, or aggregate all matching ranges.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = _find_trace(args.profile_path)
    result = analyze_forward_trace(
        trace_path,
        args.phase,
        args.range_name,
        args.range_mode,
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(result, args.markdown_output)

    printable = dict(result)
    printable["top_events"] = result["top_events"][: args.top]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
