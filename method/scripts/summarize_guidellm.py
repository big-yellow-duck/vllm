#!/usr/bin/env python3
"""Print a baseline-to-candidate Markdown table from two GuideLLM reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    ("output_tokens_per_second", "Output tok/s", False),
    ("time_to_first_token_ms", "Mean TTFT (ms)", True),
    ("request_latency", "Mean E2E (s)", True),
)


def load_means(path: Path) -> dict[int, dict[str, float]]:
    data = json.loads(path.read_text())
    result: dict[int, dict[str, float]] = {}
    for benchmark in data.get("benchmarks", []):
        strategy = benchmark.get("config", {}).get("strategy", {})
        streams = strategy.get("streams", strategy.get("max_concurrency"))
        requests = benchmark.get("requests", {}).get("successful", [])
        if streams is None or not requests:
            continue
        result[int(streams)] = {
            field: sum(float(request[field]) for request in requests) / len(requests)
            for field, _, _ in FIELDS
        }
    if not result:
        raise ValueError(f"No successful request data found in {path}")
    return result


def change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def format_pair(field: str, baseline: float, candidate: float) -> str:
    if field == "time_to_first_token_ms":
        return f"{baseline:,.0f} → {candidate:,.0f}"
    return f"{baseline:.2f} → {candidate:.2f}"


def format_change(value: float, lower_is_better: bool) -> str:
    text = f"{value:+.1f}%".replace("-", "−")
    improved = value < 0 if lower_is_better else value > 0
    return f"**{text}**" if improved else text


def validate_compatible(
    baseline_path: Path,
    candidate_path: Path,
    baseline: dict[int, dict[str, float]],
    candidate: dict[int, dict[str, float]],
) -> None:
    if baseline.keys() != candidate.keys():
        raise ValueError(
            "Configured stream levels differ: "
            f"{baseline_path} has {sorted(baseline)}; "
            f"{candidate_path} has {sorted(candidate)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = load_means(args.baseline)
    candidate = load_means(args.candidate)
    validate_compatible(args.baseline, args.candidate, baseline, candidate)

    headers = ["Streams"]
    separators = ["---:"]
    for _, label, _ in FIELDS:
        headers.extend([f"{label}, baseline → candidate", f"{label} change"])
        separators.extend(["---:", "---:"])
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(separators) + "|")
    for streams in sorted(baseline):
        cells: list[str] = [str(streams)]
        for field, _, lower_is_better in FIELDS:
            base_value = baseline[streams][field]
            candidate_value = candidate[streams][field]
            cells.extend(
                [
                    format_pair(field, base_value, candidate_value),
                    format_change(
                        change(candidate_value, base_value),
                        lower_is_better,
                    ),
                ]
            )
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
