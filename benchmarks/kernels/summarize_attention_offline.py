# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate balanced offline rounds and retain separate repeat diagnostics."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--available-rounds",
        action="store_true",
        help="Use only complete rounds shared by every variant for each KV dtype.",
    )
    args = parser.parse_args()
    variants = ("vanilla", "aiter", "ours")
    datasets = {}
    artifacts = {}
    rounds_used = {}
    repeatability = {}
    for kv in ("bf16", "fp8"):
        complete_rounds = []
        for round_number in (1, 2):
            paths = [
                args.root / f"round{round_number}-{variant}-{kv}/result.json"
                for variant in variants
            ]
            if all(
                p.exists() and len(json.loads(p.read_text())["rows"]) == 18
                for p in paths
            ):
                complete_rounds.append(round_number)
        if not args.available_rounds:
            assert complete_rounds == [1, 2], f"Incomplete comparison for {kv}"
        assert complete_rounds, f"No complete shared round for {kv}"
        rounds_used[kv] = complete_rounds
        for variant in variants:
            for round_number in (1, 2):
                path = args.root / f"round{round_number}-{variant}-{kv}/result.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                assert data["variant"] == variant and data["kv"] == kv
                datasets[kv, variant, round_number] = {
                    (r["input_len"], r["output_len"], r["batch"]): r
                    for r in data["rows"]
                }
                artifacts[str(path)] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": data["source"],
                    "commit": data["commit"],
                    "startup_s": data["engine_startup_s"],
                    "rows_completed": len(data["rows"]),
                    "used_in_comparison": round_number in complete_rounds,
                    "cache_unchanged_on_startup": (
                        data["tuning_cache_before_startup"]
                        == data["tuning_cache_after_startup"]
                    ),
                }
                if variant == "ours" and round_number == 2:
                    assert artifacts[str(path)]["cache_unchanged_on_startup"]
            first = datasets.get((kv, variant, 1), {})
            second = datasets.get((kv, variant, 2), {})
            common = first.keys() & second.keys()
            ratios = [
                second[k]["median_latency_s"] / first[k]["median_latency_s"]
                for k in common
            ]
            if ratios:
                repeatability[f"{kv}_{variant}"] = {
                    "shapes": len(ratios),
                    "round2_vs_round1_median": statistics.median(ratios),
                    "round2_vs_round1_min": min(ratios),
                    "round2_vs_round1_max": max(ratios),
                }

    rows = []
    for kv in ("bf16", "fp8"):
        shapes = sorted(datasets[kv, "vanilla", 1])
        assert len(shapes) == 18
        for shape in shapes:
            row = dict(kv=kv, input_len=shape[0], output_len=shape[1], batch=shape[2])
            tokens = {}
            for variant in variants:
                samples = [
                    s
                    for round_number in rounds_used[kv]
                    for s in datasets[kv, variant, round_number][shape]["samples"]
                ]
                assert len(samples) == 3 * len(rounds_used[kv])
                for sample in samples:
                    assert len(sample["ids"]) == shape[2]
                    assert all(len(ids) == shape[1] for ids in sample["ids"])
                latencies = [s["latency_s"] for s in samples]
                latency = statistics.median(latencies)
                row[f"{variant}_median_s"] = latency
                row[f"{variant}_min_s"] = min(latencies)
                row[f"{variant}_max_s"] = max(latencies)
                row[f"{variant}_output_tokens_per_s"] = shape[1] * shape[2] / latency
                row[f"{variant}_stable_outputs"] = all(
                    s["ids"] == samples[0]["ids"] for s in samples
                )
                tokens[variant] = samples[0]["ids"]
                second = datasets.get((kv, variant, 2), {}).get(shape)
                row[f"{variant}_round2_vs_round1"] = (
                    second["median_latency_s"]
                    / datasets[kv, variant, 1][shape]["median_latency_s"]
                    if second is not None
                    else None
                )
            row["aiter_speedup_vs_vanilla"] = (
                row["vanilla_median_s"] / row["aiter_median_s"]
            )
            row["ours_speedup_vs_vanilla"] = (
                row["vanilla_median_s"] / row["ours_median_s"]
            )
            row["ours_speedup_vs_aiter"] = row["aiter_median_s"] / row["ours_median_s"]
            row["winner"] = (
                "ours"
                if row["ours_speedup_vs_aiter"] > 1.02
                else "aiter"
                if row["ours_speedup_vs_aiter"] < 1 / 1.02
                else "within_2_percent"
            )
            for variant in ("aiter", "ours"):
                row[f"{variant}_exact_output_vs_vanilla"] = (
                    tokens[variant] == tokens["vanilla"]
                )
                row[f"{variant}_token_agreement_vs_vanilla"] = sum(
                    a == b
                    for xs, ys in zip(tokens[variant], tokens["vanilla"])
                    for a, b in zip(xs, ys)
                ) / (shape[1] * shape[2])
            rows.append(row)

    groups = {}
    for kv in ("bf16", "fp8"):
        groups[kv] = [r for r in rows if r["kv"] == kv]
        for batch in (1, 2, 4):
            groups[f"{kv}_batch{batch}"] = [
                r for r in groups[kv] if r["batch"] == batch
            ]
        for output_len in (32, 256):
            groups[f"{kv}_output{output_len}"] = [
                r for r in groups[kv] if r["output_len"] == output_len
            ]
    summary = {
        group: {
            "shapes": len(values),
            "aiter_speedup_vs_vanilla": geometric_mean(
                r["aiter_speedup_vs_vanilla"] for r in values
            ),
            "ours_speedup_vs_vanilla": geometric_mean(
                r["ours_speedup_vs_vanilla"] for r in values
            ),
            "ours_speedup_vs_aiter": geometric_mean(
                r["ours_speedup_vs_aiter"] for r in values
            ),
            "wins": {
                winner: sum(r["winner"] == winner for r in values)
                for winner in ("ours", "aiter", "within_2_percent")
            },
        }
        for group, values in groups.items()
    }
    (args.root / "summary.json").write_text(
        json.dumps(
            {
                "rounds_used": rounds_used,
                "samples_per_variant_shape": {
                    kv: 3 * len(rounds) for kv, rounds in rounds_used.items()
                },
                "repeatability": repeatability,
                "groups": summary,
                "rows": rows,
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n"
    )
    with (args.root / "comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
