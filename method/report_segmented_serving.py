"""Create per-model serving comparisons with the bench collection skill."""

import copy
import importlib.util
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

import run_segmented_serving as study

SKILL = study.BENCH / "guidellm-benchmark-compare"


def relative(path, directory):
    return os.path.relpath(path, directory)


def merged_reports(cases, target):
    reports = []
    counts = []
    for case in cases:
        folder = study.case_folder(case)
        assert (folder / "success.json").is_file(), folder
        path = folder / "agent-multiturn-benchmarks.json"
        output_tokens = 128 if case["workload"] == "short" else 512
        counts.append(study.validate_report(path, output_tokens))
        reports.append(study.read(path))
    result = copy.deepcopy(reports[0])
    result["metadata"]["derived_request_pool"] = {
        "sources": [
            str(study.case_folder(c) / "agent-multiturn-benchmarks.json") for c in cases
        ],
        "method": "Concatenate successful request records by stream count; all requests including warmup/cooldown; discard stale benchmark summary metrics.",
    }
    merged = []
    for streams in (1, 2, 4, 8):
        first = next(
            b
            for b in reports[0]["benchmarks"]
            if int(b["config"]["strategy"]["streams"]) == streams
        )
        row = {
            "config": copy.deepcopy(first["config"]),
            "requests": {"successful": [], "errored": [], "incomplete": []},
        }
        for repeat, data in enumerate(reports, start=1):
            bench = next(
                b
                for b in data["benchmarks"]
                if int(b["config"]["strategy"]["streams"]) == streams
            )
            for record in bench["requests"]["successful"]:
                request = copy.deepcopy(record)
                request["info"]["conversation_id"] = (
                    f"rep{repeat}:{request['info']['conversation_id']}"
                )
                row["requests"]["successful"].append(request)
        merged.append(row)
    result["benchmarks"] = merged
    study.write(target, result)
    return counts


def workload_identity(path):
    data = study.read(path)
    spec = data["config"]["spec"]
    return {
        "models": sorted({b["config"]["backend"]["model"] for b in data["benchmarks"]}),
        "data": spec["data"],
        "streams": spec["profile"]["streams"],
        "constraints": spec["constraints"],
        "seed": spec["seed"],
    }


def counters(path):
    result = defaultdict(float)
    for line in path.read_text().splitlines():
        match = re.match(
            r"(vllm:spec_decode_num_(?:draft_tokens|accepted_tokens|drafts)_total)(?:\{[^}]*\})?\s+([-+0-9.eE]+)",
            line,
        )
        if match:
            result[match[1]] += float(match[2])
    return result


def acceptance(case):
    folder = study.case_folder(case)
    before, after = [
        counters(folder / f"metrics-{stage}.txt") for stage in ("before", "after")
    ]
    delta = {key: after[key] - before.get(key, 0) for key in after}
    proposed = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return {
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
    }


def compare(manifest, model, name, baseline, candidate, labels):
    slug = study.SLUGS[model]
    directory = (
        study.BENCH / "comparisons/gfx1201" / slug / "segmented-update-20261016" / name
    )
    directory.mkdir(parents=True, exist_ok=True)
    plot_dir = study.BENCH / "plots/gfx1201" / slug / "segmented-update-20261016" / name
    paths = [
        directory / f"{label}-requests.json" for label in ("baseline", "candidate")
    ]
    counts = [
        merged_reports(cases, path)
        for cases, path in zip((baseline, candidate), paths, strict=True)
    ]
    assert workload_identity(paths[0]) == workload_identity(paths[1])
    title = f"{manifest['models'][model]['model']} · TP2 · {name}"
    subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts/plot_compare_multiturn.py"),
            "--report",
            f"{labels[0]}={paths[0]}",
            "--report",
            f"{labels[1]}={paths[1]}",
            "--title",
            title,
            "--output-dir",
            str(plot_dir),
        ],
        check=True,
    )
    table = subprocess.check_output(
        [
            sys.executable,
            str(SKILL / "scripts/summarize_guidellm.py"),
            *map(str, paths),
        ],
        text=True,
    )
    spec = importlib.util.spec_from_file_location(
        "guidellm_skill_summary", SKILL / "scripts/summarize_guidellm.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    means = [helper.load_means(path) for path in paths]
    changes = {
        n: {
            field: (means[1][n][field] / means[0][n][field] - 1) * 100
            for field, _, _ in helper.FIELDS
        }
        for n in means[0]
    }
    backend_counts = []
    for case in (baseline[0], candidate[0]):
        inspection = study.read(study.case_folder(case) / "inspection.json")["results"]
        backend_counts.append(
            [
                dict(Counter(r["backend"] for r in rank["attention_layers"]))
                for rank in inspection
            ]
        )
    token_count = 128 if baseline[0]["workload"] == "short" else 512
    if model.startswith("qwen"):
        change_description = (
            "This measures the combined segmented source update: FP8 conversion "
            "routing, configuration selection and compressed autotuning."
        )
    elif baseline[0]["arm"] == candidate[0]["arm"]:
        change_description = (
            "Speculation is the experimental change; the attention backend matches. "
            "The MTP source snapshot adds host support for attention layers that read "
            "the target KV cache without projected K/V. Compiled segmented kernel "
            "cache keys, configuration selection and the existing projected-K/V "
            "dispatch path are unchanged, as validated in the linked fix report."
        )
    else:
        change_description = "The attention backend is the experimental change described by the arm labels."
    lines = [
        f"# {title}",
        "",
        f"{labels[0]} → {labels[1]}. These are end-to-end serving results on two Radeon AI PRO R9700 GPUs (gfx1201), with TP2, BF16 activations, FP8 model weights and explicit FP8 KV cache in every arm.",
        "",
        f"Model revision: `{manifest['models'][model]['revision']}`. Workload: eight turns, 2,048 new synthetic prompt tokens and {token_count} output tokens per turn; streams 1/2/4/8; seed 20260715; prefix caching enabled. {len(baseline)} fresh server repetitions per arm. {change_description}",
        "",
        f"Segmented startup autotuning: `{manifest['autotune']}`. Main model limits: context 32,768, eight sequences, batched-token limit 8,192, GPU memory utilization 0.90. Backend choice was inspected on both workers; automatic selection uses no attention-backend flag.",
        "",
        f"Observed baseline backends by rank: `{backend_counts[0]}`. Candidate: `{backend_counts[1]}`.",
        "",
        "Model, data definition, streams, request limits and seed match. Every repetition contains 8/16/32/64 successful turn requests and no errored or incomplete requests. The table and charts use means over every successful request, including warmup/cooldown, through the skill's summary and plotting helpers. Positive TPS changes and negative TTFT/E2E changes improve performance.",
        "",
        table.rstrip(),
        "",
    ]
    for field, metric, _ in helper.FIELDS:
        values = [row[field] for row in changes.values()]
        lines.append(
            f"{metric} changes range from {min(values):+.1f}% to {max(values):+.1f}% across the four stream counts."
        )
    per_repetition = []
    if len(baseline) > 1:
        lines.extend(
            [
                "",
                "Per-repetition means retain every successful request and expose variation between fresh servers:",
                "",
                "| Repeat | Streams | TPS, baseline → candidate | TPS change | TTFT ms, baseline → candidate | TTFT change |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for base_case, candidate_case in zip(baseline, candidate, strict=True):
            assert base_case["repeat"] == candidate_case["repeat"]
            pair = [
                helper.load_means(
                    study.case_folder(case) / "agent-multiturn-benchmarks.json"
                )
                for case in (base_case, candidate_case)
            ]
            per_repetition.append({"repeat": base_case["repeat"], "means": pair})
            for streams in (1, 2, 4, 8):
                a, b = (m[streams] for m in pair)
                tps_change = 100 * (b["output_tokens_per_second"] / a["output_tokens_per_second"] - 1)
                ttft_change = 100 * (b["time_to_first_token_ms"] / a["time_to_first_token_ms"] - 1)
                tps_label = f"{tps_change:+.1f}%"
                ttft_label = f"{ttft_change:+.1f}%"
                if tps_change > 0:
                    tps_label = f"**{tps_label}**"
                if ttft_change < 0:
                    ttft_label = f"**{ttft_label}**"
                lines.append(
                    f"| {base_case['repeat']} | {streams} | {a['output_tokens_per_second']:.2f} → {b['output_tokens_per_second']:.2f} | {tps_label} | {a['time_to_first_token_ms']:,.0f} → {b['time_to_first_token_ms']:,.0f} | {ttft_label} |"
                )
        if model.startswith("qwen"):
            lines.extend(
                [
                    "",
                    "All four Qwen runs logged Triton JIT-monitor warnings during measured requests, including both repetitions. Every request remains in the pooled summary, following the skill's all-successful-request rule. Compiler caches persist, but their presence does not establish a fully warmed run. Absolute speed changes between repetitions are substantial; the second repetition is near parity between segmented versions. These small differences do not establish a repeatable optimization gain.",
                ]
            )
    if baseline[0]["workload"] == "short" and manifest.get("short_primer"):
        for case in baseline + candidate:
            study.validate_report(
                study.case_folder(case) / "primer-multiturn-benchmarks.json", 128
            )
        lines.extend(
            [
                "",
                "Every short condition runs a separate, untimed copy of the complete short workload before measurement. Prefix cache is reset afterward and counters are sampled after a 12-second flush interval. The measured GuideLLM report still includes all of its successful warmup/cooldown requests. Primer reports are retained separately.",
            ]
        )
    if any(case["mtp"] for case in baseline + candidate):
        lines.extend(
            [
                "",
                f"MTP cache-reader support and checks: [validation]({study.RUN / 'shared-kv-fix/validation.json'}), "
                f"[kernel/dispatch equivalence]({study.RUN / 'shared-kv-fix/math-equivalence.json'}), "
                f"[tuning-cache reuse provenance]({study.RUN / 'shared-kv-fix/cache-migration.json'}). "
                "The failed initial MTP startup and the original source snapshots are retained; no measured requests from that failure enter this comparison.",
            ]
        )
    lines.extend(
        [
            "",
            "Achieved mean concurrency for each repetition:",
            "",
            "| Arm | Repeat | Streams | Mean active requests |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, runs in zip(labels, counts, strict=True):
        for repeat, rows in enumerate(runs, start=1):
            for row in rows:
                lines.append(
                    f"| {label} | {repeat} | {row['streams']} | {row['mean_concurrency']:.3f} |"
                )
    if any(
        row["mean_concurrency"] < row["streams"] * 0.98
        for runs in counts
        for rows in runs
        for row in rows
    ):
        lines.extend(
            [
                "",
                "Some runs sustained less than 98% of their configured concurrency. The achieved-concurrency table shows this limitation; small performance differences need additional repetitions.",
            ]
        )
    if any(c["mtp"] for c in baseline + candidate):
        lines.extend(
            [
                "",
                "MTP counters are measured by subtracting Prometheus snapshots immediately before and after the measured workload; the initial smoke and any separate primer are excluded. Speculation uses one token and the matching pinned Google assistant checkpoint with its published BF16 weights.",
                "",
                "| Arm | Draft tokens | Accepted | Acceptance |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, cases in zip(labels, (baseline, candidate), strict=True):
            for case in cases:
                if case["mtp"]:
                    value = acceptance(case)
                    rate = (
                        f"{100 * value['acceptance']:.2f}%"
                        if value["acceptance"] is not None
                        else "unavailable"
                    )
                    lines.append(
                        f"| {label} | {value['proposed']:.0f} | {value['accepted']:.0f} | {rate} |"
                    )
        lines.extend(
            [
                "",
                "This short synthetic MTP run measures serving behavior under fixed output lengths. It does not establish acceptance or speedups on natural conversation data. The matching short non-MTP controls provide a same-workload speculation comparison.",
            ]
        )
    lines.extend(
        [
            "",
            "Startup and tuning are excluded from serving latency. The main comparison has two server repetitions; short exploratory controls have one. Pooled request means are descriptive and do not establish statistical significance. Prefix cache is reset before each complete GuideLLM sweep; the original workload retains cache state between its stream levels.",
            "Compiler caches persist across repetitions. If present, `logs/segmented-serving-2026-10-16/cache-reuse.json` records the reuse of matching model/backend compiler caches and target-attention tuning tables for MTP startup; existing identity and workload guards still validate cache entries.",
            "",
        ]
    )
    for kind, label in (
        ("tps", "Output TPS"),
        ("ttft", "TTFT"),
        ("e2e", "End-to-end latency"),
    ):
        chart = plot_dir / f"compare_{kind}_by_turn.png"
        assert chart.is_file(), chart
        lines.extend([f"![{label}]({relative(chart, directory)})", ""])
    lines.append("Raw reports:")
    lines.append("")
    for label, cases in zip(labels, (baseline, candidate), strict=True):
        for case in cases:
            folder = study.case_folder(case)
            lines.append(
                f"- [{label}, repetition {case['repeat']}]({relative(folder / 'agent-multiturn-benchmarks.json', directory)}), [server setup]({relative(folder / 'setup.json', directory)}), [worker inspection]({relative(folder / 'inspection.json', directory)})"
            )
            if (folder / "primer-multiturn-benchmarks.json").exists():
                lines.append(
                    f"  [Separate untimed primer]({relative(folder / 'primer-multiturn-benchmarks.json', directory)})."
                )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    value = {
        "model": model,
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "means": means,
        "per_repetition": per_repetition,
        "changes_percent": changes,
        "counts": counts,
        "backend_counts": backend_counts,
        "report": str(directory / "comparison.md"),
        "plot_dir": str(plot_dir),
        "table_markdown": table,
    }
    study.write(directory / "comparison.json", value)
    return value


def main():
    manifest = study.read(study.RUN / "manifest.json")
    reports = []
    for model in study.SLUGS:
        full = [
            c
            for c in manifest["plan"]
            if c["model"] == model and c["workload"] == "full"
        ]
        before = "segmented-before" if model.startswith("qwen") else "default"
        reports.append(
            compare(
                manifest,
                model,
                "full-workload",
                [c for c in full if c["arm"] == before],
                [c for c in full if c["arm"] == "segmented-after"],
                (
                    "Before segmented"
                    if model.startswith("qwen")
                    else "Automatic backend",
                    "Updated segmented",
                ),
            )
        )
        if model.startswith("qwen"):
            continue
        short = [
            c
            for c in manifest["plan"]
            if c["model"] == model and c["workload"] == "short"
        ]
        reports.append(
            compare(
                manifest,
                model,
                "short-mtp-backends",
                [c for c in short if c["arm"] == "default" and c["mtp"]],
                [c for c in short if c["arm"] == "segmented-after" and c["mtp"]],
                ("Automatic + MTP", "Segmented + MTP"),
            )
        )
        for arm in ("default", "segmented-after"):
            reports.append(
                compare(
                    manifest,
                    model,
                    f"short-mtp-effect-{arm}",
                    [c for c in short if c["arm"] == arm and not c["mtp"]],
                    [c for c in short if c["arm"] == arm and c["mtp"]],
                    ("Ordinary decode", "MTP1"),
                )
            )
    study.write(study.RUN / "comparison-index.json", reports)
    lines = [
        "# TP2 segmented attention serving study",
        "",
        "All cases use FP8 weights, BF16 activations and FP8 KV cache on two R9700 GPUs. Model revisions, sources, workloads and commands are pinned in the manifest. Reports use the vllm-bench-collection comparison skill and all successful request records.",
        "",
        "Both Gemma models automatically selected TRITON_ATTN; their candidate explicitly selects ROCM_SEGMENTED_ATTN. Qwen compares the previous and updated segmented implementations. The main workload has two fresh-server repetitions per arm; each short condition has one repetition after a separate matching untimed primer.",
        "",
        "All four Qwen runs logged Triton JIT-monitor warnings during measured requests. Pooled means retain those requests; the per-repetition tables show large absolute timing variation and near parity in the second repetition. Persistent compiler caches do not establish a fully warmed run. These descriptive serving results do not isolate attention kernel latency or establish statistical significance.",
        "",
    ]
    for report in reports:
        if report["name"] == "full-workload":
            lines.extend(
                [
                    f"## {manifest['models'][report['model']]['model']}",
                    "",
                    report["table_markdown"].rstrip(),
                    "",
                ]
            )
    lines.extend(
        [
            "## Short MTP controls",
            "",
            "One speculative token with the matching BF16 Google assistant; 128 output tokens per turn; one fresh-server repetition per condition, after a separate untimed matching primer and a prefix-cache reset. Positive TPS changes improve performance. These synthetic fixed-length results are exploratory.",
            "",
            "| Model | Streams | MTP TPS gain with default | MTP TPS gain with segmented | Segmented/default TPS with MTP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("gemma12-fp8", "gemma26-fp8"):
        by_name = {r["name"]: r for r in reports if r["model"] == model}
        for streams in (1, 2, 4, 8):
            changes = [
                by_name[name]["changes_percent"][streams]["output_tokens_per_second"]
                for name in (
                    "short-mtp-effect-default",
                    "short-mtp-effect-segmented-after",
                    "short-mtp-backends",
                )
            ]
            formatted = [
                f"**{value:+.1f}%**" if value > 0 else f"{value:+.1f}%"
                for value in changes
            ]
            lines.append(
                f"| {model} | {streams} | {formatted[0]} | {formatted[1]} | {formatted[2]} |"
            )
    lines.extend(["", "## Full comparisons and plots", ""])
    for report in reports:
        lines.append(
            f"- [{manifest['models'][report['model']]['model']}: {report['name']}]({report['report']})"
        )
    (study.RUN / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
