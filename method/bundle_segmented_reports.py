"""Collect the completed study into one review directory with local links."""

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import report_segmented_serving as reporting
import run_segmented_serving as study

DEST = study.ROOT / "reports/segmented-attention-2026-10-16"
SEARCH = study.ROOT / "logs/segmented-attn-gemma4/compressed-search-2026-10-16"
LINK = re.compile(r"(!?\[[^\]]*\]\()([^)]+)(\))")


def gain_label(before, after):
    change = 100 * (after / before - 1)
    if abs(change) < 0.05:
        return "~0%"
    label = f"{change:+.1f}%"
    return f"**{label}**" if change > 0 else label


def main():
    manifest = study.read(study.RUN / "manifest.json")
    audit = study.read(study.RUN / "final-audit.json")
    assert audit["cases"] == 20 and audit["comparisons"] == 9 and audit["charts_visually_inspected"]
    assert audit["measured_errors"] == audit["measured_incomplete"] == 0
    reports = study.read(study.RUN / "comparison-index.json")
    assert not DEST.exists(), f"Preserve the existing bundle: {DEST}"
    DEST.mkdir(parents=True)
    mapped = {}
    markdown = []

    def copy(source, target):
        source, target = Path(source).resolve(), Path(target)
        assert source.is_file(), source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mapped[source] = target
        if source.suffix == ".md":
            markdown.append((source, target))

    for case in manifest["plan"]:
        folder = study.case_folder(case)
        assert (folder / "success.json").is_file()
        target = DEST / "raw" / study.SLUGS[case["model"]] / folder.name
        for file in folder.iterdir():
            if file.is_file() and file.suffix in (".json", ".txt", ".log"):
                copy(file, target / file.name)

    for item in reports:
        source = Path(item["report"])
        target = DEST / "serving" / study.SLUGS[item["model"]] / item["name"]
        for name in ("comparison.md", "comparison.json", "baseline-requests.json", "candidate-requests.json"):
            copy(source.parent / name, target / name)
        for chart in Path(item["plot_dir"]).glob("*.png"):
            copy(chart, target / "plots" / chart.name)

    for name in ("manifest.json", "models.json", "client-environment.json", "git-checkpoint.json", "cache-reuse.json", "short-primer-activated.json", "comparison-index.json", "mm-prefix-followup.json", "plan.json"):
        copy(study.RUN / name, DEST / "provenance" / name)
    for name in ("final-audit.json", "visual-inspection.json"):
        copy(study.RUN / name, DEST / "validation" / name)
    for file in (study.RUN / "shared-kv-fix").iterdir():
        if file.is_file():
            copy(file, DEST / "validation/shared-kv-fix" / file.name)
    copy(SEARCH / "report.md", DEST / "autotune/report.md")
    for match in LINK.finditer((SEARCH / "report.md").read_text()):
        target = (SEARCH / match[2]).resolve()
        if target.is_relative_to(SEARCH) and target.is_file():
            copy(target, DEST / "autotune" / target.relative_to(SEARCH))
    copy(SEARCH / "search-comparison.pdf", DEST / "autotune/search-comparison.pdf")
    copy(reporting.SKILL / "SKILL.md", DEST / "method/SKILL.md")
    for name in ("plot_compare_multiturn.py", "summarize_guidellm.py"):
        copy(reporting.SKILL / "scripts" / name, DEST / "method/scripts" / name)
    for name in ("run_segmented_serving.py", "segmented_serving_probe.py", "report_segmented_serving.py", "audit_segmented_serving.py", "bundle_segmented_reports.py"):
        copy(study.ROOT / "scripts" / name, DEST / "method" / name)

    for source, target in markdown:
        def replace(match):
            link = match[2]
            if link.startswith(("https://", "http://", "#")):
                return match[0]
            original = (source.parent / link).resolve()
            assert original in mapped, (source, link)
            return match[1] + os.path.relpath(mapped[original], target.parent) + match[3]
        target.write_text(LINK.sub(replace, source.read_text()))

    checkpoint = study.read(study.RUN / "git-checkpoint.json")
    lines = [
        "# Segmented attention: benchmark review",
        "",
        "Completed: 20 TP2 serving conditions, 2,400 successful measured turn requests, no errors or incomplete requests, and nine two-case comparisons. Separate short primers contribute another 960 successful requests and are excluded from measured means and MTP counter deltas.",
        "",
        "Hardware: two Radeon AI PRO R9700 (gfx1201), 32 GiB each. Serving: FP8 checkpoint weights, BF16 activations, explicit FP8 KV cache in every case; context 32,768, max sequences 8, batched tokens 8,192, prefix caching enabled, seed 20260715. These results do not include separate BF16-weight checkpoints.",
        "",
        "## Main simple-agent workload",
        "",
        "Eight turns, 2,048 new synthetic prompt tokens and 512 output tokens per turn. Two fresh-server repetitions per arm. Gemma compares automatic TRITON_ATTN against opt-in ROCM_SEGMENTED_ATTN; Qwen compares the previous and updated segmented implementations. Values are mean per-request output tokens/s over all successful records, including warmup/cooldown. They are end-to-end serving measurements.",
        "",
        "| Model | 1 stream | 2 streams | 4 streams | 8 streams |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in study.SLUGS:
        item = next(r for r in reports if r["model"] == model and r["name"] == "full-workload")
        cells = []
        for streams in (1, 2, 4, 8):
            a, b = (m[str(streams)]["output_tokens_per_second"] for m in item["means"])
            cells.append(f"{a:.2f} → {b:.2f} ({gain_label(a, b)})")
        lines.append(f"| {manifest['models'][model]['model']} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "All four Qwen runs logged Triton JIT-monitor warnings during measurement, including both repetitions. The per-repetition tables retain every request and show substantial absolute timing variation; the second repetition is near parity. Persistent compiler caches do not establish fully warmed execution. Small pooled Qwen differences do not establish a repeatable speedup. Gemma's larger improvements also need the per-repetition context in the detailed reports.",
        "",
        "## Short Gemma MTP controls",
        "",
        "Eight turns and 128 output tokens per turn; matching Google BF16 assistant; one speculative token. Each condition uses a fresh server, a complete separate untimed matching primer, a prefix-cache reset and counter flush before measurement. One repetition per condition makes these exploratory synthetic results.",
        "",
        "| Model | Streams | Default TPS: plain → MTP | Segmented TPS: plain → MTP | Segmented/default TPS with MTP |",
        "|---|---:|---:|---:|---:|",
    ])
    acceptance_lines = []
    for model in ("gemma12-fp8", "gemma26-fp8"):
        by_name = {r["name"]: r for r in reports if r["model"] == model}
        for streams in (1, 2, 4, 8):
            cells = []
            for arm in ("default", "segmented-after"):
                item = by_name[f"short-mtp-effect-{arm}"]
                a, b = (m[str(streams)]["output_tokens_per_second"] for m in item["means"])
                cells.append(f"{a:.2f} → {b:.2f} ({gain_label(a, b)})")
            item = by_name["short-mtp-backends"]
            a, b = (m[str(streams)]["output_tokens_per_second"] for m in item["means"])
            ratio = f"{b / a:.2f}×"
            ratio = f"**{ratio}**" if b > a else ratio
            lines.append(f"| {model} | {streams} | {cells[0]} | {cells[1]} | {ratio} |")
        cases = [c for c in manifest["plan"] if c["model"] == model and c["mtp"]]
        rates = [f"{c['arm']}: {100 * reporting.acceptance(c)['acceptance']:.2f}%" for c in cases]
        acceptance_lines.append(f"Measured aggregate draft acceptance for {model}: " + "; ".join(rates) + ".")
    lines.extend(["", *acceptance_lines, ""])
    default_effect = next(r for r in reports if r["model"] == "gemma26-fp8" and r["name"] == "short-mtp-effect-default")
    plain, mtp = (m["8"] for m in default_effect["means"])
    lines.extend([
        f"At eight streams, Gemma 26B default MTP is near TPS parity "
        f"({gain_label(plain['output_tokens_per_second'], mtp['output_tokens_per_second'])}), "
        f"while mean E2E rises {plain['request_latency']:.2f} → {mtp['request_latency']:.2f}s "
        f"({100 * (mtp['request_latency'] / plain['request_latency'] - 1):+.1f}%). "
        "Mean per-request rates and mean latencies average different quantities; the per-turn plots show the variation. "
        "The detailed comparisons retain both metrics.",
        "",
    ])
    lines.extend([
        "",
        "MTP startup exposed attention layers that read the target KV cache with no projected K/V. The segmented host path now supports those readers. The fix passed 20 GPU tests, 319 CPU contracts and repository hooks; all three compiled Triton cache keys and 6,144 existing projected-K/V dispatch checks match. The MTP source snapshot is separate from the preserved main-study snapshots. [Validation](validation/shared-kv-fix/validation.json) and [cache reuse provenance](validation/shared-kv-fix/cache-migration.json).",
        "",
        "## Compressed FP8 autotuning",
        "",
        "The comprehensive kernel study passed 1,704/1,704 numerical cases. Eligible FP8 candidate count fell 43.0%, and summed search time fell 55.9% (18,498.51 → 8,150.61 seconds across unique profiles with compiler caches prewarmed). This is not one model's startup time. BF16/normal-only search is unchanged.",
        "",
        "Aggregate compressed/full tuned performance is near parity, but 98 per-shape losses reproduced in independent fresh-input retests. The compressed search is not regression-free. [Full autotune report, tail results and plots](autotune/report.md).",
        "",
        "## Detailed comparisons",
        "",
    ])
    for item in reports:
        path = mapped[Path(item["report"]).resolve()]
        lines.append(f"- [{manifest['models'][item['model']]['model']}: {item['name']}]({path.relative_to(DEST)})")
    lines.extend([
        "",
        "Each comparison includes TPS, TTFT and E2E charts, request-level means, workload checks, achieved concurrency and raw-result links. Positive TPS changes and negative TTFT/E2E changes improve performance. All 18 required TPS/TTFT charts were visually inspected; all 27 serving plots are included.",
        "",
        "## Provenance and next work",
        "",
        f"Verified pushed checkpoint: `{checkpoint['local_head']}` on `{checkpoint['branch']}`. The benchmark snapshots and exact file hashes are preserved separately in [the manifest](provenance/manifest.json). [Final audit](validation/final-audit.json), [visual inspection](validation/visual-inspection.json), [comparison skill](method/SKILL.md).",
        "",
        f"Next scope: Gemma 4 multimodal prefix attention support on `{checkpoint['next_branch']}`, based on the pushed checkpoint. These serving runs use language-model-only mode; multimodal support is separate follow-up work. [Masking observations and validation plan](provenance/mm-prefix-followup.json).",
        "",
        "The raw serving JSONs, primers, server setup/inspection, logs, derived request pools, chart helpers and validation records are included in this directory. Historical runtime paths inside JSON records describe the original workspace. Model weights, compiler caches and source worktrees remain in that workspace.",
    ])
    (DEST / "README.md").write_text("\n".join(lines) + "\n")
    checked = 0
    for page in DEST.rglob("*.md"):
        for match in LINK.finditer(page.read_text()):
            if match[2].startswith(("https://", "http://", "#")):
                continue
            target = (page.parent / match[2]).resolve()
            assert target.is_relative_to(DEST) and target.is_file(), (page, match[2])
            checked += 1
    files = {str(p.relative_to(DEST)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in DEST.rglob("*") if p.is_file()}
    study.write(DEST / "bundle-manifest.json", {"created": datetime.now(UTC).isoformat(), "source_run": str(study.RUN), "checked_links": checked, "files": files})
    print(json.dumps({"directory": str(DEST), "files": len(files), "checked_links": checked, "bytes": sum(v["bytes"] for v in files.values())}))


if __name__ == "__main__":
    main()
