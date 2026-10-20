"""Audit retained serving inputs, outputs, links and measured JIT warnings."""

import re
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import report_segmented_serving as reporting
import run_segmented_serving as study


def main():
    manifest = study.read(study.RUN / "manifest.json")
    study.frozen(manifest)
    entries = []
    total_requests = 0
    for case in manifest["plan"]:
        folder = study.case_folder(case)
        success = study.read(folder / "success.json")
        report_path = folder / "agent-multiturn-benchmarks.json"
        assert study.sha(report_path) == success["report_sha256"]
        counts = study.validate_report(report_path, 128 if case["workload"] == "short" else 512)
        total_requests += sum(row["successful"] for row in counts)
        setup = study.read(folder / "setup.json")
        command = setup["command"]
        assert command[command.index("--tensor-parallel-size") + 1] == "2"
        assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
        assert command[command.index("--dtype") + 1] == "bfloat16"
        version = "before" if case["arm"] == "segmented-before" else "after-mtp" if case["mtp"] else "after"
        assert setup["source"] == manifest["sources"][version]["root"]
        inspection = study.read(folder / "inspection.json")["results"]
        assert len(inspection) == 2
        for rank in inspection:
            assert rank["tp"] == 2 and rank["kv_cache_dtype"] == "fp8"
            for key, rel in (("ops", study.REL_OPS), ("tuner", study.REL_TUNER)):
                assert rank["segmented_sources"][key]["sha256"] == manifest["sources"][version]["sha256"][rel]
        if case["workload"] == "short":
            primer = folder / "primer-multiturn-benchmarks.json"
            assert study.sha(primer) == success["primer_report_sha256"]
            study.validate_report(primer, 128)
        acceptance = reporting.acceptance(case) if case["mtp"] else None
        if acceptance:
            assert acceptance["proposed"] > 0 and 0 <= acceptance["accepted"] <= acceptance["proposed"]
        data = study.read(report_path)
        records = [r for b in data["benchmarks"] for r in b["requests"]["successful"]]
        first = min(r["request_start_time"] for r in records)
        last = max(r["request_end_time"] for r in records)
        measured_jit = []
        all_jit = []
        for line in (folder / "server.log").read_text().splitlines():
            if "WARNING" not in line or "[jit_monitor.py:" not in line:
                continue
            all_jit.append(line)
            match = re.search(r"WARNING (\d\d)-(\d\d) (\d\d:\d\d:\d\d)", line)
            if match:
                stamp = datetime.strptime(f"2026-{match[1]}-{match[2]} {match[3]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Kuala_Lumpur")).timestamp()
                if first - 1 <= stamp <= last + 1:
                    measured_jit.append(line)
        entries.append({"case": case, "counts": counts, "source_version": version, "report_sha256": study.sha(report_path), "mtp": acceptance, "observed_jit_warnings": len(all_jit), "observed_measured_jit_warnings": measured_jit})
    assert len(entries) == 20
    reports = study.read(study.RUN / "comparison-index.json")
    assert len(reports) == 9
    checked_links = 0
    visual = study.read(study.RUN / "visual-inspection.json")["comparisons"]
    for item in reports:
        path = Path(item["report"])
        assert path.is_file()
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("https://", "http://")):
                continue
            assert (path.parent / target).is_file(), (path, target)
            checked_links += 1
        key = f"{item['model']}/{item['name']}"
        seen = visual[key]
        assert len(seen["sha256"]) >= 2
        for rel, digest in seen["sha256"].items():
            assert study.sha(study.ROOT / rel) == digest, rel
    value = {"completed": datetime.now(UTC).isoformat(), "cases": len(entries), "measured_successful_requests": total_requests, "measured_errors": 0, "measured_incomplete": 0, "comparisons": len(reports), "checked_local_links": checked_links, "charts_visually_inspected": True, "jit_warning_caveat": "JIT monitor warning_once suppresses repeated kernel warnings. Counts are observed lower bounds; zero warnings does not prove zero compilation.", "entries": entries}
    study.write(study.RUN / "final-audit.json", value)
    print({k: v for k, v in value.items() if k != "entries"})


if __name__ == "__main__":
    main()
