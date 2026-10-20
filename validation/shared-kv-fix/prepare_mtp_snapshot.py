"""Record and activate a separate, validated cache-reader MTP snapshot."""

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "logs/segmented-serving-2026-10-16"
FIX = RUN / "shared-kv-fix"
SOURCE = ROOT / "vllm-segmented-attn-gemma"
DRIVER = ROOT / "scripts/run_segmented_serving.py"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


spec = importlib.util.spec_from_file_location("serving_driver", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
manifest = read(RUN / "manifest.json")
driver.frozen(manifest)
assert "after-mtp" not in manifest["sources"]
proof = read(FIX / "math-equivalence.json")
assert proof["projected_kv_dispatch"]["identical"] == proof["projected_kv_dispatch"]["cases"]
assert all(v["ast_equal"] and v["line_equal"] and v["triton_cache_key"] for v in proof["jit_functions"].values())
assert sha(SOURCE / driver.REL_OPS) == proof["after_ops_sha256"]
old_source = Path(manifest["sources"]["after"]["root"])
assert sha(old_source / driver.REL_OPS) == proof["before_ops_sha256"]
for rel in (driver.REL_CONFIG, driver.REL_TUNER):
    assert sha(SOURCE / rel) == sha(old_source / rel)

assert "319 passed" in (FIX / "cpu-contracts-final.log").read_text()
assert "Failed" not in (FIX / "precommit-final.log").read_text()
assert "Run mypy" in (FIX / "precommit-final.log").read_text()
for rank in (0, 1):
    assert "10 passed" in (FIX / f"gpu{rank}-pytest.log").read_text()
validation = {
    "source_sha256": {rel: sha(SOURCE / rel) for rel in (driver.REL_OPS, driver.REL_BACKEND, driver.REL_CONFIG, driver.REL_TUNER, "tests/kernels/attention/test_segmented_attention.py")},
    "cpu_passed": 319,
    "gpu_passed": 20,
    "precommit": "passed (including mypy)",
    "equivalence": proof,
    "change": "Allow paired None K/V for read-only attention against existing KV cache; preserve kernel arithmetic and existing projected-K/V dispatch.",
}
write(FIX / "validation.json", validation)

amendment = RUN / "protocol-amendments/mtp-cache-only-reader"
assert not amendment.exists()
amendment.mkdir(parents=True)
shutil.copy2(DRIVER, amendment / DRIVER.name)
shutil.copy2(RUN / "manifest.json", amendment / "manifest-before.json")
prior_results = {}
for case in manifest["plan"]:
    folder = driver.case_folder(case)
    if (folder / "success.json").exists():
        success = read(folder / "success.json")
        report = folder / "agent-multiturn-benchmarks.json"
        assert sha(report) == success["report_sha256"]
        prior_results[str(folder)] = success["report_sha256"]
assert len(prior_results) == 14, len(prior_results)

snapshot = RUN / "source-after-mtp"
assert not snapshot.exists()
subprocess.run(["git", "-C", str(SOURCE), "worktree", "add", "--detach", str(snapshot), manifest["head"]], check=True)
for rel in (driver.REL_OPS, driver.REL_BACKEND, driver.REL_CONFIG, driver.REL_TUNER, "vllm/_version.py"):
    shutil.copy2(old_source / rel, snapshot / rel)
for rel in (driver.REL_OPS, driver.REL_BACKEND):
    shutil.copy2(SOURCE / rel, snapshot / rel)
for binary in (SOURCE / "vllm").glob("*.so"):
    (snapshot / "vllm" / binary.name).symlink_to(binary.resolve())
source_hashes = {
    str(p.relative_to(snapshot)): sha(p)
    for p in (snapshot / "vllm").rglob("*")
    if p.is_file() and p.suffix in (".py", ".json", ".so")
}
old_hashes = manifest["sources"]["after"]["sha256"]
assert source_hashes.keys() == old_hashes.keys()
changed = [rel for rel in source_hashes if source_hashes[rel] != old_hashes[rel]]
assert set(changed) == {driver.REL_OPS, driver.REL_BACKEND}, changed
manifest["sources"]["after-mtp"] = {"root": str(snapshot), "sha256": source_hashes}

from vllm.model_executor.warmup import rocm_segmented_attn_autotune_warmup as tuner

migrated = []
for model in ("gemma12-fp8", "gemma26-fp8"):
    cache = ROOT / ".cache/segmented-serving-20261016" / model / "segmented-after/mtp1/vllm/rocm_segmented_attention"
    paths = list(cache.glob("*.json"))
    assert len(paths) == 2, paths
    for old_path in paths:
        saved = read(old_path)
        identity = saved["identity"]
        assert identity["kernel_sha256"] == proof["before_ops_sha256"]
        assert identity["selector_sha256"] == sha(SOURCE / driver.REL_CONFIG)
        assert identity["tuner_sha256"] == sha(SOURCE / driver.REL_TUNER)
        updated = copy.deepcopy(saved)
        updated["identity"]["kernel_sha256"] = proof["after_ops_sha256"]
        digest = hashlib.sha256(json.dumps(updated["identity"], sort_keys=True).encode()).hexdigest()
        new_path = old_path.with_name(digest + ".json")
        assert not new_path.exists()
        write(new_path, updated)
        loaded, shards = tuner._load_records(new_path, updated["identity"], identity["heads"], identity["kv_heads"], identity["dim"], True)
        assert not shards
        assert loaded["records"] == saved["records"]
        assert len(loaded["records"]) == 176
        migrated.append({"old_path": str(old_path), "new_path": str(new_path), "old_sha256": sha(old_path), "new_sha256": sha(new_path), "records": len(loaded["records"]), "only_identity_change": "kernel_sha256", "original_preserved": True})
write(FIX / "cache-migration.json", {"proof": str(FIX / "math-equivalence.json"), "entries": migrated})

failed = driver.case_folder({"model": "gemma12-fp8", "arm": "segmented-after", "workload": "short", "mtp": True, "repeat": 1})
assert failed.exists() and not (failed / "success.json").exists()
assert not (failed / "agent-multiturn-benchmarks.json").exists()
failure_archive = RUN / "preflight-failures/mtp-cache-only-reader"
assert not failure_archive.exists()
failure_archive.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(failed), failure_archive)

old_line = '    version = "before" if case["arm"] == "segmented-before" else "after"\n'
new_line = '''    version = (
        "before"
        if case["arm"] == "segmented-before"
        else "after-mtp"
        if case["mtp"]
        else "after"
    )
'''
text = DRIVER.read_text()
assert text.count(old_line) == 1
DRIVER.write_text(text.replace(old_line, new_line))
manifest["runner_sha256"] = sha(DRIVER)
manifest["mtp_cache_only_reader"] = {
    "activated": datetime.now(UTC).isoformat(),
    "source_version": "after-mtp",
    "validation": str(FIX / "validation.json"),
    "cache_migration": str(FIX / "cache-migration.json"),
    "reason": "Gemma assistant attention reads target KV with None projected K/V. Original segmented backend sliced None during graph capture; host-only reader support is required.",
    "prior_successes": prior_results,
    "changed_files": changed,
}
write(RUN / "manifest.json", manifest)
write(amendment / "amendment.json", manifest["mtp_cache_only_reader"])
write(amendment / "manifest-after.json", manifest)
print(json.dumps({"snapshot": str(snapshot), "changed_files": changed, "migrated_tables": len(migrated), "preserved_successes": len(prior_results), "runner_sha256": manifest["runner_sha256"]}, indent=2), flush=True)
