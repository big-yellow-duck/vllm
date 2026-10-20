"""Matched TP2 simple-agent serving comparisons for segmented attention."""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/segmented-serving-2026-10-16"
SOURCE = ROOT / "vllm-segmented-attn-gemma"
BENCH = ROOT / "vllm-bench-collection"
CLIENT_PY = (
    ROOT.parent
    / "gfx1201-archive/vllm-sources-20261013/vllm-bench-collection/.venv/bin/python"
)
BASELINE = ROOT / "logs/segmented-attn-gemma4/selector-integration-2026-10-15/baseline"
PORT = 8026
REL_OPS = "vllm/v1/attention/ops/segmented_attention.py"
REL_CONFIG = "vllm/v1/attention/ops/segmented_attention_config.py"
REL_TUNER = "vllm/model_executor/warmup/rocm_segmented_attn_autotune_warmup.py"
REL_BACKEND = "vllm/v1/attention/backends/rocm_segmented_attn.py"
SLUGS = {
    "qwen27-fp8": "qwen3.8-27b-fp8",
    "gemma12-fp8": "gemma-4-12b-fp8-dynamic",
    "gemma26-fp8": "gemma-4-26b-a4b-fp8-dynamic",
}
ACTIVE = []


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def status(phase, **details):
    value = dict(phase=phase, updated=datetime.now(UTC).isoformat(), **details)
    write(RUN / "status.json", value)
    print(json.dumps(value), flush=True)


def git(*args, repo=SOURCE):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare(repeats, autotune):
    path = RUN / "manifest.json"
    if path.exists():
        manifest = read(path)
        assert manifest["repeats"] == repeats and manifest["autotune"] == autotune
        frozen(manifest)
        return manifest
    models = read(RUN / "models.json")
    assert set(SLUGS) <= models.keys()
    assert {"gemma12-assistant", "gemma26-assistant"} <= models.keys()
    head = git("rev-parse", "HEAD")
    sources = {}
    for version in ("before", "after"):
        snapshot = RUN / f"source-{version}"
        if not snapshot.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(SOURCE),
                    "worktree",
                    "add",
                    "--detach",
                    str(snapshot),
                    head,
                ],
                check=True,
            )
        assert git("rev-parse", "HEAD", repo=snapshot) == head
        for rel in (REL_OPS, REL_TUNER, REL_BACKEND):
            origin = BASELINE / Path(rel).name if version == "before" else SOURCE / rel
            shutil.copy2(origin, snapshot / rel)
        if version == "after":
            assert (SOURCE / REL_CONFIG).is_file(), (
                "Apply and validate the typed selector first"
            )
            shutil.copy2(SOURCE / REL_CONFIG, snapshot / REL_CONFIG)
        for binary in (SOURCE / "vllm").glob("*.so"):
            dest = snapshot / "vllm" / binary.name
            if not dest.exists():
                dest.symlink_to(binary.resolve())
        shutil.copy2(SOURCE / "vllm/_version.py", snapshot / "vllm/_version.py")
        files = [
            p
            for p in (snapshot / "vllm").rglob("*")
            if p.is_file() and p.suffix in (".py", ".json", ".so")
        ]
        sources[version] = {
            "root": str(snapshot),
            "sha256": {str(p.relative_to(snapshot)): sha(p) for p in files},
        }
    plan = []
    for repeat in range(1, repeats + 1):
        for model in SLUGS:
            arms = (
                ["segmented-before", "segmented-after"]
                if model.startswith("qwen")
                else ["default", "segmented-after"]
            )
            if (repeat + list(SLUGS).index(model)) % 2 == 0:
                arms.reverse()
            plan.extend(
                {
                    "model": model,
                    "arm": arm,
                    "repeat": repeat,
                    "workload": "full",
                    "mtp": False,
                }
                for arm in arms
            )
    for model in ("gemma12-fp8", "gemma26-fp8"):
        for mtp in (False, True):
            arms = ["default", "segmented-after"]
            if mtp:
                arms.reverse()
            plan.extend(
                {
                    "model": model,
                    "arm": arm,
                    "repeat": 1,
                    "workload": "short",
                    "mtp": mtp,
                }
                for arm in arms
            )
    manifest = {
        "head": head,
        "repeats": repeats,
        "autotune": autotune,
        "short_primer": True,
        "models": models,
        "sources": sources,
        "workload": read(
            BENCH / "benchmarks-configs/simple-agent-multiturn-prefix-cache.json"
        ),
        "plan": plan,
        "runner_sha256": sha(__file__),
        "inspector_sha256": sha(ROOT / "scripts/segmented_serving_probe.py"),
        "bench_head": git("rev-parse", "HEAD", repo=BENCH),
        "client_environment": read(RUN / "client-environment.json"),
        "runtime": {
            str(rank): read(
                ROOT
                / f"logs/segmented-attn-gemma4/compressed-search-2026-10-16/post-types/gpu{rank}/results.metadata.json"
            )
            for rank in (0, 1)
        },
        "method": "Fresh TP2 servers, alternating arm order, pinned model snapshots, FP8 weights/BF16 activations/FP8 KV. Main workload unchanged; short MTP controls reduce output to128 tokens. All segmented arms use the same autotune setting.",
    }
    write(path, manifest)
    return manifest


def frozen(manifest):
    assert sha(__file__) == manifest["runner_sha256"]
    assert (
        sha(ROOT / "scripts/segmented_serving_probe.py") == manifest["inspector_sha256"]
    )
    for data in manifest["sources"].values():
        for rel, value in data["sha256"].items():
            assert sha(Path(data["root"]) / rel) == value, rel


def request(path, data=None, timeout=30):
    req = Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    if proc in ACTIVE:
        ACTIVE.remove(proc)


def interrupted(signum, _frame):
    for process in ACTIVE[:]:
        stop(process)
    raise SystemExit(128 + signum)


def validate_report(path, output_tokens, streams=(1, 2, 4, 8)):
    data = read(path)
    assert len(data["benchmarks"]) == len(streams)
    counts = []
    for bench in data["benchmarks"]:
        n = int(bench["config"]["strategy"]["streams"])
        reqs = bench["requests"]
        assert n in streams and not reqs.get("errored") and not reqs.get("incomplete")
        records = reqs["successful"]
        assert len(records) == n * 8, (n, len(records))
        turns = defaultdict(list)
        for row in records:
            assert row["output_tokens"] == output_tokens
            assert all(
                math.isfinite(row[k]) and row[k] > 0
                for k in (
                    "output_tokens_per_second",
                    "time_to_first_token_ms",
                    "request_latency",
                )
            )
            turns[row["info"]["conversation_id"]].append(row["info"]["turn_index"])
        assert len(turns) == n and all(
            sorted(t) == list(range(8)) for t in turns.values()
        )
        active = peak = 0
        events = sorted(
            (t, delta)
            for r in records
            for t, delta in ((r["request_start_time"], 1), (r["request_end_time"], -1))
        )
        for _, delta in events:
            active += delta
            peak = max(peak, active)
        window = max(r["request_end_time"] for r in records) - min(
            r["request_start_time"] for r in records
        )
        mean_concurrency = sum(r["request_latency"] for r in records) / window
        assert peak == n, (n, peak)
        counts.append(
            {
                "streams": n,
                "successful": len(records),
                "errors": 0,
                "incomplete": 0,
                "peak_concurrency": peak,
                "mean_concurrency": mean_concurrency,
            }
        )
    assert sorted(c["streams"] for c in counts) == list(streams)
    return counts


def case_folder(case):
    name = f"tp2-{case['arm']}-{case['workload']}-{'mtp1' if case['mtp'] else 'decode'}-rep{case['repeat']}"
    return (
        BENCH
        / "results/gfx1201"
        / SLUGS[case["model"]]
        / "segmented-update-20261016"
        / name
    )


def wait_for_idle_gpus():
    code = "import json,torch; print(json.dumps([{'name':torch.cuda.get_device_name(i),'memory':torch.cuda.mem_get_info(i)} for i in range(torch.cuda.device_count())]))"
    deadline = time.monotonic() + 3600
    while True:
        state = json.loads(
            subprocess.check_output([sys.executable, "-c", code], text=True)
            .strip()
            .splitlines()[-1]
        )
        busy = [
            int(p.read_text())
            for p in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")
        ]
        assert len(state) == 2, state
        if all(row["memory"][0] > row["memory"][1] * 0.92 for row in state) and all(
            value < 10 for value in busy
        ):
            return {"devices": state, "busy_percent": busy}
        if time.monotonic() > deadline:
            raise TimeoutError(f"Both GPUs were not idle: {state}, busy={busy}")
        status("waiting-for-idle-gpus", devices=state, busy_percent=busy)
        time.sleep(15)


def server_env(source, cache):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "VLLM_",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        )
    }
    env.pop("ROCPROFILER_REGISTER_LIBRARY", None)
    env.update(
        PYTHONPATH=f"{source}:{ROOT}:{os.environ.get('PYTHONPATH', '')}",
        CUDA_VISIBLE_DEVICES="0,1",
        PYTORCH_ROCM_ARCH="gfx1201",
        NCCL_PROTO="Simple",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_LOGGING_LEVEL="INFO",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
    )
    return env


def run_case(manifest, case):
    folder = case_folder(case)
    output_tokens = 128 if case["workload"] == "short" else 512
    report = folder / "agent-multiturn-benchmarks.json"
    if (folder / "success.json").exists():
        success = read(folder / "success.json")
        assert success["report_sha256"] == sha(report)
        if case["workload"] == "short" and manifest.get("short_primer"):
            assert success["primer_report_sha256"] == sha(
                folder / "primer-multiturn-benchmarks.json"
            )
        validate_report(report, output_tokens)
        return
    assert not folder.exists(), (
        f"Inspect or archive the previous incomplete attempt: {folder}"
    )
    frozen(manifest)
    hardware = wait_for_idle_gpus()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", PORT)) != 0, "Port is occupied"
    folder.mkdir(parents=True)
    version = (
        "before"
        if case["arm"] == "segmented-before"
        else "after-mtp"
        if case["mtp"]
        else "after"
    )
    source = Path(manifest["sources"][version]["root"])
    model = manifest["models"][case["model"]]
    cache = (
        ROOT
        / ".cache/segmented-serving-20261016"
        / case["model"]
        / case["arm"]
        / ("mtp1" if case["mtp"] else "decode")
    )
    env = server_env(source, cache)
    resolved = (
        subprocess.check_output(
            [sys.executable, "-c", "import vllm; print(vllm.__file__)"],
            cwd=source,
            env=env,
            text=True,
        )
        .strip()
        .splitlines()[-1]
    )
    assert Path(resolved).resolve() == (source / "vllm/__init__.py").resolve(), resolved
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["snapshot"],
        "--served-model-name",
        model["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--language-model-only",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--seed",
        "20260715",
        "--generation-config",
        "vllm",
        "--worker-extension-cls",
        "scripts.segmented_serving_probe.SegmentedServingInspector",
    ]
    if case["arm"] != "default":
        command.extend(
            [
                "--attention-backend",
                "ROCM_SEGMENTED_ATTN",
                "--kernel-config",
                json.dumps(
                    {"enable_rocm_segmented_attn_autotune": manifest["autotune"]}
                ),
            ]
        )
    if case["mtp"]:
        assistant = manifest["models"][case["model"].replace("-fp8", "-assistant")]
        command.extend(
            [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "model": assistant["snapshot"],
                        "num_speculative_tokens": 1,
                    }
                ),
            ]
        )
    workload = copy.deepcopy(manifest["workload"])
    workload["spec"]["backend"].update(
        target=f"http://127.0.0.1:{PORT}",
        model=model["model"],
        extras={"temperature": 0},
    )
    workload["spec"]["tokenizer"] = {
        "kind": "huggingface_auto",
        "model": model["snapshot"],
    }
    workload["spec"]["data"][0]["output_tokens"] = output_tokens
    workload["spec"]["outputs"] = [{"kind": "json", "path": str(report)}]
    write(folder / "workload.json", workload)
    write(
        folder / "setup.json",
        {
            "case": case,
            "hardware": hardware,
            "command": command,
            "model": model,
            "source": source.as_posix(),
            "resolved_vllm": resolved,
            "env": {
                k: v
                for k, v in env.items()
                if k.startswith(
                    (
                        "VLLM_",
                        "NCCL_",
                        "CUDA_",
                        "TRITON_",
                        "TORCHINDUCTOR_",
                        "PYTHONPATH",
                    )
                )
            },
            "workload": workload,
            "separate_short_primer": bool(
                case["workload"] == "short" and manifest.get("short_primer")
            ),
        },
    )
    status("starting-server", case=case, folder=str(folder))
    started = time.monotonic()
    with (folder / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ACTIVE.append(server)
        try:
            deadline = time.monotonic() + 10800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {server.returncode}: {folder / 'server.log'}"
                    )
                try:
                    if request("/v1/models", timeout=5).get("data"):
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(5)
            else:
                raise TimeoutError("Server startup exceeded three hours")
            startup_s = time.monotonic() - started
            inspection = request(
                "/collective_rpc",
                {"method": "inspect_segmented_serving", "timeout": 120},
                timeout=180,
            )
            write(folder / "inspection.json", inspection)
            assert len(inspection["results"]) == 2
            for rank in inspection["results"]:
                assert rank["tp"] == 2 and rank["kv_cache_dtype"] == "fp8"
                for label, rel in (("ops", REL_OPS), ("tuner", REL_TUNER)):
                    actual = rank["segmented_sources"][label]
                    assert Path(actual["path"]).resolve() == source / rel, actual
                    assert (
                        actual["sha256"] == manifest["sources"][version]["sha256"][rel]
                    ), actual
                assert rank["dtype"] == "torch.bfloat16"
                attention = [
                    r
                    for r in rank["attention_layers"]
                    if r["head_dim"] in (64, 128, 256, 512)
                ]
                assert attention
                if case["arm"] != "default":
                    assert all(r["backend"] == "ROCM_SEGMENTED_ATTN" for r in attention)
                    if manifest["autotune"]:
                        assert any(t["records"] > 0 for t in rank["tuning_tables"]), (
                            rank
                        )
                if case["mtp"]:
                    assert rank["speculative_method"] == "mtp"
            smoke = request(
                "/v1/chat/completions",
                {
                    "model": model["model"],
                    "messages": [
                        {"role": "user", "content": "Reply with a short greeting."}
                    ],
                    "max_tokens": 32,
                    "temperature": 0,
                },
                timeout=300,
            )
            assert (
                smoke.get("choices")
                and smoke.get("usage", {}).get("completion_tokens", 0) > 0
            )
            write(folder / "smoke.json", smoke)
            client_env = dict(
                os.environ,
                GUIDELLM__MP_MAX_PENDING_BUFFER_PERCENT="2.0",
                GUIDELLM__MAX_WORKER_PROCESSES="1",
            )
            primer_request_count = 0
            request("/reset_prefix_cache", {}, timeout=60)
            primer_report = folder / "primer-multiturn-benchmarks.json"
            if case["workload"] == "short" and manifest.get("short_primer"):
                primer_workload = copy.deepcopy(workload)
                primer_workload["spec"]["outputs"] = [
                    {"kind": "json", "path": str(primer_report)}
                ]
                primer_path = folder / "primer-workload.json"
                write(primer_path, primer_workload)
                with (folder / "primer-benchmark.log").open("w") as primer_log:
                    primer = subprocess.Popen(
                        [
                            str(CLIENT_PY),
                            "-m",
                            "guidellm",
                            "run",
                            "--config",
                            str(primer_path),
                            "--disable-console-interactive",
                        ],
                        cwd=folder,
                        env=client_env,
                        stdout=primer_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    ACTIVE.append(primer)
                    status(
                        "priming-short-workload",
                        case=case,
                        server_pid=server.pid,
                        client_pid=primer.pid,
                        folder=str(folder),
                    )
                    try:
                        deadline = time.monotonic() + 14400
                        while primer.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError("Server exited during primer")
                            if time.monotonic() > deadline:
                                raise TimeoutError("Primer exceeded four hours")
                            time.sleep(10)
                        assert primer.returncode == 0, (
                            f"Primer exited {primer.returncode}: {folder / 'primer-benchmark.log'}"
                        )
                    finally:
                        stop(primer)
                primer_counts = validate_report(primer_report, output_tokens)
                primer_request_count = sum(r["successful"] for r in primer_counts)
                request("/reset_prefix_cache", {}, timeout=60)
            time.sleep(12)
            metrics_url = f"http://127.0.0.1:{PORT}/metrics"
            with urlopen(metrics_url) as response:
                (folder / "metrics-before.txt").write_bytes(response.read())
            with (folder / "benchmark.log").open("w") as log:
                client = subprocess.Popen(
                    [
                        str(CLIENT_PY),
                        "-m",
                        "guidellm",
                        "run",
                        "--config",
                        str(folder / "workload.json"),
                        "--disable-console-interactive",
                    ],
                    cwd=folder,
                    env=client_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                ACTIVE.append(client)
                status(
                    "benchmark",
                    case=case,
                    startup_s=startup_s,
                    server_pid=server.pid,
                    client_pid=client.pid,
                    folder=str(folder),
                    primer_requests=primer_request_count,
                )
                try:
                    deadline = time.monotonic() + 14400
                    while client.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError("Server exited during benchmark")
                        if time.monotonic() > deadline:
                            raise TimeoutError("Benchmark exceeded four hours")
                        time.sleep(10)
                    assert client.returncode == 0, (
                        f"GuideLLM exited {client.returncode}: {folder / 'benchmark.log'}"
                    )
                finally:
                    stop(client)
            if case["mtp"]:
                time.sleep(12)
            with urlopen(metrics_url) as response:
                (folder / "metrics-after.txt").write_bytes(response.read())
            if case["mtp"]:
                metric = "vllm:spec_decode_num_draft_tokens_total"
                values = []
                for stage in ("before", "after"):
                    values.append(
                        sum(
                            float(line.split()[-1])
                            for line in (folder / f"metrics-{stage}.txt")
                            .read_text()
                            .splitlines()
                            if line.startswith((metric + "{", metric + " "))
                        )
                    )
                assert values[1] > values[0], (
                    "MTP produced no draft tokens during the benchmark"
                )
            counts = validate_report(report, output_tokens)
            frozen(manifest)
            write(
                folder / "success.json",
                {
                    "case": case,
                    "counts": counts,
                    "report_sha256": sha(report),
                    "startup_s": startup_s,
                    "primer_report_sha256": sha(primer_report)
                    if primer_report.exists()
                    else None,
                    "completed": datetime.now(UTC).isoformat(),
                },
            )
            status("case-complete", case=case, folder=str(folder))
        finally:
            stop(server)
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", choices=tuple(SLUGS))
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    manifest = prepare(args.repeats, not args.static)
    if args.prepare_only:
        status("prepared", cases=len(manifest["plan"]))
        return
    completed = 0
    for case in manifest["plan"]:
        if args.model and case["model"] != args.model:
            continue
        run_case(manifest, case)
        completed += 1
        if args.max_cases and completed >= args.max_cases:
            break
    status("completed-selected-plan", cases=completed)


if __name__ == "__main__":
    main()
