# Persistent ROCm context attention startup tuning

Branch: `perf/rocm-context-attention-autotune`, based on vanilla vLLM
`8ebc5b0a18`. Feature commit: `d435f5f38f`. This isolates launch configuration tuning; the attention math and
Triton kernel are unchanged. It does not depend on the SplitKV feature branch.

Start the Qwen TP2 server in this workspace:

```bash
cd /app/vllm-perf-rocm-context-attention-autotune
bash serve-qwen38-context-autotune.bash
```

The launcher enables `VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE=1`, selects
`ROCM_ATTN`, uses BF16 KV cache, and sets `NCCL_PROTO=Simple`. It uses eager
execution for the bounded validation setup. You can disable this feature by
setting `VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE=0`.

## Startup and persistence

The backend captures its engine configuration during construction. The first
profiling forward, before main KV-cache allocation, tunes representative
workloads using the finalized attention page size. Layers with identical local
attention dimensions reuse the same in-memory table. A file lock allows TP ranks
on identical GPUs to share one tuning pass, and winners are saved atomically
after each bucket completes.

Files live under:

```text
$VLLM_CACHE_ROOT/rocm_context_attention/<fingerprint>.json
```

This launcher defaults `VLLM_CACHE_ROOT` to `.cache/vllm` in this checkout. Each
file records the workload, winning launch parameters, and all valid candidate
timings. The fingerprint includes GPU name/architecture/CU count, Torch/HIP and
Triton versions, compiler environment, kernel/tuner source hashes, local Q/KV
head counts, head dimension, physical page size, dtype, scale, and candidate set.
A source or compiler change intentionally creates a new cache identity.

The next engine start reads the saved winners and logs `tuned=0 loaded=N`.
It does not rewrite a complete cache. Raising engine limits adds missing
workload buckets to the existing file. A truncated or invalid file is rejected
and rebuilt. Normal Triton JIT compilation and other vLLM/GDN warmup mechanisms
remain separate; this feature removes repeated **context attention tuning**.

## Workloads and scope

Query representatives are 32, 128, 512, and 2048, clipped to the engine's token
and sequence limits. Context representatives are 0, 4096, and 8192, clipped to
its sequence limit. Batch representatives are 1 and up to 4, subject to scheduler
limits. The current 2048-token, batch-one launcher uses seven buckets.

There are 24 candidates: query tiles 32/64/128, KV tiles 32/64, cache unrolling
1/4, and 4/8 warps; request unrolling and stages remain 1. Every candidate must
match the original launch's output at `atol=0.01, rtol=0.01` before timing. The
original configuration is a candidate. Winner selection uses median GPU-event
timing with Triton's L2-clearing benchmark, excluding compilation and allocation.
These are representative workload winners, not a guarantee of the optimum for
every arbitrary mixed request batch.

Serving uses ceiling buckets for batch/query/total sequence lengths and performs
only an in-memory lookup. Uncovered larger batches/shapes fall back to the
original launch. Tuning is currently enabled for validated gfx1201 BF16 layouts
with head dimensions 64/128/256. ALiBi, sliding windows, sinks, noncausal attention,
FP8 caches/output, and cached-only K/V retain their existing paths. The
single-query decode path is separate and continues to use existing decode code.

## Repeat the restart validation

```bash
bash validate-context-attention-restart.bash
```

This starts two independent Qwen TP2 engines, performs deterministic greedy
inference for 33-, 129-, and 513-token prompts, and compares output tokens and
cache SHA256/mtime. During the second start the tuning function is patched to
raise if called. It also requires a cache-load log. Existing valid caches are
reused by the first start too; set a fresh `VLLM_CACHE_ROOT` to exercise a cold
pass. JSON outputs and logs are saved in `results/context-attention-restart`.

Kernel correctness and persistence tests:

```bash
source context-attention-env.bash
.venv/bin/python -m pytest tests/kernels/attention/test_prefix_prefill.py \
    -k rocm_context_tuning -q
```

## Validation results (2026-09-15)

Hardware: two AMD Radeon AI PRO R9700 GPUs, gfx1201, 64 CUs each.
Runtime: Torch 2.11.0+rocm7.14.0, Triton 3.7.1; existing shared ROCm extensions
and uv-managed Python environment. Qwen TP2 local full-attention dimensions:
HQ=12, HKV=2, D=256, BF16, physical page=784.

| Check | Result |
| --- | --- |
| First engine's context tuning | 7 buckets tuned, 24 candidates validated per bucket, 103.20 s |
| Other TP rank on first start | Loaded the same 7 buckets; no duplicate tuning |
| Second independent engine | Both ranks: `tuned=0 loaded=7`, 0.00–0.05 s including locking |
| Overall offline engine startup | 149.33 s first start, 36.16 s cached start |
| Persistent file | SHA256 and nanosecond mtime identical between starts |
| Winner integrity | Every saved `best` equals the lowest-median candidate in its saved timings |
| Restart inference | Identical greedy token IDs for 33-, 129-, and 513-token synthetic prompts |
| Pytest | 5 passed: 3 GPU SDPA boundary cases, persistent reuse, corrupt-cache recovery |
| Style | Ruff check/format and shell syntax checks passed |

The second engine had `_tune_workload` replaced with a function that raises if
called, including in spawned workers. No tuning was attempted. Overall startup
also includes other compilation/warmup mechanisms; the reduction is not a pure
measurement of the JSON load alone.

Saved tuning measurements (batch=1, same dtype/head dimensions/page):

| Query | Cached prefix | Original launch (us) | Saved winner (us) | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 104.32 | 10.78 | 9.68x |
| 32 | 2016 | 715.81 | 132.68 | 5.39x |
| 128 | 0 | 110.88 | 20.44 | 5.42x |
| 128 | 1920 | 731.29 | 148.68 | 4.92x |
| 512 | 0 | 264.80 | 74.00 | 3.58x |
| 512 | 1536 | 1162.59 | 241.08 | 4.82x |
| 2048 | 0 | 2092.89 | 486.05 | 4.31x |

These are the tuner measurements used to choose configurations, not an
independent end-to-end serving throughput benchmark. Actual requests can have
different strides and mixtures from dummy startup workloads; normal Triton JIT
specialization can still occur even though runtime tuning cannot.

The feature-disabled original launch produced identical greedy outputs on the
33- and 129-token synthetic prompts. On the 513-token synthetic prompt, one of
eight generated tokens differed (`16` versus `15` at the second generated
position). Both tuned starts reproduced their tokens exactly. Launch tiles alter
BF16 intermediate rounding, so bitwise output equivalence to the original
launch is not promised. This bounded check is not a full model accuracy eval.

Artifacts are in [`results/context-attention`](results/context-attention):
`cold.json`, `warm.json`, `baseline.json`, engine logs, kernel test logs, and
`summary.json`.

To investigate that token difference, a separate cached engine compared each
actual model prefill attention output with the original launch on the same
Q/K/V and paged cache. All **128 comparisons** passed `atol=0.01, rtol=0.01`;
worst relative L2 error was **0.00080882 (0.080882%)**. Maximum absolute
error was 0.25 on large-magnitude model activations; the relative tolerance
passed. This supports BF16 tile-rounding variation rather than a detected
attention correctness failure. It does not establish downstream accuracy on a
full evaluation dataset.

Repeat this additional check:

```bash
source context-attention-env.bash
CONTEXT_TUNING_FORBID=1 CONTEXT_VALIDATE_NUMERICS=1 .venv/bin/python \
    benchmarks/kernels/validate_context_attention_startup.py \
    --output results/context-attention/numerics.json
```

The exact `bash serve-qwen38-context-autotune.bash` launcher was also tested:
both TP workers loaded all seven cached configurations, `/health` and
`/v1/models` returned HTTP 200, and `/v1/completions` generated 32 tokens for a
natural-language prompt with HTTP 200. The server was stopped after validation
to leave both GPUs available. Response: `results/context-attention/server-smoke.json`;
startup/request log: `results/context-attention/server.log`.

The committed feature was validated again after the repository API checks:
`bash validate-context-attention-restart.bash` passed with two independent
engines, both ranks loaded seven configurations with zero tuning, cache
SHA256/mtime stayed unchanged, and greedy output matched. Cached startup was
36.28 s and 37.01 s. The final file's kernel/tuner hashes match the committed
source, and every winner matches its stored minimum median timing. Logs and
JSON snapshots: `results/context-attention/final-restart`.

Final cache filename:
`4c01d190a2a3609691495f72beed4bad2330e50111f5f29254f932fc52991f57.json`.
The earlier cold timing table remains the initial compilation/tuning measurement;
repository API cleanup intentionally created this new source fingerprint.
All repository pre-commit checks passed for both the feature and setup commits.
