# Persistent ROCm context attention startup tuning

Branch: `perf/rocm-context-attention-autotune`, based on vanilla vLLM
`8ebc5b0a18`. Initial feature commit: `d435f5f38f`; 65,536-token expansion: `0b19ba894c`; two-token minimum: `771ae79b64`. This isolates launch configuration tuning; the attention math and
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

Query representatives are every power of two from **2 through 65,536**:
2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16,384,
32,768, and 65,536.
They are clipped to the engine's token and sequence limits. Context representatives are 0, 4096, and 8192, clipped to
its sequence limit. Batch representatives are 1 and up to 4, subject to scheduler
limits. The current 2048-token, batch-one launcher uses 21 buckets. With both limits
set to 65,536 and batch one, startup uses 46 buckets. The limit is an autotune
query ceiling; it does not automatically raise the launcher's engine limits.

There are 24 candidates: query tiles 32/64/128, KV tiles 32/64, cache unrolling
1/4, and 4/8 warps; request unrolling and stages remain 1. Every candidate must
match the original launch's output at `atol=0.01, rtol=0.01` before timing. The
original configuration is a candidate. Candidates also require relative L2
error at most 0.5% against the original output. Winner selection uses median
GPU-event timing, excluding compilation and allocation. Below 8192 query tokens,
Triton's L2-clearing benchmark uses 5 ms warmup and 20 ms measurement. For larger
queries, the median of three launches is measured after compilation/validation;
a 256 MiB cache flush precedes each launch, outside its timing interval. This
bounds startup work for kernels that take hundreds of milliseconds per call.
These are representative workload winners, not a guarantee of the optimum for
every arbitrary mixed request batch.

Serving uses ceiling buckets for batch/query/total sequence lengths and performs
only an in-memory lookup. For example, a query of 5155 tokens selects the
8192-query bucket when that bucket is available; it runs the actual 5155-token
shape without padding. It also selects a ceiling for total sequence length.
With the default 2048-token launcher, larger queries remain uncovered. Queries
above 65,536 and other uncovered larger batches/shapes fall back to the
original launch. Tuning is currently enabled for validated gfx1201 BF16 layouts
with head dimensions 64/128/256. ALiBi, sliding windows, sinks, noncausal attention,
FP8 caches/output, and cached-only K/V retain their existing paths. The
single-query decode path is separate and continues to use existing decode code.
Multi-token speculative verification through `ROCM_ATTN` uses
`context_attention_fwd` when `max_query_len > 1`, so it can use the 2/4/8/16
buckets too. For example, a three-token query selects the four-token bucket
and runs with its actual query length. This does not tune another attention
backend or guarantee an end-to-end speculative decoding speedup.

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

## Initial validation before the 65,536-token expansion (2026-09-15)

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

Cache filename for that earlier source snapshot:
`4c01d190a2a3609691495f72beed4bad2330e50111f5f29254f932fc52991f57.json`.
The earlier cold timing table remains the initial compilation/tuning measurement;
repository API cleanup intentionally created this new source fingerprint.
All repository pre-commit checks passed for both the feature and setup commits.

## Previous 32–65,536 power-of-two sweep (2026-09-15)

The new sweep covers every query power of two from 32 to 65,536. Measurements
use one GPU with the **per-rank Qwen TP2 dimensions** listed above, batch one,
and identical inputs for the original and tuned launches. These are kernel
measurements; full 65,536-token model inference was not tested. The physical
GPU has 64 CUs; Torch reports `multi_processor_count=32` on this ROCm runtime,
which is the value recorded in the cache fingerprint.

Pure causal prefill, with no cached prefix. Configuration columns are
`BLOCK_M/BLOCK_N/cache-unroll/warps`; stages and request unroll are always 1.
The original launch is `128/64/4/4`.

| Query tokens | Original (ms) | Tuned (ms) | Speedup | Winning configuration |
| ---: | ---: | ---: | ---: | --- |
| 32 | 0.107 | 0.011 | 9.63x | 32/32/1/8 |
| 64 | 0.107 | 0.014 | 7.92x | 32/64/1/8 |
| 128 | 0.111 | 0.020 | 5.48x | 32/32/1/8 |
| 256 | 0.131 | 0.036 | 3.68x | 32/64/1/4 |
| 512 | 0.264 | 0.074 | 3.59x | 64/64/1/4 |
| 1024 | 0.696 | 0.172 | 4.04x | 64/64/1/4 |
| 2048 | 2.079 | 0.478 | 4.35x | 64/64/1/4 |
| 4096 | 7.334 | 1.751 | 4.19x | 64/32/4/4 |
| 8192 | 28.245 | 6.604 | 4.28x | 64/32/4/4 |
| 16384 | 111.013 | 26.743 | 4.15x | 64/32/4/4 |
| 32768 | 446.592 | 108.327 | 4.12x | 64/32/4/4 |
| 65536 | 1805.347 | 436.953 | 4.13x | 128/32/4/8 |

Large query buckets with cached prefixes:

| Query tokens | Prefix tokens | Original (ms) | Tuned (ms) | Speedup | Winning configuration |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 4096 | 24.510 | 4.864 | 5.04x | 128/32/1/8 |
| 4096 | 8192 | 42.009 | 7.957 | 5.28x | 128/32/1/8 |
| 8192 | 4096 | 63.515 | 13.279 | 4.78x | 128/32/1/8 |
| 8192 | 8192 | 98.842 | 19.429 | 5.09x | 128/32/1/8 |
| 16384 | 4096 | 182.707 | 40.578 | 4.50x | 128/32/1/8 |
| 16384 | 8192 | 255.650 | 52.542 | 4.87x | 128/32/1/8 |
| 32768 | 4096 | 592.234 | 138.652 | 4.27x | 128/32/1/8 |
| 32768 | 8192 | 741.800 | 163.120 | 4.55x | 128/32/1/8 |

The above timings are the candidate measurements used to select winners.
Separate probes exercised the saved configurations on actual odd-sized inputs,
without tuning those sizes:

| Actual query | Prefix | Query ceiling | Original (ms) | Cached configuration (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5155 | 0 | 8192 | 11.754 | 2.902 | 4.05x |
| 5155 | 8192 | 8192 | 56.651 | 11.085 | 5.11x |
| 32769 | 0 | 65536 | 447.293 | 110.268 | 4.06x |
| 65535 | 0 | 65536 | 1805.110 | 435.109 | 4.15x |

All **960 candidate/workload pairs** (24 candidates × 40 records) passed the
original-output tolerance and the relative-L2 guard. Worst candidate relative L2
error was **0.1010%**. All four odd-sized probes also passed; their worst
relative L2 versus the original launch was **0.1016%**.
An independent FP32 causal reference checked rows 0, 31, 32, the midpoint, and
last query row for each probe, including the paged-prefix case. Worst row
relative L2 versus FP32 was **0.2343%** across
both original and tuned outputs. This is sampled FP32 validation, alongside
full-output comparisons to the original launch, rather than a full-model accuracy
evaluation.

The 65,536-token engine limits required **34 records**, tuned in **217.00 s**.
Warming the default 2048-token limits afterwards added only six clipped-prefix
records in **4.00 s**, reusing the seven common records; the shared file now has
**40 records**. This also validates that changing engine limits in the same
process checks for missing buckets instead of returning early just because a
table exists. A fresh process with `_tune_workload` forbidden loaded the saved
file in **0.039 s**. All saved winners match their minimum candidate timing;
cache SHA256 and nanosecond mtime were identical before and after loading.
Queries of 65,537 tokens correctly returned the original-launch fallback.

Cached-prefix representatives remain 0, 4096, and 8192; this change expands query
length coverage, not every possible long-prefix combination. Total sequence
length still needs a covering bucket, and otherwise uses the original launch.

Reproduce the full sweep and cache-only validation:

```bash
source context-attention-env.bash
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --warm-short-engine --probes \
    --output results/context-attention/buckets-65536.json
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --load-only --warm-short-engine \
    --output results/context-attention/buckets-65536-reload.json
```

The second command forbids tuning. To test a cold pass, use a fresh
`VLLM_CACHE_ROOT`. To enable the complete query range in the server, explicitly
raise both engine limits (subject to model/GPU memory capacity):

```bash
MAX_MODEL_LEN=65536 MAX_BATCHED_TOKENS=65536 \
    bash serve-qwen38-context-autotune.bash
```

The validated default launcher remains at 2048 tokens. Raw sweep/probe results:
[`buckets-65536.json`](results/context-attention/buckets-65536.json),
[`sweep log`](results/context-attention/buckets-65536.log), and
[`cache-only reload`](results/context-attention/buckets-65536-reload.json).
Source fingerprint/cache filename for the previous 32-token-minimum sweep:
`dfb277ee8348e7e61d5254c118b349034b9cfa68c61b85fb479f17f11057ce69.json`.

Two independent default-limit Qwen TP2 engines also passed the restart check
after the expansion. Both ranks on both starts reported `tuned=0 loaded=13`;
the second engine forbade tuning in its spawned workers. Cache SHA256/mtime and
greedy outputs for 33-, 129-, and 513-token prompts stayed identical. Engine
startup was **33.99 s** and **37.42 s**, with no context retuning. The three CPU
persistence/recovery/limit-change tests passed, and repository checks passed for
the tuning commit. Its final source hash matches the measured cache identity.
Logs and JSON: [`expanded restart validation`](results/context-attention/buckets-65536-restart).

## Dedicated 2/4/8/16-token tuning (2026-09-15)

The minimum query representative is now **2**, retaining every power of two up
to 65,536. No attention math, tiles, candidate list, or decode kernel changed.
This targets small multi-token queries, including speculative verification when
it uses `ROCM_ATTN`. The dispatch in
[`chunked_prefill_paged_decode.py`](vllm/v1/attention/ops/chunked_prefill_paged_decode.py)
calls `context_attention_fwd` for `max_query_len > 1`. Its saved configuration
lookup therefore covers these queries. Single-token decode still uses the
existing decode implementation. Query representatives describe total target
verification tokens, not the configured number of draft tokens alone.

Previously, queries below 32 already selected the 32-query ceiling bucket.
Dedicated small representatives let startup benchmark their actual workload
instead of borrowing that representative. On the tested Qwen TP2 dimensions,
**all twelve new full-range workloads chose the same parameters as the previous
32-query bucket**: `BLOCK_M=32, BLOCK_N=32, cache-unroll=1, warps=8`, with stages
and request unroll 1. Accordingly, the gains below are versus vanilla vLLM's
original `128/64/4/4` launch, not an additional gain over the previous autotuned
branch. This range extension alone does not establish speculative end-to-end
throughput gains.

New representatives, batch one, BF16, local HQ=12/HKV=2/D=256, page=784, on the
same gfx1201 R9700 and runtime as the previous sweep. Compilation/allocation are
excluded; timing uses Triton's median L2-clearing benchmark described above.

| Query | Cached prefix | Original (us) | Tuned (us) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 106.36 | 9.92 | 10.72x |
| 2 | 4096 | 1388.21 | 267.08 | 5.20x |
| 2 | 8192 | 2632.45 | 520.25 | 5.06x |
| 4 | 0 | 104.52 | 9.72 | 10.75x |
| 4 | 4096 | 1342.98 | 261.04 | 5.14x |
| 4 | 8192 | 2619.03 | 513.09 | 5.10x |
| 8 | 0 | 103.92 | 9.92 | 10.48x |
| 8 | 4096 | 1357.85 | 261.78 | 5.19x |
| 8 | 8192 | 2619.05 | 512.63 | 5.11x |
| 16 | 0 | 104.36 | 10.08 | 10.35x |
| 16 | 4096 | 1352.30 | 262.40 | 5.15x |
| 16 | 8192 | 2614.45 | 514.37 | 5.08x |

The current-source sweep validated **1344 candidate/workload pairs** across
**56 saved records**. The full 65,536-token limits tuned 46 records in 225.31 s;
warming the 2048-token launcher afterwards added only ten clipped-prefix records
in 6.57 s, reusing eleven shared records. Total tuning time was **231.92 s**.
The default launcher now needs 21 of those records.

Separate probes tested queries **2, 3, 4, 5, 8, 9, 16, and 17**, each with
**0, 4096, and 8192** cached-prefix tokens (24 cases). All full outputs matched
the original launch bitwise on these short inputs, and the independent sampled
FP32 reference passed, with worst row relative L2 **0.2332%**. Exact and odd
sizes exercised ceiling selection; for example, 3 selects 4 and 17 selects 32.
A fresh cache-only process also ran each probe with `_launch_config=None`, so
normal runtime lookup selected the saved configuration. A lookup spy required
exactly one call with the actual batch/query/total-sequence lengths for each
automatic invocation. Every automatic output
was bitwise identical to the explicitly supplied winner. No probe tuned at
runtime.

With `_tune_workload` patched to raise, fresh-process loading took
**0.037 s** and left cache SHA256 and nanosecond mtime unchanged. Every saved
winner equals its minimum median candidate timing. Worst candidate relative L2
across the full range was **0.1010%**. Three CPU persistence/recovery/limit-change
tests and three new GPU SDPA tests passed; the latter exercised 2-, 3-, and
9-token, batch-two paged queries at head dimensions 64, 128, and 256 against
all 24 launch candidates.

Reproduce the current range and normal automatic lookup probes:

```bash
source context-attention-env.bash
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --warm-short-engine --short-probes \
    --output results/context-attention/buckets-2-65536.json
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --load-only --warm-short-engine --short-probes \
    --output results/context-attention/buckets-2-65536-reload.json
```

Raw results: [`2–65,536 sweep`](results/context-attention/buckets-2-65536.json),
[`fresh-process automatic probes`](results/context-attention/buckets-2-65536-reload.json),
[`CPU tests`](results/context-attention/buckets-2-cpu-tests.log), and
[`short-query SDPA tests`](results/context-attention/buckets-2-gpu-tests.log).
Current feature commit: `771ae79b64`. Current source fingerprint/cache filename:
`653e368f4c73b1e1186242d59d94a41305ff127a212abc19d8ba25037570adff.json`.

An independent default-limit Qwen TP2 engine started with tuning forbidden in
spawned workers. Both ranks reported **`tuned=0 loaded=21`**, using the same
current-source cache. Engine startup was **36.66 s**; its greedy inference for
33-, 129-, and 513-token prompts completed, and cache SHA256/mtime stayed
unchanged. This was a cache/startup check, not an end-to-end speculative decoding
benchmark. Results: [`cached engine`](results/context-attention/buckets-2-engine.json)
and [`engine log`](results/context-attention/buckets-2-engine.log).
