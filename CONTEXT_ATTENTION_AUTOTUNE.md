# Persistent ROCm context attention startup tuning

Branch: `perf/rocm-context-attention-autotune`, based on vanilla vLLM
`8ebc5b0a18`. Initial feature commit: `d435f5f38f`; 65,536-token expansion: `0b19ba894c`; two-token minimum: `771ae79b64`; pruned batch/prefix expansion: `656dd56cf3`. This isolates launch configuration tuning; the attention math and
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

The backend captures its engine configuration during construction. After
attention page sizes are finalized, the GPU worker tunes representative
workloads **before model memory profiling and main KV-cache allocation**.
This keeps temporary tuning allocations out of vLLM's activation-memory peak.
Layers with identical local
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
They are clipped to the engine's token and sequence limits. Cached-prefix
representatives are **0 and every power of two from 1024 through 262,144**,
clipped to `max_model_len - Q`. Batch representatives are **1/2/4/8/16/32**,
clipped to `min(32, max_num_seqs, max_num_batched_tokens // Q)`.

The launcher now defaults to **8192 new tokens per step and 32 sequences**;
its validated model-length default remains 2048. Set `MAX_MODEL_LEN` to activate
longer prefixes, subject to actual model/KV-memory capacity. The tuning ranges
do not independently raise the engine's sequence-length limit.

Two memory screens prune the uniform synthetic workloads. Temporary storage
includes paged K/V rounded to physical pages, dense Q/K/V, outputs, FP32 comparison
scratch, and the 256 MiB benchmark cache-flush buffer. Its budget is the smaller
of half current free VRAM and one-quarter total VRAM, rounded down to a power
of two (**4 GiB** in these runs). Engine warmup uses the same conservative budget
for a model-cache estimate: full-attention pages across all nonshared layers,
plus other cache specs' per-sequence maximum storage. This is a screening
estimate, not vLLM's exact final cache-admission calculation. The real engine
still enforces its final KV capacity; shared-prefix batches may use less storage
than the independent-prefix synthetic inputs. Skipped shapes retain the original
launch fallback. Budget rounding reduces sensitivity to small startup-memory
fluctuations, and existing validated records remain reusable.

There are 24 candidates: query tiles 32/64/128, KV tiles 32/64, cache unrolling
1/4, and 4/8 warps; request unrolling and stages remain 1. Every candidate must
match the original launch's output at `atol=0.01, rtol=0.01` before timing. The
original configuration is a candidate. Candidates also require relative L2
error at most 0.5% against the original output. Winner selection uses median
GPU-event timing, excluding compilation and allocation. The original launch is timed after compilation. If its median latency is below
1 ms, Triton's L2-clearing benchmark uses 5 ms warmup and 20 ms measurement.
Otherwise candidates use the median of three launches after validation;
a 256 MiB cache flush precedes each launch, outside its timing interval. This
bounds measurement work for expensive queries and long prefixes alike.
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
    --max-tokens 65536 --max-model-len 65536 --max-seqs 1 \
    --warm-short-engine --probes \
    --output results/context-attention/buckets-65536.json
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --max-tokens 65536 --max-model-len 65536 --max-seqs 1 \
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
    --max-tokens 65536 --max-model-len 65536 --max-seqs 1 \
    --warm-short-engine --short-probes \
    --output results/context-attention/buckets-2-65536.json
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --max-tokens 65536 --max-model-len 65536 --max-seqs 1 \
    --load-only --warm-short-engine --short-probes \
    --output results/context-attention/buckets-2-65536-reload.json
```

Raw results: [`2–65,536 sweep`](results/context-attention/buckets-2-65536.json),
[`fresh-process automatic probes`](results/context-attention/buckets-2-65536-reload.json),
[`CPU tests`](results/context-attention/buckets-2-cpu-tests.log), and
[`short-query SDPA tests`](results/context-attention/buckets-2-gpu-tests.log).
Feature commit for the preceding two-token-minimum snapshot: `771ae79b64`.
Its source fingerprint/cache filename:
`653e368f4c73b1e1186242d59d94a41305ff127a212abc19d8ba25037570adff.json`.

An independent default-limit Qwen TP2 engine started with tuning forbidden in
spawned workers. Both ranks reported **`tuned=0 loaded=21`**, using the same
current-source cache. Engine startup was **36.66 s**; its greedy inference for
33-, 129-, and 513-token prompts completed, and cache SHA256/mtime stayed
unchanged. This was a cache/startup check, not an end-to-end speculative decoding
benchmark. Results: [`cached engine`](results/context-attention/buckets-2-engine.json)
and [`engine log`](results/context-attention/buckets-2-engine.log).

## Expanded batch/prefix range with memory pruning (2026-09-15)

Feature commit: `656dd56cf3`, still isolated over vanilla `8ebc5b0a18`.
Batch representatives now reach 32 and cached-prefix representatives reach
262,144. Query representatives remain 2–65,536, clipped to the configured token
budget. The first screen enforces `B*Q <= token budget` and `Q+C <= model limit`;
temporary-memory and model-cache estimates then prune the remaining tuples.
Startup logs show workload count, memory-pruned count, and memory budgets.

The worker now warms context attention before entering vLLM's memory profiler.
The profiling forward itself performs no tuning, so temporary synthetic KV
buffers cannot inflate the measured activation peak and shrink the serving cache.
File locking, atomic per-record persistence, and inference-only ceiling lookup
remain unchanged. The Triton kernel math remains unchanged.

Long-context standalone test: one gfx1201 R9700, per-rank Qwen TP2 BF16
HQ=12/HKV=2/D=256, page=784; token budget 8192, sequence limit 524288, max sequences
32. The sequence limit is intentionally larger than the maximum prefix so
`C=262144` plus new queries fits. This is a kernel test, not full-model 256K
serving validation.

The workload planner produced **630 token/sequence-feasible tuples**; the 4 GiB
scratch screen removed **52**, leaving **578**. A deliberately bounded sample
of **15** tuples covered all batch representatives, prefixes through 262144,
and queries through 8192. All **360 candidate/workload pairs** passed full-output
comparison to the original launch and the relative-L2 guard. Cold tuning of the
sample took **63.82 s**. Independent probes timed the stored winners:

| B | Q | C | Original (ms) | Tuned (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 262144 | 86.887 | 15.983 | 5.44x |
| 1 | 4 | 262144 | 86.254 | 15.545 | 5.55x |
| 1 | 8 | 262144 | 84.891 | 15.689 | 5.41x |
| 1 | 32 | 131072 | 42.029 | 7.985 | 5.26x |
| 1 | 256 | 65536 | 22.049 | 5.475 | 4.03x |
| 1 | 8192 | 65536 | 592.208 | 105.964 | 5.59x |
| 2 | 16 | 65536 | 23.274 | 4.093 | 5.69x |
| 2 | 4096 | 32768 | 295.388 | 53.877 | 5.48x |
| 4 | 8 | 32768 | 19.884 | 2.165 | 9.18x |
| 4 | 2048 | 16384 | 147.836 | 27.545 | 5.37x |
| 8 | 4 | 16384 | 19.478 | 1.374 | 14.18x |
| 16 | 4 | 8192 | 19.329 | 1.533 | 12.61x |
| 32 | 4 | 4096 | 20.607 | 1.389 | 14.84x |
| 32 | 32 | 1024 | 5.565 | 0.466 | 11.95x |
| 32 | 256 | 8192 | 81.610 | 13.867 | 5.89x |
| 1 | 3 | 260001 | 84.978 | 15.828 | 5.37x |
| 3 | 5 | 30001 | 13.608 | 1.952 | 6.97x |
| 31 | 3 | 4097 | 20.460 | 1.350 | 15.16x |

All **18 automatic-call probes** passed, including odd batch/query/prefix values
`(1,3,260001)`, `(3,5,30001)`, and `(31,3,4097)`. A lookup spy checked actual
workload metadata, and automatic results matched explicit saved-winner results
bitwise. Per-sequence FP32 references sampled rows 0, 31, 32, midpoint, and final
row (all rows for the three-token cases). Worst sampled row relative L2 was
**0.2422%**. Worst candidate relative L2 versus the original was **0.00996%**.
The four CPU persistence/recovery/range/pruning tests passed.

Reproduce the bounded long-context test and cache-only repeat:

```bash
source context-attention-env.bash
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --sample-long-context --output results/context-attention/pruned-long-context.json
.venv/bin/python benchmarks/kernels/bench_context_attention_buckets.py \
    --load-only --sample-long-context \
    --output results/context-attention/pruned-long-context-reload.json
```

Omit `--sample-long-context` to tune the complete memory-pruned standalone plan;
that complete 578-tuple sweep was not run. Engine startup always uses its complete
pruned plan. Artifacts: [`long-context results`](results/context-attention/pruned-long-context.json),
[`long-context reload`](results/context-attention/pruned-long-context-reload.json),
and [`CPU tests`](results/context-attention/pruned-cpu-tests.log).

The exact default engine configuration was tested with Qwen TP2, an 8192-token
budget, max sequences 32, model limit 2048, BF16 KV, ROCM_ATTN, and eager mode.
The complete plan contained **170** token/sequence-feasible workloads; the
model-cache estimate pruned **8**, keeping **162**. The first rank tuned **161**
missing records in **73.41 s**, reusing one record from the long-context sample;
the other rank loaded all 162 under the shared lock without duplicate tuning.
The cache now contains **176** total records, **4224** validated candidate pairs.

The second independent engine forbade `_tune_workload` in spawned workers. Both
ranks reported **`tuned=0 loaded=162`**, and cache SHA256/nanosecond mtime and
greedy token IDs for 33-, 129-, and 513-token prompts stayed identical. Overall
startup was **117.89 s** first and **39.30 s** cached. The standalone 15-record reload
also forbade tuning and left SHA256/mtime unchanged, loading in **0.0024 s**.
Measured engine KV capacities were 174762 and 177493 tokens on the two starts;
capacity is determined by each engine's memory profiling rather than by the
autotune range. This was not a comparison of serving throughput against baseline.

Reproduce engine cache persistence with the new defaults:

```bash
bash validate-context-attention-restart.bash
```

Engine snapshots/logs: [`pruned engine restart`](results/context-attention/pruned-engine-restart).
Current source/cache fingerprint:
`f4900adad45eebd5ecc7d93863ba83604b3a81d97c70e4f6e6df769e82bc50fa.json`.
All saved winners equal their minimum candidate median. The committed source
hash matches the measured cache identity, and repository hooks passed for the
feature commit.

The exact `bash serve-qwen38-context-autotune.bash` launcher also passed with
its new 8192-token/32-sequence defaults. Both TP workers loaded the completed
cache without tuning. `/health` and `/v1/models` returned HTTP 200, and a
`/v1/completions` request with **32 prompts of 129 tokens each** returned 32
choices and **128 generated tokens** (four per prompt). Cache SHA256/mtime
remained unchanged. The server was stopped afterwards. This validates a
32-prompt serving request, not a distributed performance measurement or proof
that all prompts necessarily entered the same scheduler iteration.
Artifacts: [`server response`](results/context-attention/pruned-server-smoke.json)
and [`server log`](results/context-attention/pruned-server.log).

## Matched prefill versus TPS AITER unified attention (2026-09-15)

The isolated comparison tested the union of **all 211 previously saved
distinct workloads** for HQ=12/HKV=2/D=256/page=784, each in BF16 and E4M3 FP8
KV: **422 cases** on one gfx1201 R9700. Both backends used identical logical
Q/K/V and the same page size, in their required packed versus NHD layouts.
This is Qwen TP2 rank-local kernel work, not distributed serving. The complete
578-cell long-context plan was not newly swept.

Historical source fingerprints were retuned before timing: **35 tuned, 176
loaded**, in **311.41 s**. Offline warmup selected this exact saved array with
an 8 GiB scratch allowance for the historical large-query points; engine
default pruning and serving limits are unchanged. The current cache now holds
**211 records / 5064 validated candidate pairs**. Automatic dispatch then
forbade retuning and left cache SHA256/mtime unchanged.

| KV / AITER path | Shapes | ROCM wins >2% | AITER wins >2% | Within 2% | Geometric-mean result |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16, all | 211 | 132 | 72 | 7 | AITER 1.049x faster; strongly depends on workload |
| BF16, 2D | 136 | 125 | 6 | 5 | ROCM 1.366x faster |
| BF16, 3D multi-query | 75 | 7 | 66 | 2 | AITER 2.016x faster |
| FP8, actual default fallback | 211 | 0 | 211 | 0 | AITER 5.515x faster |

### Launch-grid dimensions versus segmented KV

`context_attention_fwd` has a three-dimensional launch grid of
`(batch, query_head, query_tile)`. That is **not segmented KV attention**:
each workgroup scans the relevant cached prefix and causal current-chunk KV
itself, and produces the final output. There is no KV-segment launch dimension
or separate partial-attention reduction in this context path.

Our autotuner changes tile sizes, warp count, and loop unrolling within this
existing algorithm. It cannot add sequence segmentation through configuration.
SplitKV decode is a separate path for query-length-one rows; it does not give
`context_attention_fwd` a multi-query segmented implementation.

AITER can select a grid over query blocks, KV heads, and KV segments. Segment
workgroups compute partial attention, then a reduction merges their softmax
maxima, exponential sums, and outputs with correct global normalization.
This creates more KV parallelism for small queries with long cached prefixes;
it uses the same split-KV principle as our decode kernels for multi-token work.
It still attends to all valid KV tokens with causal masking.

The matched BF16 comparison therefore used our **same tuned standard context
algorithm** against either AITER's 2D or segmented 3D implementation. Ours won
125/136 2D shapes; AITER won 66/75 3D shapes, rather than winning every 3D
case. A new segmented-prefill kernel and reduction consuming the ROCM_ATTN
packed KV layout would be needed to close this capability gap. These prefill
results do not establish a winner against our separate SplitKV decode path.

`context_attention_fwd` already supports FP8 KV loads and scale-based
dequantization to the query dtype. The current autotune gate and tuning-cache
identity are BF16-only; FP8 used the unchanged default launch in this benchmark.
FP8 needs independently validated tuning configurations, rather than assuming
that the BF16 winners are optimal for its different memory/register behavior.

FP8 KV **bypasses the current BF16-only autotune gate**, even when the flag is
enabled. These measurements compare its actual default launch, not a separately
tuned FP8 candidate. To match attention values, FP8 current-chunk dense K/V
were dequantized from the same FP8 values AITER consumes; quantization and
layout preparation were excluded from timing.

All 422 cases passed full-output baseline/pair checks, sampled FP32 causal
references for every sequence, graph capture, and changed-Q graph replay.
Worst full-output pair relative L2 was **0.2878%**; worst sampled FP32 row error
was **0.2329%**. Five rounds alternated candidates with cold-L2 CUDA graph GPU
event timing. A six-case recheck of the final reusable harness passed and
preserved winner directions and the cache.

The AITER source was pinned to TPS `f07170b53a`. Our earlier roadmap review
incorrectly assumed all multi-query AITER calls use 2D. Its gfx1201 selector
chooses 3D for 75 of these shapes, and the BF16 patch changes all 75 launches.
The 2D prefill policy is unchanged by that commit. **Keep roadmap item 4:** our
BF16 tuning wins most 2D shapes but does not supersede patched segmented
prefill. The separate matched decode follow-up is now complete; end-to-end
backend selection remains open.

See [the complete paired table and methodology](../tps-rdna4-qwen38-tp2/PREFILL_AUTOTUNE_VS_AITER_UNIFIED.md),
[`benchmark source`](benchmarks/kernels/bench_rocm_prefill_vs_aiter.py),
[`exact workload manifest`](results/context-attention/aiter-prefill-manifest.json),
[`full timings`](results/context-attention/aiter-prefill-full.json),
[`CSV`](results/context-attention/aiter-prefill-full.csv), and
[`summary`](results/context-attention/aiter-prefill-summary.json).

### Separate SplitKV decode comparison

The query-length-one follow-up tested our production SplitKV split heuristic
and selected FlyDSL/Triton routes against complete TPS AITER `f07170b53a`,
with identical logical Q/K/V, KV dtype, page geometry, and causal semantics.
On one R9700 it produced **381 valid cold-cache pairs**: AITER was **1.424x
faster across 180 BF16 KV cases**, while ours was **1.425x faster across 201
FP8 KV cases**, geometrically. Actual FlyDSL activation alone was essentially
tied for BF16 (AITER 1.007x) and favored ours by **2.055x for FP8**.

Cache reuse changes the BF16 verdict materially. At TP2/batch 1/8192 tokens,
our active FlyDSL route took **45.860 us** versus AITER's **24.800 us**;
AITER was **1.849x faster**. FP8 favored ours: **46.240 us** versus **135.322 us**,
or **2.926x faster**. All 16 targeted reuse cases passed full-reference checks.
These TP labels are rank-local head shapes, not distributed TP4/TP8 runs.

There is no unconditional decode lead: TP8 FP8 `wave8` loses at long contexts
on hybrid pages but wins some equivalent 64-token-page shapes; high-batch
fallbacks also lose. **23 standard-page non-split Triton cases failed
correctness or faulted and were excluded from timing**, and 22 oversized
cases were memory-pruned. A separate pinned-vanilla standard-page reproducer
hit an LLVM compilation assertion. This is a decode/compiler follow-up,
not a failure of the prefill autotuning cache or a resolved root cause.

Keep roadmap item 4. These are kernel comparisons, not an end-to-end backend
verdict. See [the complete decode and cache-reuse report](../tps-rdna4-qwen38-tp2/SPLITKV_VS_AITER_UNIFIED_DECODE.md)
for active routes, all paired results, correctness details and reproduction.
