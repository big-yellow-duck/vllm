# TP2 offline comparison of complete attention paths

## Scope

Compare Qwen/Qwen3.8-27B-FP8 offline generation on two physical Radeon AI PRO
R9700 GPUs (gfx1201). Model weights remain FP8 in every run; **BF16 and FP8
below refer to the full-attention KV cache**. GDN state and kernels remain
unchanged.

The three configurations are:

| Variant | vLLM source | Attention |
| --- | --- | --- |
| Vanilla | Pinned upstream `8ebc5b0a182b3351a1c62fe73e97be2c489ed2a6` | Default `ROCM_ATTN` |
| AITER | The same pinned upstream source | Explicit `ROCM_AITER_UNIFIED_ATTN`, TPS AITER `f07170b53a178d1de20001b130bb40b44127be12` |
| Ours | `perf/rdna4-prefill-autotune-splitkv` | `ROCM_ATTN` with persistent context-prefill tuning and our SplitKV decode |

Measured combined source revision: `695569a5af`. Later packaging changes
update aggregation and documentation, leaving the measured attention source
and its cache fingerprints unchanged.

The combined checkout is
`/app/vllm-perf-rdna4-prefill-autotune-splitkv`. Its only production source
changes over the pinned baseline are the context tuner, SplitKV kernels and
routing, and the workspace/startup integration needed by those features.
It contains no custom FP8 GEMM, GDN, all-reduce, or local-argmax optimization.
The vanilla/AITER checkout is `/app/vllm-bench-rdna4-attention-vanilla`.

All configurations use the same venv and prebuilt ROCm native extensions.
Their exact binary hashes and resolved paths are recorded in the manifest;
the source pin describes the Python/Triton checkout, and does not imply the
shared native binaries were rebuilt from that pin.

## Whole-generation results

Speedups below are geometric means of per-shape median batch times. Each KV
dtype has all **18** requested input/output/concurrency combinations. The
BF16 comparison uses the complete first round; FP8 uses both complete rounds.

| KV cache | AITER speedup vs vanilla | Ours speedup vs vanilla | AITER vs ours |
| --- | ---: | ---: | --- |
| BF16 | 1.358x | 1.308x | AITER 3.85% faster |
| FP8 | 1.410x | 1.401x | AITER 0.61% faster (near tie) |

| KV cache | Concurrency | AITER speedup vs vanilla | Ours speedup vs vanilla | Ours speedup vs AITER |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1.316x | 1.312x | 0.997x |
| BF16 | 2 | 1.362x | 1.322x | 0.971x |
| BF16 | 4 | 1.398x | 1.289x | 0.922x |
| FP8 | 1 | 1.494x | 1.526x | 1.021x |
| FP8 | 2 | 1.391x | 1.398x | 1.006x |
| FP8 | 4 | 1.349x | 1.289x | 0.956x |

**Verdict:** our combined attention work does not supersede the patched AITER
backend. BF16 favors AITER overall; all six concurrency-1 shapes are within
2%, while its advantage grows at higher concurrency. FP8 is a near tie
overall, with modest gains for ours on many short/medium cases and a larger
AITER lead on a few long concurrency-4 cases. Keep item 4 as an independent
BF16 correctness/launch-policy PR, and retain our FP8 SplitKV and context tuner
as separate features. These results concern this attention-only TP2 setup;
other common optimizations can change attention's share of engine latency.

The clearest AITER wins are at 8192 input tokens/concurrency 4. For 32 output
tokens, median BF16 times are **10.430 s AITER / 13.267 s ours**; FP8 times are
**10.296 s / 13.484 s**. At 256 output tokens they are **18.643 s / 21.191 s**
for BF16 and **19.199 s / 21.605 s** for FP8. Ours instead wins the FP8
8192-input/256-output concurrency-1 case: **10.383 s vs 10.775 s**.

## Protocol

- Actual TP2 on physical GPUs 0 and 1; no TP4/TP8 emulation.
- Input lengths: 128, 2048, 8192 tokens. Output lengths: 32, 256 tokens.
- Concurrency: 1, 2, 4 requests submitted together as a closed offline batch.
  All six input/output combinations are tested at each concurrency, giving
  18 shapes per KV dtype.
- Deterministic token-ID prompts differ between requests in a batch but are
  identical between variants. Greedy generation ignores EOS and returns the
  requested number of tokens; detokenization and prefix caching are disabled.
- `max_model_len=9216`, `max_num_batched_tokens=8192`, `max_num_seqs=32` and
  `gpu_memory_utilization=0.90` are identical. Chunked prefill can create mixed
  prefill/decode steps when the batch exceeds the token budget.
- `FULL_DECODE_ONLY` graphs with capture sizes 1/2/4; ordinary Triton GEMM and
  GDN; custom all-reduce disabled; `NCCL_PROTO=Simple` in every variant.
- AITER's master optimization flag stays **off**, and other AITER feature
  flags are disabled. Explicit backend selection directly loads its unified
  attention wrapper, without enabling AITER GEMM, RMSNorm, GDN or other fusions.
- One full workload warmup precedes three timed repetitions per shape. The
  second fresh-engine round reverses variant and workload order. At the user's
  request, further repeats stopped after all FP8 rounds and ten BF16 combined
  repeat shapes. The balanced comparison uses **three BF16 samples** per
  variant/shape and **six FP8 samples** per variant/shape. Those ten extra
  BF16 shapes are repeatability diagnostics only. **486 samples** enter the
  comparison; **516 completed samples** are saved in total. Startup, tuning,
  compilation warmup and diagnostic profiling are excluded from timings.
- Report median wall time for `LLM.generate` and output tokens/second over
  that whole prefill-plus-decode interval. These are offline batch completion
  measurements, not network TTFT or steady-state serving throughput.

## Native backend differences

Keep each backend's normal storage and query behavior rather than converting
layouts per request. For this hybrid model the physical attention pages are
784 BF16 / 1568 FP8 tokens under `ROCM_ATTN`, and 832 BF16 / 1600 FP8 under
AITER, which prefers a 64-token base block. The resulting Mamba padding and
KV-writer costs are part of the complete backend comparison.

AITER's FP8 KV path also quantizes queries to FP8. `ROCM_ATTN` keeps them in
BF16 and uses fresh BF16 current-chunk K/V for context prefill. AITER attends
to current K/V after cache quantization. Consequently the FP8 comparison is
between native engine paths, with different query/current-chunk precision;
it is not an equal-query-precision isolated kernel comparison. Worker
inventories record scales, query quantization and page sizes, and raw output
token IDs allow agreement to be assessed.

## FP8 tuner extension and validation

The isolated prefill branch now contains commit `62dcbf46a1`, adding E4M3 FP8
KV tuning for BF16 queries. It prepares correctly packed 16-byte K-cache
vectors, uses FP8-specific synthetic KV/scales, and puts KV dtype into both
the persistent fingerprint and in-memory key. BF16 winners cannot leak into
an FP8 lookup, even at the same physical page size. The kernel math and
candidate set are unchanged.

All **17** focused tuner tests passed. These include every launch candidate
against FP32 attention references at ragged query/page boundaries, separate
non-unit K/V scales for FP8, persistent reuse, corrupt-cache recovery, and
range/pruning behavior.

The real FP8 engine tuned **341 buckets**, pruning 24 from its bounded plan,
and saved **8184 valid candidate measurements**. The worst candidate relative
L2 difference against the original launch was **0.101947%**. Its cold pass
took **588.10 seconds**; the other TP rank loaded the same 341 records instead
of benchmarking independently. This is context-tuning time, not full engine
startup or inference latency.

An independent fresh FP8 engine subsequently loaded all **341 buckets with
zero retuning** on both TP ranks. One rank logged a **0.02-second** load;
its total engine startup was **57.16 seconds**. Both cache SHA256 and
nanosecond modification time were unchanged before/after startup. This is a
real engine-restart reuse check, in addition to the persistence unit tests.

For the BF16 engine under those same limits, memory pruning retained
**315 buckets** and pruned 50. All **7560 candidate measurements** passed;
the worst relative L2 difference was also **0.101947%**. Cold context tuning
took **258.66 seconds**, and the other rank loaded all 315 records. Its total
engine startup was **359.80 seconds**, which also includes model loading,
compilation and graph capture and is excluded from generation timing.

The independent BF16 restart loaded all **315 buckets with zero retuning**,
taking **0.01 s / 0.07 s** across its two ranks. Cache hash/mtime snapshots
were unchanged. Its total startup was **53.94 seconds**.

Initial FP8 smoke runs for both AITER and the combined branch completed
four-token generation at concurrency 1/2/4 with matching token IDs. GPU traces
confirm AITER's 2D/segmented-3D kernels and the combined branch's context
kernel plus FlyDSL `stage_0`/`reduce_kernel_0` decode. The FlyDSL route marker
is `generic_tile32`.

## Repeated FP8 optimized-path comparison

The two optimized FP8 paths have completed both fresh-engine rounds: **six
samples per shape, 18 shapes**. Their overall geometric-mean difference is
small enough to treat as a near tie. The median-per-shape AITER advantage is
**0.61%** geometrically; ours wins eight shapes, AITER three, and seven are
within 2%. AITER's fewer wins are larger, so the win count does not imply an
overall win for ours.

| Concurrency | Ours speedup vs AITER | Interpretation |
| --- | ---: | --- |
| 1 | 1.0215x | Ours modestly faster |
| 2 | 1.0055x | Near tie |
| 4 | 0.9560x | AITER 4.61% faster |
| All 18 shapes | 0.9939x | Near tie; AITER 0.61% faster |

The isolated FP8 decode advantage does not establish a universal native
engine advantage once context attention, mixed scheduling, native storage
and query quantization are included. FP8 unified-attention policy is unchanged
by TPS's BF16 patch; these FP8 results describe the existing AITER backend,
not a gain attributable to the BF16 patch itself.

## Interpreting generation outputs and routes

Kernel-reference validation and generated-token agreement answer different
questions. The vanilla engine itself produced different greedy token IDs
between some repeated concurrent batches, despite fixed prompts and token
counts. A first differing token can cause later positions to diverge. These
runs therefore do not establish exact-output equivalence or model-quality
parity between backends. Raw IDs and within-variant repeat agreement are
retained; every completed batch must still return its requested token count.
The source of vanilla's variability has not been isolated.

Untimed route profiles include a long input at each concurrency. They record
actual GPU kernel names and context-tuner lookups, including lookups that
fall back to the original launch. A mixed scheduler step can have a maximum
query length much larger than its average per request; a uniform tuning
bucket at that batch/query combination may have been pruned by the shared
token budget. Such a miss uses the existing kernel configuration and does
not start runtime benchmarking. These are diagnostic probes, not complete
kernel-time attribution for every timed workload.

In the first combined BF16 long-input probe, each TP rank had **64 tuned
lookups out of 128** context calls. Hits were `(B=1, Q=8192, seq=8192)` and
`(B=3, Q=4, seq=8194)`. Misses included `(2, 8191, 8193)`,
`(3, 8190, 8194)` and `(4, 8188, 8195)`: the maximum per-request query can
approach 8192 even though all requests share an 8192-token scheduler budget.
The current planner represents uniform queries, so those batch/max-query
buckets are absent. This is a real tuning-coverage limitation included in the
inference results, not a runtime-retuning delay. A future ragged-workload
planner would need to represent maximum query length separately from total
scheduled query tokens.

## Repeatability and numerical limits

For the 18 FP8 shapes, median round-2/round-1 latency ratios were **1.00022**
for vanilla, **0.99972** for AITER and **0.99999** for ours. The ten completed
BF16 repeat diagnostics for ours had a median ratio of **0.99913** and a range
of **0.97657–1.00023**. No second-round BF16 samples enter its main comparison.
Some short-batch samples had sizable spikes: BF16 ours at 128/32/concurrency 4
ranged **1.228–2.419 s**, median **1.368 s**; FP8 ours at
2048/32/concurrency 2 ranged **2.119–3.499 s**, median **2.135 s**. Vanilla's
short 128/32/concurrency-2 FP8 round median changed by 9.91%, although its
across-shape median drift was only 0.02%. Raw ranges remain available; small
short-batch differences are less conclusive than the stable long-batch gaps.

Within the main samples, exact repeated IDs were stable on **12/18 vanilla,
10/18 AITER and 11/18 ours BF16** shapes; on **6/18 vanilla, 10/18 AITER and
9/18 ours FP8** shapes. First-sample exact agreement with vanilla was **4/18
AITER and 6/18 ours BF16**, and **4/18 AITER and 2/18 ours FP8**. These are
descriptive greedy-ID checks, not accuracy scores: vanilla itself varies,
early differences propagate, and native FP8 query precision also differs.
The 17 independent kernel/tuner tests passed, but model-quality parity is
not established by this timing experiment.

## All workload medians

All times are seconds per completed offline batch. Values below 1x in the
last column mean AITER was faster. BF16 uses three samples, FP8 six, equally
for all variants within each dtype.

### BF16 KV cache

| Input | Output | Concurrency | Vanilla s | AITER s | Ours s | Ours speedup vs AITER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 1 | 1.094 | 1.055 | 1.073 | 0.983x |
| 128 | 32 | 2 | 1.169 | 1.123 | 1.237 | 0.908x |
| 128 | 32 | 4 | 1.703 | 1.225 | 1.368 | 0.896x |
| 128 | 256 | 1 | 8.412 | 8.038 | 8.071 | 0.996x |
| 128 | 256 | 2 | 11.371 | 8.562 | 8.609 | 0.995x |
| 128 | 256 | 4 | 12.047 | 9.098 | 9.160 | 0.993x |
| 2048 | 32 | 1 | 2.076 | 1.504 | 1.503 | 1.001x |
| 2048 | 32 | 2 | 2.716 | 2.119 | 2.124 | 0.998x |
| 2048 | 32 | 4 | 3.968 | 3.271 | 3.267 | 1.001x |
| 2048 | 256 | 1 | 10.601 | 8.519 | 8.558 | 0.995x |
| 2048 | 256 | 2 | 14.186 | 9.619 | 9.669 | 0.995x |
| 2048 | 256 | 4 | 15.878 | 11.174 | 11.178 | 1.000x |
| 8192 | 32 | 1 | 4.539 | 3.294 | 3.256 | 1.012x |
| 8192 | 32 | 2 | 7.660 | 5.717 | 5.980 | 0.956x |
| 8192 | 32 | 4 | 14.400 | 10.430 | 13.267 | 0.786x |
| 8192 | 256 | 1 | 20.954 | 10.378 | 10.405 | 0.997x |
| 8192 | 256 | 2 | 24.285 | 13.334 | 13.672 | 0.975x |
| 8192 | 256 | 4 | 31.765 | 18.643 | 21.191 | 0.880x |

### FP8 KV cache

| Input | Output | Concurrency | Vanilla s | AITER s | Ours s | Ours speedup vs AITER |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 1 | 1.127 | 1.089 | 1.070 | 1.018x |
| 128 | 32 | 2 | 1.237 | 1.245 | 1.233 | 1.010x |
| 128 | 32 | 4 | 1.395 | 1.396 | 1.329 | 1.050x |
| 128 | 256 | 1 | 8.838 | 8.255 | 8.077 | 1.022x |
| 128 | 256 | 2 | 9.459 | 8.838 | 8.629 | 1.024x |
| 128 | 256 | 4 | 9.867 | 9.398 | 9.152 | 1.027x |
| 2048 | 32 | 1 | 2.097 | 1.532 | 1.505 | 1.017x |
| 2048 | 32 | 2 | 2.748 | 2.152 | 2.135 | 1.008x |
| 2048 | 32 | 4 | 3.981 | 3.308 | 3.265 | 1.013x |
| 2048 | 256 | 1 | 13.451 | 8.745 | 8.548 | 1.023x |
| 2048 | 256 | 2 | 14.606 | 9.916 | 9.649 | 1.028x |
| 2048 | 256 | 4 | 16.185 | 11.501 | 11.170 | 1.030x |
| 8192 | 32 | 1 | 5.806 | 3.294 | 3.258 | 1.011x |
| 8192 | 32 | 2 | 8.648 | 5.674 | 5.946 | 0.954x |
| 8192 | 32 | 4 | 16.423 | 10.296 | 13.484 | 0.764x |
| 8192 | 256 | 1 | 29.131 | 10.775 | 10.383 | 1.038x |
| 8192 | 256 | 2 | 32.621 | 13.739 | 13.591 | 1.011x |
| 8192 | 256 | 4 | 40.799 | 19.199 | 21.605 | 0.889x |

## Follow-up work

1. Extend pruning and synthetic inputs to represent ragged mixed batches,
   separating maximum query length from total scheduled query tokens. Validate
   ceiling lookup and candidate math before comparing the affected workloads.
2. Benchmark those workloads independently to separate launch-coverage gains
   from the segmented-3D algorithm advantage. Context launch tuning does not
   add segmentation; a new packed-layout multi-query segmented path remains
   a separate algorithmic opportunity.
3. Preserve the native query/cache precision contract and establish a model
   quality gate before changing a default backend. A fully optimized serving
   comparison with separate TTFT/TPOT remains part of roadmap item 17.

## Artifacts and reproduction

All data is under
`/app/vllm-perf-rdna4-prefill-autotune-splitkv/results/offline-attention/`:

- `manifest.json`: source revisions and kernel/harness/native binary hashes.
- All six `round1-*` and three FP8 `round2-*` directories are complete.
  `round2-ours-bf16` has ten completed shapes; the other BF16 second-round
  variants were not run. Their `result.json`/`run.log` files retain raw timings,
  token IDs, settings, worker inventories, untimed GPU route profiles and
  tuning-cache SHA256/mtime snapshots before/after engine start.
- `comparison-run.log`: sequential driver log.
- `summary.json` and `comparison.csv`: complete per-shape aggregation.
- `smoke*` directories: untimed setup/route validation, excluded from results.

Run the same protocol:

```bash
cd /app/vllm-perf-rdna4-prefill-autotune-splitkv
./run-attention-offline-comparison.bash --probe
.venv/bin/python benchmarks/kernels/summarize_attention_offline.py \
    results/offline-attention --available-rounds
```

Run one variant:

```bash
RUN_TAG=recheck ./bench-attention-offline.bash ours fp8 --probe
```

Setting `--smoke --iterations 1` runs a bounded engine check. The context
startup tuner still obeys the full configured scheduler/model limits.
