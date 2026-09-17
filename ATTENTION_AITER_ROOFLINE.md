# AITER attention roofline and ROCM_ATTN prefill feasibility

Measured 2026-09-15 on physical GPU 0, Radeon AI PRO R9700 (`gfx1201`).
This is a rank-local Qwen3.8 TP2 kernel study: HQ=12, HKV=2, D=256.
It is not a new TP2 engine benchmark.

## Decision

Extending the work is viable, but it should have three independently measured
parts: fix ragged tuning coverage, add segmented multi-query attention for
short chunks over long prefixes, and improve the existing decode fallback.
Keep the tuned context kernel for long prefill. It already beats AITER on
some matched workloads.

The largest long-input engine gap is not evidence that every prefill needs
SplitKV. A matched ragged probe reproduces a prefill tuning miss. Reusing a
saved single-request configuration, solely as an isolated experiment, takes
the BF16 ROCM_ATTN path from **29.13 ms to 7.69 ms**, versus **9.25 ms AITER**.
That is a 3.79x improvement without changing the attention algorithm.
It establishes the value of fixing coverage; it does not establish that a
single-request configuration is a safe universal replacement for ragged tuning.

There is separate evidence for segmentation: **Q32/C8192 BF16 takes 174 us
with AITER versus 546 us with the already tuned context kernel**. Conversely,
**Q2048/C0 takes 508 us with the tuned context kernel versus 653 us AITER**.
The implementation should select among these regimes.

## Scope and controls

Sources are the current combined vLLM checkout and TPS AITER
`f07170b53a178d1de20001b130bb40b44127be12`. Exact source revisions, protocol,
raw results, scripts, profiles and the roofline plot are in
[/app/tps-rdna4-qwen38-tp2/results/aiter-roofline-20260915](/app/tps-rdna4-qwen38-tp2/results/aiter-roofline-20260915).

The primary matrix has 34 backend cases over 14 workload/dtype combinations:
three decode shapes with AITER, selected ROCM_ATTN and forced Triton; four
prefill/mixed shapes with AITER and selected ROCM_ATTN; BF16 and FP8 KV.

- Both sides receive identical BF16 queries and effective KV values, with
  BF16 outputs. FP8 E4M3FN KV uses non-unit K=0.125 and V=0.25 scales.
- ROCM_ATTN receives packed K/head-major V, and AITER receives NHD K/V.
  Both use interleaved K/V backing strides, with the same physical page size:
  784 BF16 or 1568 FP8. Layout preparation is outside timing.
- For FP8 prefill, the dense current chunk on ROCM_ATTN is reconstructed from
  the quantized cache. This controls effective KV values, but differs from
  the engine's usual fresh BF16 current chunk.
- Saved startup tables are loaded only when their identity matches. No
  runtime tuning takes place. The recorded configuration distinguishes hits,
  misses and isolated surrogate experiments.
- Timings use HIP events around a captured graph replay without a profiler.
  Each mode has three rounds of 20 samples; report the median of round medians.
  “Cold” means a 256 MiB eviction buffer is written before the timing interval.
  “Reuse” repeatedly replays the same working set. Eviction is a cache-thrashing
  protocol, not a counter-verified guarantee that every read reaches GDDR6.
- All matrix cases passed the FP32 reference check and a second check after
  changing Q and replaying the graph. The reference checks every decode token
  and up to five selected positions per prefill request. The largest initial
  sampled-token relative L2 error was **0.227%**, below the 1% probe threshold.
  This is a kernel diagnostic, not a full correctness suite or model evaluation.

The offline engine comparison remains the authority for whole-generation
performance. Its AITER FP8 path quantizes Q and uses different native page
sizes. The matched BF16-query numbers below must not be presented as that
engine's FP8 performance.

## Matched decode results

All times are microseconds and include the partial-result reduction where used.

| KV | Workload | AITER cold | Ours cold | Triton cold | AITER reuse | Ours reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | B1/S8192 | 81.82 | 84.18 | 119.92 | 26.12 | 46.80 |
| BF16 | B4/S8192 | 266.74 | 255.40 | 409.50 | 82.54 | 96.98 |
| FP8 | B1/S8192 | 181.02 | 63.44 | 732.21 | 133.18 | 47.32 |
| FP8 | B4/S8192 | 423.77 | 147.18 | 1996.94 | 382.42 | 95.44 |

At B1/S512, BF16 still falls back to unsplit Triton: 89.36 us versus
37.06 us AITER. FP8 activates FlyDSL and takes 26.16 us versus 117.12 us
for matched BF16-query AITER. Short BF16 decode and the Triton fallback are
optimization opportunities in their own right.

## Matched prefill and mixed results

Q is the number of current query tokens per request; C is the cached prefix.
Times are microseconds under the eviction protocol.

| Workload | BF16 AITER | BF16 ours | FP8 AITER | FP8 ours |
| --- | ---: | ---: | ---: | ---: |
| Q32/C8192 | 174.12 | 545.99 | 880.43 | 1563.90 |
| Q128/C8192 | 581.29 | 578.45 | 3072.17 | 1621.10 |
| Q2048/C0 | 653.43 | 507.67 | 1249.61 | 500.55 |
| Ragged long | 9254.02 | 29130.08 | 17504.23 | 26711.32 |

The ragged case has query lengths `[1,1,1,8188]` and prefix lengths
`[8194,8194,8194,7]`: 8191 scheduled tokens, max query 8188, and max sequence
8195. It is representative of the documented missing batch/max-query bucket,
not a replay of an exact captured engine step. Both paths compute all four
requests, including the three decode rows.

AITER selects 32 segments for Q32/C8192 and eight segments for Q128/C8192.
It selects 2D for Q2048/C0 and the ragged long batch. Our saved Q32 BF16
configuration is BM32/BN32/eight warps; Q128 uses BM32/BN32/four warps;
Q2048 uses BM64/BN64/four warps. The ragged case misses the table and uses
the original BM128/BN64/four-warp launch.

The BF16 surrogate experiment uses a saved B1 ceiling bucket's
BM128/BN32/eight-warp configuration on the same ragged inputs. It passes
both reference checks and reduces latency to 7694.19 us. No production
router, kernel or tuning table was changed for this experiment.

The corresponding FP8 surrogate takes **7679.81 us**, down from 26711.32 us.
The surrogate's saved launch is an experiment on these specific inputs;
production-quality ragged tuning still needs representative workload generation
and validation across the scheduler's supported shapes.

An additional native-style FP8-query probe, with Q scale 0.125, exceeded this
study's strict 1% reference threshold (2.047% sampled-token L2). It is retained
as a rejected numerical probe and has no accepted timing result. This does
not establish that the native backend is incorrect or assess model quality;
it reinforces why the BF16-query comparison cannot be substituted for a
precision-matched evaluation of native engine behavior.

## Roofline model

Use AMD's published **191 TFLOP/s dense half-precision matrix ceiling** and
**640 GB/s GDDR6 bandwidth**, with an ideal ridge at **298.44 FLOP/byte**.
AMD also documents the RDNA4 BF16 dense matrix rate alongside FP16.
Sources: [R9700 specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
and [AMD GPUOpen matrix rates](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/).
These are advertised ideal ceilings, not measured sustained microbenchmark ceilings.

For request i with current query length Q_i and prefix C_i:

```text
causal_pairs = sum_i(Q_i*C_i + Q_i*(Q_i+1)/2)
useful_FLOPs = 4 * HQ * D * causal_pairs
minimum_bytes = sum_i(Q_i)*HQ*D*(query_bytes + output_bytes)
              + 2*sum_i(C_i+Q_i)*HKV*D*KV_bytes
intensity = useful_FLOPs / minimum_bytes
ideal_time = max(useful_FLOPs / compute_ceiling, minimum_bytes / bandwidth)
```

QK and PV count an FMA as two operations. This ideal model reads K/V once
per KV head, Q once and O once. It excludes scalar softmax, masked matrix
overcompute, page tables, repeated reads, spills and split-buffer traffic.
It measures useful-attention efficiency, not direct matrix-unit utilization.

| BF16-query workload | FLOP/B | Ideal floor us | AITER TFLOP/s | Ours TFLOP/s | AITER ideal-roof % | Ours ideal-roof % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1/S8192 decode, BF16 KV | 6.00 | 26.23 | 1.23 | 1.20 | 32.06 | 31.16 |
| B1/S8192 decode, FP8 KV | 11.98 | 13.13 | 0.56 | 1.59 | 7.25 | 20.69 |
| Q32/C8192, BF16 KV | 187.27 | 26.93 | 18.54 | 5.91 | 15.47 | 4.93 |
| Q2048/C0, BF16 KV | 878.14 | 134.99 | 39.46 | 50.79 | 20.66 | 26.59 |

Decode lies on the ideal bandwidth slope. Long prefill lies on the ideal
compute plateau. Neither classification proves the implementation's actual
bottleneck: a low-intensity kernel can still lose time to barriers, address
arithmetic, spills or insufficient work distribution.

Cache-reuse timings require a cache-level roof and cannot be interpreted as
GDDR6 utilization by dividing minimum bytes by runtime. Similarly, “useful
GB/s” in the CSV is algorithmic bytes divided by time, not a measured memory
controller counter. The plot intentionally uses only the eviction timings.

![Matched attention roofline](/app/tps-rdna4-qwen38-tp2/results/aiter-roofline-20260915/attention-roofline.svg)

## HSA, generated code and ATT evidence

The captures include HSA API calls, kernel dispatches, allocation traces,
code objects and ATT samples. The table below describes the B1/S8192 stage
kernels. LDS combines static FlyDSL allocation with Triton's launch-time
shared-memory metadata. The captured Triton code objects were matched to the
compiler cache by SHA256; the CSV alone reports zero for their static LDS
and would incorrectly suggest that they use no shared memory.

| Stage | KV tile | Segments | Workgroups | Threads/WG | LDS bytes/WG | Reported VGPRs | Scratch bytes/work item |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AITER BF16 | 16 | 64 | 128 | 128 | 8192 | 248 | 0 |
| FlyDSL BF16 | 32 | 16 | 32 | 512 | 41472 | 168 | 0 |
| Triton fallback BF16 | 32 | 16 | 32 | 128 | 16384 | 256 | 216 |
| AITER FP8 KV / BF16 Q | 64 | 128 | 256 | 64 | 33024 | 256 | 956 |
| FlyDSL FP8 KV / BF16 Q | 32 | 16 | 32 | 512 | 41472 | 176 | 0 |

`SQ_WAVES` confirms **512 stage waves for both BF16 AITER and FlyDSL**.
Their organization differs: 128 four-wave groups versus 32 sixteen-wave
groups. This is not simply “AITER launches more waves.” The FlyDSL stage
uses more LDS per group and explicitly assigns only two waves to QK before
redistributing work for PV. Workgroup synchronization and data redistribution
are plausible optimization targets.

The selected single-CU/SIMD ATT samples show the following shares of sampled
stall cycles. Categories are decoded separately to avoid double-counting the
combined `s_wait_loadcnt_dscnt` instruction as two independent waits.

| Stage | Sampled stalled fraction | Main sampled stall categories |
| --- | ---: | --- |
| AITER BF16 decode | 65.9% | combined load/LDS wait 47.0%; load-only wait 10.3%; LDS-only wait 8.5%; `s_wait_kmcnt` 18.9% |
| FlyDSL BF16 decode | 55.2% | LDS/SMEM `s_waitcnt lgkmcnt(0)` 42.6%; load wait 39.7% |
| Triton BF16 fallback | 46.8% | load-only wait 50.1%; buffer loads 15.8%; LDS stores 10.2%; combined load/LDS wait 10.0% |
| AITER FP8 KV / BF16 Q decode | 77.8% | load-only wait 80.6%; LDS-only wait 12.5% |
| FlyDSL FP8 KV / BF16 Q decode | 50.7% | LDS/SMEM wait 46.9%; load wait 29.8% |

Fractions from different wave samples do not rank kernel speed. A slower
kernel can have a lower stalled fraction because it issues more instructions
or distributes work differently. The unprofiled timings establish speed;
the samples locate optimization candidates.

Concrete findings:

- FlyDSL's main BF16 source hotspot is the LDS/SMEM wait at the end-of-tile
  barrier ([stage helper:469](/app/vllm-perf-rdna4-prefill-autotune-splitkv/vllm/v1/attention/ops/flydsl_kernels/rdna4_splitkv_gqa16_d128.py:469)),
  followed by the immediate wait after K loads at line 286. A broader recapture
  reproduces this ordering. Explore a smaller workgroup, less LDS exchange,
  and overlap of next-tile loads with current computation. Required waits
  cannot simply be removed before LDS reuse.
- AITER's matched FP8/BF16-query stage has a **956-byte private frame** and
  its 128-segment reduction has **1208 bytes**. The stage's V conversion/load
  region at unified-kernel line 744 dominates the broader recapture. The
  reduction sample is 95.0% stalled, with 88.6% of its stalls on load-only
  waits. This explains a concrete weakness of this mixed-precision launch;
  do not copy its TILE64/two-wave/128-segment policy into our fallback.
- The patched BF16 AITER stage and reduction have no scratch spills. Our
  FlyDSL stage and compact reducer are also spill-free. Our Triton BF16
  fallback has a 216-byte private frame, so fallback launch/memory layout
  deserves attention independently of FlyDSL.
- Q32/C8192 launches only **12 workgroups** in our tuned context kernel,
  despite being spill-free. AITER's segmented path launches **1088**, including
  upper-bound padding groups that return early. This supplies a concrete
  parallelism explanation for the short-query prefill opportunity.
- Long Q2048/C0 prefill still spills on both sides: 432 bytes/work item for
  AITER and 680 for ours. Ours nevertheless wins. Its captured trace reported
  partial data loss, so source stall totals there are diagnostic only.

### Segment-count experiment

Forcing AITER's BF16 B1/S8192 stage and reduction to 16 segments, while
retaining its TILE16/four-wave configuration, passes the reference checks:

| AITER policy | Cold us | Reuse us |
| --- | ---: | ---: |
| Default 64 segments | 81.82 | 26.12 |
| Isolated 16-segment experiment | 73.64 | 36.44 |

Fewer segments improve this eviction case but worsen cache reuse. The
winning split policy depends on both work distribution and working-set
residency; it should be tested under both conditions. Our current FlyDSL
router and reducer only support 2/4/8/16 splits, so adopting 64 also requires
an implementation/validation change, not just a heuristic edit.

AITER obtains a physical CU count through `chip_info.get_cu_num`; the local
Torch properties report 32 multiprocessors for this 64-CU GPU. Our heuristic
uses the latter. Account for the CU/WGP unit and workgroup size explicitly
when comparing their occupancy targets; do not blindly multiply a count.

### Profiling limits

`FETCH_SIZE` returned zero for both kernels and the workload's memory
operations on this setup; it is not usable as a GDDR6 byte measurement.
`SQ_WAVES` returned meaningful counts. The report therefore provides an
ideal algorithmic roofline backed by measured latency and resource/stall
evidence, not a measured multi-level memory roofline.

ATT samples one selected CU per shader engine. Some small-grid kernels
produced no sampled waves, including the compact FlyDSL reducer and Q32
context kernel even after a broader recapture. Their dispatch/resource
records remain valid, but no stall percentages are claimed for them.
The shipped hotspot analyzer assumes CDNA for these traces and mislabels
them `gfx942`; its occupancy estimate is not used. `analyze.py` summarizes
the RDNA4 opcodes directly, while the shipped analyzer supplies source
attribution. See `att-summary.json`, `trace-resources.json`,
`compiled-resources.json` and `pmc-summary.json` for the underlying evidence.

## Implementation plan supported by the evidence

1. **Fix ragged tuning coverage first.** Represent max query length separately
   from total scheduled tokens and request count. Generate nonuniform query
   lengths and prefix lengths within the scheduler token budget, preserve KV
   dtype and physical-page identity, and validate ceiling lookup. A B1
   surrogate is an experiment, not a replacement for that workload model.
2. **Add a packed-layout segmented multi-query Triton path.** Start with
   causal short chunks, such as Q2–Q32 over long prefixes. Tile query tokens
   and GQA heads together, split the key axis, and merge partial softmax
   statistics. Preserve the existing dense-current-chunk BF16 behavior for
   FP8 KV. Unsupported features continue through the existing context path.
3. **Extend FlyDSL after establishing that baseline.** Reuse packed-cache
   addressing and matrix fragments, but redesign query indexing, per-token
   causal masks and the reduction. The current stage and reducer both
   explicitly require `query_len == 1`; changing only the dispatch guard is
   incorrect. Scratch must be indexed by query token, head and split.
4. **Keep split workspace bounded.** D256/HQ12/Q32/S32 already requires
   12 MiB just for FP32 partial output. Q8192/S16 would require 1.5 GiB.
   Restrict splitting to regimes that gain enough parallelism to pay for the
   extra launch, partial-result traffic and storage. Long prefill should keep
   a direct tiled path unless measurements establish otherwise.
5. **Measure a decode/fallback improvement separately.** Tune workgroup size,
   segment count and reduction shape together. Preserve a direct short-context
   path. Do not assume AITER's numeric segment settings transfer to a
   16-wave FlyDSL workgroup or to the current compact serial reducer.

For mixed batches, each output token must have one owner. Add disjoint query
length routing or a query-block map so the old prefill kernel, new short-query
kernel and decode kernel do not write the same output. Use capture-safe
metadata/workspaces without GPU-to-CPU synchronization in the attention call.

Validation before production promotion should cover ragged/empty tails,
page-crossing tiles, causal positions across prefix/current boundaries,
non-unit FP8 scales, graph replay with changed inputs, and unsupported-feature
fallbacks. Reduction must mask empty segments and include any sink only once.
Then repeat the affected offline engine shapes and model evaluation; these
kernel measurements do not establish whole-engine or model-quality parity.

## Reproduction

```bash
cd /app/tps-rdna4-qwen38-tp2/results/aiter-roofline-20260915
./run_matrix.bash
./run_experiments.bash
./run_profiles.bash
./run_recapture.bash
/app/vllm-perf-rdna4-prefill-autotune-splitkv/.venv/bin/python analyze.py
```

Run GPU jobs sequentially. `env.bash` pins source paths and isolated kernel
caches. `manifest.json` records the protocol; individual JSON files record
reference errors, selected launch settings and round medians. `roofline.csv`
and `roofline-summary.json` retain every measured backend case.
