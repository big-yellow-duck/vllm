# Triton short-prefill roofline prototype

Measured 2026-09-15 on physical GPU 0, Radeon AI PRO R9700, gfx1201.
The one-hour attempt began at 14:26:57 UTC; the revised shape first exceeded
80% at approximately 14:52 UTC. This report records the validated result.

## Result and revision of the target

**The prototype reaches 90.7% of the ideal roof on BF16 Q2/C65536**, including
stage and reduction. It takes **231.18 us**, below the **262.20 us** 80% limit.
AITER's default takes 254.22 us. Forcing AITER to 32 segments takes 228.22 us:
our result is **within 1.3% of tuned AITER**, rather than a demonstrated win
against its best configuration in the bounded search.

The initial target was **Q32/C8192**, with an ideal floor of 26.93 us and an
80% limit of 33.664 us. It finishes at **78.96 us / 34.1%**, versus 152.40 us
AITER and 493.59 us the selected, tuned ROCM_ATTN context path. That is a
6.25x improvement over the current path, but **the original 80% target was
not reached**. The target was explicitly revised during the attempt to a
smaller query chunk over a longer prefix, where KV traffic dominates and
fixed costs are better amortized. This does not establish 80% for all short
prefill workloads or for the original shape.

The cache-eviction method also changed, for the reason documented below.
The 90.7% claim applies to read-only eviction. It does not apply to the old
write-eviction protocol, whose result is retained in the comparison table.

## Matched complete-call measurements

Both sides use BF16 Q/K/V/output, HQ=12, HKV=2, D=256, page=784 and the same
effective values. ROCM_ATTN uses packed K/head-major V; AITER uses NHD, with
interleaved K/V backing on both sides. Data preparation and compilation are
outside timing. Measurements include every attention kernel and reduction.
Each number is the median of five round medians, each containing 30 HIP-event
samples of a captured graph. Backend order reverses on alternating rounds.
Captured input/output/workspace lifetimes are retained and outputs are checked
after every timing round.

| Shape | Implementation | Read eviction us | Write eviction us | Reuse us | Ideal roof, read eviction |
| --- | --- | ---: | ---: | ---: | ---: |
| Q2/C65536 | Triton prototype | 231.18 | 483.03 | 229.32 | 90.73% |
| Q2/C65536 | AITER default | 254.22 | 381.98 | 253.04 | 82.51% |
| Q2/C65536 | AITER forced 32 segments | 228.22 | 353.48 | 233.12 | 91.91% |
| Q2/C65536 | ROCM_ATTN selected | 20401.53 | 20351.88 | 20413.78 | 1.03% |
| Q2/C65536 | ROCM_ATTN isolated launch tuning | 2785.99 | 2979.29 | 2786.35 | 7.53% |
| Q32/C8192 | Triton prototype | 78.96 | 102.46 | 68.02 | 34.11% |
| Q32/C8192 | AITER default | 152.40 | 164.34 | 147.26 | 17.67% |
| Q32/C8192 | ROCM_ATTN selected | 493.59 | 534.37 | 441.52 | 5.46% |

The Q2/C65536 selected ROCM_ATTN path misses the saved tuning table and uses
its default launch. A separate 12-configuration launch search finds
BM16/BN64/four warps/unroll=1; the repeated comparison above includes it.
The kernel improvement is therefore not solely a comparison against the
20 ms tuning miss. These overrides are isolated experiments; no saved tuning
table or production route was changed. The Q32 selected configuration is a
real cache hit: BM32/BN32/eight warps.

The AITER check searches 16/32/64/128 segments and KV tiles 16/32. Its best
candidate is tile16/32 segments, then measured again in the five-round
comparison. AITER's Q32 row is its default selected launch, not an exhaustive
AITER tuning result for that shape.

## Roofline and eviction correction

Keep the original advertised ceilings: **640 GB/s GDDR6** and **191 TFLOP/s
dense half-precision matrix throughput**. Sources:
[AMD R9700 specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
and [AMD matrix instruction rates](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/).

```text
pairs = Q*C + Q*(Q+1)/2
useful_FLOPs = 4 * HQ * D * pairs
minimum_bytes = Q*HQ*D*4 + 2*(C+Q)*HKV*D*2
ideal_time = max(useful_FLOPs / 191e12, minimum_bytes / 640e9)
roof_fraction = ideal_time / measured_complete_call_time
```

For Q2/C65536: 1,610,649,600 useful FLOPs, 134,246,400 minimum bytes,
11.998 FLOP/byte, and a 209.76 us ideal floor. Dividing these algorithmic
bytes by 231.18 us gives 580.69 GB/s. This is an algorithmic effective rate,
not a memory-controller counter measurement. The useful-work definition
excludes padding, softmax operations, repeated loads, and workspace traffic;
none were added to the numerator to improve the percentage.

The earlier study evicted caches by **writing 256 MiB immediately before
attention**. A 16 MiB read-and-reduce calibration reaches only 255–257 GB/s
after that operation, versus approximately 470 GB/s after **reading** a
256 MiB eviction buffer. Five alternating rounds reproduce the difference.
Dirty-cache writeback interference is a plausible explanation; the available
counters do not isolate it, so this is not presented as a proven causal
counter attribution.

The new read-eviction kernel loads the entire buffer and writes only a small
checksum array. Eviction remains outside the timed region. The old write
protocol is preserved because it represents a materially different condition:
Q2 is **483 us** there, worse than AITER's **382 us** default and **353 us**
32-segment result. Thus this experiment does not establish a universal cache
policy win. Reuse is reported separately and is not reinterpreted as GDDR6
utilization. The Q2 KV working set is approximately 128 MiB.

![Prototype roofline](/app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915/prototype-roofline.svg)

## Implementation and what the experiments support

The accepted implementation is entirely **Triton `tl.dot` kernels** with Python
allocation/launch code. It does not invoke AITER, FlyDSL or a handwritten HIP
attention kernel. The compiler emits RDNA4 WMMA instructions.

- Flatten query positions and GQA heads within each KV head. Q2 has 12 useful
  rows, padded to a matrix tile of 16. This shares K/V across all six heads.
- Split the 65,536-token prefix into 32 segments of 2,048 tokens. KV tile32,
  D256 QK reduction, four warps and one stage launch 64 stage workgroups.
- Read packed K and head-major V directly, including their interleaved outer
  strides. The final split consumes fresh dense current K/V with a causal
  mask. Cached current values need not match those fresh tensors.
- Use stable online softmax with FP32 accumulation. Store normalized FP32
  partial outputs and base-2 log-sum-exp values, then merge them in a second
  kernel. Q2 workspace is 789,504 bytes; Q32 workspace is 3,158,016 bytes.
- Q32 uses BM32/BN64/BK64, eight splits, four warps and two stages. Its 96
  stage workgroups improve parallelism relative to the selected context
  kernel's 12 groups. Smaller QK reduction tiles reduce register pressure.

The retained searches also reject output-dimension splitting, a materialized
QK/softmax/PV pipeline, larger query tiles, physical-page segmentation, BF16
partial storage, and the tested explicit Gluon layouts as replacements for
the selected configurations. Some improved intermediate baselines but did
not beat the final choice. More launch sweeping alone did not reach 80% on Q32.

## HSA / ATT verification

Separate rocprofv3 captures collect HSA calls, kernel dispatches, code objects
and ATT samples. Profiled runtimes are not used in the performance table.
Compiled code objects are matched to the Triton cache by SHA256.

| Q2 kernel | Launched workgroups | Threads/group | Dynamic LDS bytes | Compiler used VGPRs | HSA reported VGPRs | Scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prototype stage | 64 | 128 | 16384 | 188 | 248 | 0 |
| Prototype merge | 24 | 128 | 2048 | 86 | 88 | 0 |
| AITER default stage | 256 | 128 | 8192 | 241 | 248 | 0 |
| AITER default merge | 24 | 128 | 2048 | 241 | 248 | 0 |

AITER's stage grid contains upper-bound padding groups that can return early;
launched group counts are not equivalent to useful work counts. The prototype
stage's descriptor reserves 241 VGPRs despite compiler metadata reporting 188
used, and HSA reports the rounded 248 allocation. Therefore the lower compiler
number is **not evidence of an occupancy gain**. Both stage kernels are
spill-free. The reduction has a real allocation reduction, as well as half as
many segments to combine.

The selected ATT wave samples are dominated by global-load and combined
load/LDS waits, consistent with the bandwidth-oriented timing result. Sampled
stall fractions do not rank the two kernels' speed. No CDNA occupancy estimate
is applied to this RDNA4 GPU. The earlier `FETCH_SIZE` counter was unusable,
so this remains an ideal useful-work roofline backed by timing and HSA/ATT,
not a measured multi-level memory roofline.

## Validation and remaining scope

Both selected shapes pass full-output FP32 reference checks over five held-out
cases, each checked again after graph replay with changed inputs. These cover
normal values at scales 0.25/1/3, zero queries, shuffled physical pages,
independently changed fresh K/V, and NaNs deliberately placed in cached current
positions and padding. The worst per-query/per-head relative L2 error is
0.259%, below the 1% diagnostic gate. Partial buffers remain FP32.

The public preparation API is additionally tested with identity and fragmented
page tables across five seeds and 128/256/512 MiB read-eviction buffers.
All 15 cases remain between 231.40 and 233.04 us
(approximately 90% of the same fixed roof), and pass the full reference check.

This supports adding a narrowly routed segmented Triton short-prefill path
and separately fixing tuning coverage. It does not establish an 80% Q32
kernel or production readiness. The public factory accepts only the two
validated single-request BF16 shapes and page784. Ragged routing, FP8, windows,
sinks, other head dimensions, workspace integration, engine latency and model
quality still require separate work. No production backend or engine code was
modified in this attempt.

## Code and reproduction

- [Kernel and prepared-call API](/app/vllm-perf-rdna4-prefill-autotune-splitkv/benchmarks/kernels/rdna4_prefill_prototype/kernel.py)
- [Usage and reproduction](/app/vllm-perf-rdna4-prefill-autotune-splitkv/benchmarks/kernels/rdna4_prefill_prototype/README.md)
- [Final Q2 data](/app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915/final-q2.json)
- [Final Q32 data](/app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915/final-q32.json)
- [Fragmentation/eviction checks](/app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915/robustness.json)
- [Profiles, code objects, searches and plot](/app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915)

The requested venv is activated by `env.bash`. Source hashes, package versions,
revisions, protocol and commands are recorded in `manifest.json`. Run GPU jobs
sequentially. Formatting and repository Ruff checks pass for the prototype.
