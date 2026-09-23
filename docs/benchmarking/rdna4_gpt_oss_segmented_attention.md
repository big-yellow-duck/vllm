# RDNA4 GPT-OSS segmented attention: D64, sinks, and tuned performance

Date: 2026-09-23. This follows the
[DFlash2 segmented-attention work](rdna4_dflash2_segmented_attention_speedup.md).
GPT-OSS uses 64-wide heads, per-head attention sinks, and alternating
sliding-window and full-attention layers. The opt-in `ROCM_SEGMENTED_ATTN`
backend now accepts that combination.

## Implementation and routing

The Triton stages, reduction, workspace helpers, and runtime dispatcher live
in [`segmented_attention.py`](../../vllm/v1/attention/ops/segmented_attention.py).
Persistent startup tuning lives in
[`segmented_attention_tuning.py`](../../vllm/v1/attention/ops/segmented_attention_tuning.py).
The dispatcher loads the tuner when it needs a configuration, avoiding an
import cycle while keeping both modules focused on their runtime roles.

The [segmented prefill kernel](../../vllm/v1/attention/ops/segmented_attention.py)
loads one sink logit per query head and initializes one split's online softmax
with that logit and a zero value. Other splits remain unchanged. Their existing
log-sum-exp reduction then combines the sink with all real keys exactly once.
The sink is in natural-log units; the kernel converts it to the log2 units used
by its online softmax. Both one-stage and multi-split execution are covered by
the dense-reference tests.

The [backend](../../vllm/v1/attention/backends/rocm_segmented_attn.py),
[eligibility check](../../vllm/v1/attention/ops/segmented_attention.py), and
[startup tuner](../../vllm/v1/attention/ops/segmented_attention_tuning.py)
accept D64 and sink-bearing layers. A D64 one-query decode uses a 64-wide K
tile; the old 128-wide choice would execute zero K iterations. The default
router covers short-window decode, windowed prefill, FP8 full prefill, and long
full-prefix decode. The dispatcher runs every supported GPT-OSS shape through
segmented attention, including FP8 full attention. AITER appears only in the
benchmark as a comparison target.

The startup tuner stores separate records for sink and non-sink layers and
compiles the sink path during candidate screening. D64 now searches up to 32
launch configurations, including 16-, 32-, and 64-token sequence tiles, larger
query tiles, and more split choices for long contexts. Single-request D64
decode at 128K or longer uses a 256-split, one-warp default and searches
64–512 splits with one- and two-warp variants. Its scratch-budget check and
graph workspace reservation include the largest searched split count. D128
and D256 retain their 16-candidate search and workspace bound. D64 sink
candidates are scored with GPU graph
replays; the tuner builds the same interleaved KV layout as serving. For
sliding-window layers, it also uses the physical sequence length for the
synthetic block table, uses the full physical cache for single-request decode,
and raises the promotion threshold to 10% to avoid selecting noise from the
shortest kernels.

## Direct comparison on one R9700

The [reproducible benchmark](../../benchmarks/kernels/benchmark_rdna4_segmented_gpt_oss.py)
uses batch one, 16 query heads, two KV heads, D64, 64-token pages, and a
per-head sink. It loads identical KV values into each backend's native cache
layout. `--autotune` invokes the production sink-aware search for each shape,
installs its selected record in the dispatcher table, and times the routed
segmented path. Times are median GPU microseconds per complete attention call
across five graph replays, each containing five calls. KV updates and host
metadata are excluded. AITER uses its default unified-attention implementation
on this installation. Ratios are AITER time divided by segmented time; values
above 1 favor segmented.

| Queries | KV length | Mode | BF16 segmented / AITER µs (ratio) | FP8 segmented / AITER µs (ratio) |
| ---: | ---: | --- | ---: | ---: |
| 1 | 8K | Full | 12.73 / 13.87 (1.09×) | 13.95 / 13.02 (0.93×) |
| 8 | 8K | Full | 14.10 / 24.15 (1.71×) | 18.79 / 25.10 (1.34×) |
| 1 | 128K | Full | 47.64 / 65.98 (1.39×)¹ | 52.43 / 74.65 (1.42×)¹ |
| 8 | 128K | Full | 78.75 / 159.61 (2.03×) | 149.79 / 221.40 (1.48×) |
| 1 | 128K | Window 128 | 6.30 / 6.30 (1.00×) | 7.57 / 7.04 (0.93×) |
| 8 | 128K | Window 128 | 7.17 / 7.64 (1.07×) | 8.38 / 8.81 (1.05×) |
| 128 | 8K | Full | 68.09 / 149.96 (2.20×) | 118.86 / 213.09 (1.79×) |
| 128 | 128K | Full | 1018.36 / 2399.37 (2.36×) | 1660.91 / 3514.04 (2.12×) |
| 128 | 8K | Window 128 | 8.01 / 8.52 (1.06×) | 10.46 / 10.67 (1.02×) |
| 512 | 8K | Window 128 | 12.22 / 11.22 (0.92×) | 18.58 / 17.54 (0.94×) |
| 4096 | 8K | Window 128 | 47.20 / 45.13 (0.96×) | 90.84 / 83.10 (0.91×) |

¹ Targeted follow-up run with the same five-call, five-replay timing settings;
the other rows come from the original 22-case suite. The 512-query window case
loses about 6–9%; the 4096-query window case loses about 5–9%. The FP8
one-query 8K full and 128K window cases lose about 7%. The other cases are at
parity or faster. The largest observed per-row output error against AITER was below
0.005 in relative norm. Differences of a few percent at 5–15 µs are sensitive
to GPU clock and launch variation. These are attention-only measurements,
not GPT-OSS serving throughput or a full-model quality evaluation.

## Q=1, 128K follow-up

Before this change, the static D64 route used 64 splits and four warps. A
`rocprofv3` trace of the BF16 stage attributed 51.7% of sampled stall cycles
to the K-load source line and 40.1% to the V-load source line. The dominant
instructions there were `s_barrier_wait` and `s_wait_loadcnt`. The trace
analyzer misidentified gfx1201 as gfx942, so its occupancy estimate was not
used. Kernel statistics showed a minimum stage duration of 61.36 µs and a
minimum reduction duration of 1.84 µs.

A controlled graph-replay sweep tested 32–512 splits, one to four warps, and
32- or 64-token K tiles against the baseline output. Both BF16 and FP8 chose
256 splits, a 32-token tile, one warp, and one pipeline stage. This increases
parallel work from 128 to 512 stage programs for the two KV heads and cuts
each program's 128K span from roughly 2048 to 512 tokens. The updated
profile measured a 39.56 µs minimum stage duration and 2.28 µs reduction
duration: the reduction grew slightly, while the stage improved enough to
recover the total call.

| KV dtype | Previous segmented µs | Updated segmented µs | Updated AITER µs | Updated ratio |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 79.95 | 47.64 | 65.98 | 1.39× |
| FP8 | 91.89 | 52.43 | 74.65 | 1.42× |

The updated route cuts measured segmented latency by about 40% for BF16 and
43% for FP8 relative to the previous run. AITER timings varied between runs,
so the ratios use only the paired updated measurements. The production
autotuner retained the new default for both dtypes. Its standalone benchmark
now releases tuning scratch before timing, matching the production warmup's
cache cleanup. The long FP8 decode route also passed a 128K dense-reference
test with a per-row relative error below 0.01.

Correctness tests in
[`test_prefix_prefill.py`](../../tests/kernels/attention/test_prefix_prefill.py)
compare D64 and D128 sink outputs with a dense reference for BF16 and FP8 KV,
full and sliding-window attention, and forced multi-split reduction. They also
check the D64 backend route, sink-aware tuning identity, and FP8 full-attention
decode routing. The raw output is
`/tmp/rdna4_gptoss_segmented_tuned_suite_final.log` on the profiling host;
rerun the benchmark with `--suite --autotune --calls 5 --replays 5` to
regenerate the full suite. Rerun the updated Q=1 row with
`--query-len 1 --seq-len 131072 --window 0 --autotune --calls 5 --replays 5`
and add `--fp8` for FP8 KV. The earlier static suite used an AITER runtime
fallback for several FP8 rows and should not be used as a pure segmented
comparison.

A separate production warmup check used D64, FP8 KV, sinks, and a 128-token
window. It tuned eight reachable buckets in 22.7 seconds, returned a sink
config for the eight-query bucket, and kept the non-sink table separate. The
32-candidate search can add noticeable first-start latency for models with
many reachable buckets; persistent tuning records avoid paying it again for
the same model and hardware identity.
