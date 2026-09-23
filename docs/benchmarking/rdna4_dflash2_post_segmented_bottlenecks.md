# RDNA4 DFlash2: bottlenecks after segmented attention

Date: 2026-09-23. Follow-up measurements after the
[segmented-attention speedup](rdna4_dflash2_segmented_attention_speedup.md).
The workload is batch one, TP4, FP8 KV, with a 127,998-token prompt
unless a section specifies another shape. The pre-change diagnosis is in the
[baseline note](rdna4_dflash2_spec_decode_investigation.md).

The target's GPU verification range is 27.11 ms of the 32.59 ms patched 128K
step. Its `packed_kernel_0` launches account for 11.49 ms and are the largest
measured kernel group; [the speedup note](rdna4_dflash2_segmented_attention_speedup.md#next-targets)
tracks the next targets. The sections below separate live acceptance changes,
target FP8 work, draft convolution, and target preparation.

## Live request: acceptance versus round speed

The user supplied two steady ten-second intervals from one real request on
2026-09-23. Each DFlash round proposes seven tokens. The counts therefore
recover the number of rounds, and accepted drafts plus one target token per
round reproduce the generation throughput exactly:

| Interval ending | Drafted | Rounds | Accepted drafts | Mean output/round | Final output/s | Rounds/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 07:03:36 | 2611 | 373 | 718 | 2.92 | 109.1 | 37.3 |
| 07:03:46 | 2604 | 372 | 1019 | 3.74 | 139.1 | 37.2 |

The round rate stayed nearly constant. The 30 output-token/s change came from
accepting more drafts, not from a faster draft or target pass. The last two
draft positions had unconditional acceptance probabilities of 0.102 and 0.070
in the first interval, versus 0.239 and 0.194 in the second. These serving
metrics are not a GPU profile and cannot be directly compared with the static
offline 32.59 ms round; this live run uses the tuned path and a different prompt.

## Target FP8 projection attribution

The earlier 127,998-token PyTorch trace contains 256 `packed_kernel_0.kd`
launches per target verification step, totaling 11.49 ms. The profiler also
attributes the kernel family to `vllm::rdna4_fp8_block_scaled_mm`. The
[RDNA4 FP8 operator](../../vllm/model_executor/kernels/linear/scaled_mm/rdna4.py)
routes small decode matrices to the FlyDSL
[`packed_kernel`](../../vllm/model_executor/kernels/linear/scaled_mm/flydsl_kernels/rdna4_fp8_blockscale_small_m.py).
The 64-layer model has four repeating packed launches per layer. The projection
labels below follow their order and the
[Qwen3.5 layer](../../vllm/model_executor/models/qwen3_5.py),
[GDN](../../vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py), and
[MLP](../../vllm/model_executor/models/qwen2_moe.py) code; individual replayed
GPU kernels do not carry layer or projection names.

| Projection inferred from launch order | Median launch, linear layers | Median launch, full-attention layers | Approximate total per target step |
| --- | ---: | ---: | ---: |
| Attention input projection | 38.92 µs × 48 | 35.28 µs × 16 | 2.43 ms |
| Attention output projection | 22.40 µs × 48 | 16.92 µs × 16 | 1.35 ms |
| MLP gate/up projection | 75.72 µs × 48 | 74.16 µs × 16 | 4.82 ms |
| MLP down projection | 47.56 µs × 48 | 38.48 µs × 16 | 2.90 ms |
| **All packed projections** | | | **11.49 ms measured** |

These medians come from 16 steady steps across four TP ranks. MLP gate/up and
down together account for approximately **7.72 ms**, or 67% of packed-kernel
time and 24% of the 32.59 ms static round. For TP4, the gate/up FP8 weight
matrix is 8704 × 5120, about 44.6 MB per rank; the down matrix is 5120 × 4352,
about 22.3 MB. Dividing those bytes by the 75.72 and 47.56 µs medians gives
roughly 589 and 469 GB/s, respectively, if each weight is fetched once. This
is a traffic estimate, not a measured DRAM counter: cache hits, repeated
loads, and launch cost can change the interpretation. The direct measurement
below tests the weight-traffic hypothesis.

### Direct MLP kernel measurement and roofline

On one Radeon AI PRO R9700 (gfx1201), a CUDA-graph benchmark called the
checkout's `rdna4_fp8_block_scaled_mm` at the exact TP4 target shapes, with
`M=8` speculative verification rows. Each graph had 16 GEMMs. The **hot**
variant reused one weight tensor for all calls; the **rotated** variant used
16 distinct weight tensors. Timings are the median per call over 21 graph
replays, without rocprof instrumentation:

| Projection (M,N,K) | FP8 weight | Hot | Rotated | Logical weight-read rate |
| --- | ---: | ---: | ---: | ---: |
| Gate/up (8,8704,5120) | 44.564 MB | 27.190 µs | **74.008 µs** | 602.2 GB/s |
| Down (8,5120,4352) | 22.282 MB | 15.480 µs | **39.180 µs** | 568.7 GB/s |

The repeatable command is
`PYTHONPATH="$PWD" uv run --no-project -- /opt/venv/bin/python benchmarks/kernels/benchmark_rdna4_fp8_mlp_roofline.py`
from the checkout root.

The graph's distinct gate/up weights total 713 MB and its distinct down
weights total 356 MB. Both exceed the R9700's 64 MB Infinity Cache. A single
weight of either shape fits, explaining why a repeated-weight microbenchmark
looks much faster than the 128K target trace. The rotated gate/up result
matches the target's 74–76 µs launches; the rotated down result matches its
full-attention layers' 38.48 µs, while GDN layers still take 47.56 µs. That
last gap needs a same-context trace before attributing it to the GEMM.

AMD specifies **640 GB/s peak memory bandwidth** and **383 TFLOP/s peak FP8
matrix throughput** for this card ([product specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)).
Each shape performs `2MNK` operations. Counting only one FP8 read per weight
gives 16 FLOP/byte for both: 0.713 GFLOP / 44.564 MB for gate/up and
0.357 GFLOP / 22.282 MB for down. This puts both on the memory side of the
simple roofline (ridge point: 383,000/640 = 598 FLOP/byte):

| Projection | Weight-only bandwidth floor | FP8 compute floor | Rotated time / bandwidth floor | Modeled headroom |
| --- | ---: | ---: | ---: | ---: |
| Gate/up | 69.632 µs | 1.862 µs | 1.063× | 4.376 µs |
| Down | 34.816 µs | 0.931 µs | 1.125× | 4.364 µs |

The observed logical weight-read rates are 94.1% and 88.9% of the published
peak. They are **not measured DRAM bandwidth**: the model omits scales,
activations, outputs, cache hits, redundant loads, and contention. The roof
assumes one DRAM read per weight and peak sustained bandwidth, so it is a
screening estimate rather than a rigorous latency lower bound. On that model,
eliminating *all* remaining time above the roof in both MLP projections would
save about `64 × (4.376 + 4.364) µs = 0.56 ms`, or 1.7% of the 32.59 ms
static step. Better reuse across steps or fewer bytes would change the model.

An `rocprofv3` Advanced Thread Trace of both rotated shapes puts the largest
sampled stall at `s_wait_loadcnt 0x17`, mapped from FP8 fragment loads to the
K-block loop in
[`rdna4_fp8_blockscale_small_m.py`](../../vllm/model_executor/kernels/linear/scaled_mm/flydsl_kernels/rdna4_fp8_blockscale_small_m.py).
That supports a load-wait bottleneck qualitatively. The trace reported lost
data, and the available hotspot analyzer misidentified gfx1201 as gfx942, so
its exact stall percentages and occupancy estimates are not used. Attempted
`GL2C_*` cache/DRAM counters all returned zero even though an `SQ_WAVES`
sanity counter was nonzero; physical DRAM traffic remains unmeasured.

The path is `Qwen2MoeMLP.gate_up_proj` / `.down_proj` →
`RDNA4Fp8BlockScaledMMKernel.apply_block_scaled_mm` →
`rdna4_fp8_block_scaled_mm` → `_run_small_m` → `packed_kernel`.
For these `M=8`, `N>3072` shapes, `select_kernel_config` chooses
`KernelConfig(64, 1, 64, rotate_k=2, load_batch=8)`: two wave32s per
64 output columns, with eight K fragments loaded together before WMMA work.

The target range contains 1,752 kernels in a representative step. Its 5.85 ms
of inter-kernel spacing consists of small gaps; none exceeded 10 µs in that
step. The 3.7–5.0 ms host metadata scope described below has no matching long
GPU idle interval in this trace. Metadata work may still matter for concurrency,
but its host duration cannot be subtracted directly from round latency.

For the last live interval, reducing the draft length from seven to six would
lose the seventh position's 0.194 expected output tokens per round if earlier
acceptance stayed fixed. That is 5.2% of the observed 3.74 output/round, so a
six-token proposal would need to shorten the full round by **more than 5.2%**
to improve output/s in that interval. In the preceding, lower-acceptance
interval the corresponding break-even saving is only 2.4%. This calculation is
a screening rule; a same-prompt five/six/seven-token A/B must measure both
round time and actual acceptance.

The next kernel question is why down projection in the GDN layers takes
47.56 µs in the full target trace versus 39.18 µs with rotating weights
alone. A same-context serving trace and a working DRAM-traffic counter would
separate cache contention from kernel behavior. Simple tile tuning has little
modeled room for gate/up; reducing weight traffic could have a larger effect.

## Grouped-convolution roofline estimate

The [Triton kernel](../../vllm/model_executor/models/qwen3_dflash2.py) computes
an eight-row, 5120-channel, two-tap grouped convolution at batch one. Seven
rows use both taps; the first row uses one. Counting coefficient additions,
multiplications, and accumulation gives approximately 189,440 FLOPs/call.

The minimum distinct BF16 data for one call is about 0.194 MB: input 0.082 MB,
output 0.082 MB, base coefficients 0.020 MB, and used projected coefficients
0.010 MB. Counting repeated loads in each channel instead gives about 0.543 MB
of logical load/store traffic. Arithmetic intensity is therefore about
0.98 FLOP/byte with ideal reuse, or 0.35 FLOP/byte counting all loads. Actual
DRAM traffic was **not measured**; repeated values and the small working set can
be served by cache.

A standalone Triton CUDA-graph benchmark on one R9700 measured:

| Rows | Element block | Warps | Time |
| ---: | ---: | ---: | ---: |
| 8 | 512 (current choice) | 4 | 2.57 µs |
| 8 | 1024 | 4 | 2.45 µs |
| 8 | 512 | 8 | 2.43 µs |
| 8 | 1024 | 8 | 2.54 µs |
| 64 | 512 | 4 | 3.18 µs |
| 64 | 1024 | 4 | 2.88 µs |
| 128 | 1024 (current choice) | 4 | 3.64 µs |

A tiny Triton kernel took 2.12 µs in the same graph benchmark. A 128 MiB
device-to-device copy reached about 603 GB/s when counting both read and write
bytes. At that rate, moving 0.194 MB would take about 0.32 µs, before launch
cost. This is a **traffic-based bound**, not a counter-based hardware roofline:
the convolution's repeated graph inputs can remain cached. The batch-one
measurements are close to the graph-launch floor; changing the block size
improved the standalone result by only about 5%.

Using the in-vLLM mean, 20 convolution calls cost about 0.030 ms/draft pass,
or 0.09% of the patched 32.59 ms round at 128K. Even eliminating them would
have a negligible effect on batch-one throughput. The standalone benchmark and
the in-vLLM trace have different timing conditions; both put this kernel far
below the next measured targets. Its share may matter more at larger batches.

## Target verification preparation at 128K

An annotated offline run at 127,998 prompt tokens wrapped the V2 target
runner's input and attention preparation with `torch.profiler.record_function`.
The profile contains one prefill and six decode preparations per TP rank. The
table gives the range of per-rank medians over the final five decode steps.
Nested rows are included in their parent and must not be added.

| Host scope | Median CPU time across TP ranks | What it does |
| --- | ---: | --- |
| `GPUModelRunner.prepare_inputs` | 0.35–0.43 ms | Stage request indices, query offsets, positions, sequence lengths, and sampled/draft tokens |
| `GPUModelRunner.prepare_attn` | 0.13–0.15 ms | Gather block tables and compute KV write slots |
| `MambaHybridModelState.preprocess_state` | 0.09–0.11 ms | Advance and pre-copy recurrent state at block boundaries |
| `MambaHybridModelState.prepare_attn` | 3.81–5.20 ms | Build hybrid attention metadata, including the row below |
| `build_attn_metadata` | 3.68–5.02 ms | Build metadata for 15 KV cache groups |
| `DefaultModelState.prepare_inputs_embeds` | 0.55–1.19 ms | Prepare target input embeddings |
| `DefaultModelState.prepare_inputs` | 0.07–0.15 ms | Prepare model-specific position inputs |

The runner's `prepare_attn` invokes one fused block-table gather and one
fused slot-mapping kernel for the 15 cache groups. Their median GPU times are
approximately 3.1–3.4 µs and 2.8 µs, respectively. The input-staging kernels
are similarly small, about 1.7–2.7 µs each. These calls do not explain the
27.11 ms target verification GPU range.

A second run labeled each group's metadata builder. Ten groups use
`GDNAttentionMetadataBuilder`; each normally takes about 0.3–0.4 ms of CPU
time, totaling about 3.3–3.8 ms on ranks 0–2. The five segmented
attention metadata builders together take only tens of microseconds. GDN
builders account for over 90% of the metadata-building scope on those ranks.
Rank 3 had larger and less stable host timings, so the group figures are
representative rather than a precise four-rank mean.

For one steady rank-0 metadata build, CPU launch correlation links about 190
small GPU kernels to the 5.02 ms host scope, including 80 ROCm copy kernels.
Their summed GPU execution is about 0.33 ms and their GPU span about 1.05 ms.
The GDN builder repeatedly computes the same speculative-decode masks and
query lengths for each group, indexes its own block table, then copies results
into persistent CUDA-graph metadata buffers. The group-specific state indices
must be preserved, but batch-wide masks and token indices are candidates for
one shared computation or a fused per-group update.

This is a meaningful **host-side optimization candidate**, but the profiler
does not establish that all 3.7–5.0 ms extends the end-to-end step. Some CPU
preparation may overlap outstanding GPU work. A change should be judged by the
target-start-to-target-start GPU interval and request throughput, as well as
its local CPU timing. The instrumented 128K request generated 32 tokens in
1.37–1.38 seconds including cached-prefix preparation; it is not a steady
serving throughput measurement.

The benchmark and preparation artifacts were stored under
`/tmp/dflash2_profile_segmented128k/`,
`/tmp/dflash2_fp8_mlp_bench.py`, `/tmp/dflash2_mlp_att_gate_min/`,
`/tmp/dflash2_mlp_att_down_min/`, `/tmp/dflash2_mlp_pmc_gate/`,
`/tmp/bench_dflash2_grouped_conv.py`,
`/tmp/dflash2_128k_prepare_probe.py`, `/tmp/dflash2_profile_prepare128k/`,
`/tmp/dflash2_128k_builder_probe.py`, and
`/tmp/dflash2_profile_builders128k/` on the profiling host. These temporary
artifacts are not committed with this note.
