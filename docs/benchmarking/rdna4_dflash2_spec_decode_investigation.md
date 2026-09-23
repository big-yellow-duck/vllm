# RDNA4 DFlash2 speculative decoding: baseline investigation

Date: 2026-09-23. This note records the **pre-change baseline** and root-cause
investigation at batch one for
`Qwen/Qwen3.8-27B-FP8` with `incoai/Qwen3.8-27B-DFlash2` on four AMD Radeon AI
PRO R9700 GPUs (tensor parallel size 4). The serving configuration is in
the now-updated [`serve_rdna4-spec-decode.bash`](../../serve_rdna4-spec-decode.bash).
The checkout was at `ee303eec4fca`; the offline engine log reported version
`v0.1.dev21496+g8538017f4.d20260917`.

For the implementation, before/after comparison, autotuning, and next targets,
see the [segmented-attention speedup note](rdna4_dflash2_segmented_attention_speedup.md).

## Baseline finding

At 15,952 input tokens, the draft's five prefix-attention kernels consumed
28.72 ms of a 57.91 ms speculative step. They scanned the entire prefix before
masking away keys outside the 2048-token sliding window. The target already used
segmented attention; the draft selected `ROCM_ATTN` because its non-causal
sliding-window attention was unsupported by the segmented path. The original
request logs and the offline profile below use different prompts and should not
be compared as a controlled serving A/B.

## Symptom and break-even calculation

The serving logs supplied for one running request with an approximately
128K-token prompt showed about 58.5 generated tokens/s without speculation and
about 21 tokens/s with seven draft tokens per round. With speculation, a
representative ten-second interval reported 553
drafted tokens, 131 accepted draft tokens, and a mean acceptance length of 2.66.
The per-position acceptance rates fell from roughly 0.65–0.70 at position one
to roughly 0.01–0.05 at position seven.

At seven proposed tokens per round, 553 drafted tokens in ten seconds imply
about 7.9 rounds/s, or 127 ms/round. The observed 21 generated tokens/s imply
about 2.66 output tokens/round: roughly 1.66 accepted draft tokens plus one
target token. At the 58.5 tokens/s baseline, producing 2.66 tokens would take
about 45.5 ms. Speculation would need to save roughly 81 ms/round to break even
on this workload. These are estimates from aggregate logs, not a per-round
latency trace.

## Decode path and attention backends

The `DFlash2DraftModel` architecture selects
[`DFlash2Speculator`](../../vllm/v1/worker/gpu/spec_decode/__init__.py). Its
[`_generate_draft`](../../vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py)
runs the draft model, computes candidates and unary scores, scores candidate
paths, and selects a path for target verification. The base DFlash
[`propose`](../../vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) prepares
draft context KV before the query pass. Seven proposed tokens require eight
draft query positions, including the anchor position.

The target selected `ROCM_SEGMENTED_ATTN`. Its 64-layer configuration has one
full-attention layer in each group of four; the remaining layers use linear
attention. The offline trace confirms target segmented attention ran.

The draft has five non-causal sliding-attention layers with a 2048-token window.
The [draft loader](../../vllm/v1/worker/gpu/spec_decode/dflash/utils.py)
selects an attention backend compatible with non-causal attention. Before the
change described below, the draft selected `ROCM_ATTN`. The
[segmented-attention fast-path predicate](../../vllm/v1/attention/ops/segmented_attention.py)
required causal attention without a sliding window, and the segmented backend
did not advertise non-causal support. Thus the five draft layers did not use the
segmented kernel even when the target explicitly selected it. This did not
indicate that target segmented attention failed.

The warning `Cannot use ROCm custom paged attention kernel, falling back to
Triton implementation` comes from the draft's
[`chunked_prefill_paged_decode`](../../vllm/v1/attention/ops/chunked_prefill_paged_decode.py).
It concerns the native ROCm *paged decode* kernel, not the target's segmented
kernel. On RDNA, [native paged-decode eligibility](../../vllm/platforms/rocm.py)
requires no sliding window and `kv_cache_dtype="auto"`; this draft uses a
sliding window and FP8 KV cache.

For the eight-query draft pass, `chunked_prefill_paged_decode` launches Triton
[`context_attention_fwd`](../../vllm/v1/attention/ops/prefix_prefill.py) with
`skip_decode=True`. The fallback also launches `kernel_paged_attention_2d`,
but [that kernel returns](../../vllm/v1/attention/ops/chunked_prefill_paged_decode.py)
when the request's query length is greater than one. The prefix-attention
kernel does the substantive draft attention work in this case.

## Offline measurement

An offline run used the same target, draft, TP4, FP8 KV cache, and target
attention backend, with `max_model_len=8192`, `max_num_seqs=8`, and a short
repeated prompt (`"Explain why the sky appears blue in a concise paragraph. "`
repeated eight times). Sampling used temperature zero and `ignore_eos=True`.
The first 64-token request took 1.683 s, including warmup and JIT activity;
the second took 0.576 s (111 tokens/s). A subsequent 32-token request produced
a rank-zero PyTorch GPU trace. This short, warm run is **not** equivalent to the
long-context serving measurements above.

Selected rank-zero GPU kernel times from that trace:

| Kernel | Calls | Total GPU time | Mean time | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Draft Triton prefix attention `_fwd_kernel` | 45 | 5.857 ms | 130.1 µs | Five layers across nine draft passes |
| Triton `kernel_paged_attention_2d` | 45 | 0.056 ms | 1.24 µs | Returns for eight-query passes |
| DFlash2 grouped convolution | 180 | 0.257 ms | 1.43 µs | 20 calls per draft pass |
| Target `_segmented_prefill_stage` (now `_segmented_attention_stage`) | 144 | 0.871 ms | 6.05 µs | Target segmented path was active |

The grouped-convolution total was 0.10% of rank-zero GPU time in this profile.
That percentage includes other work in the capture window and should not be
used as the serving-run percentage. The absolute call count and times are more
useful for locating the optimization opportunity.

## Complete speculative step in the short-prompt trace

The PyTorch trace includes `execute_context_0(0)_generation_1(8)` GPU ranges
around each target pass. For six steady passes, the median time from one target
range start to the next was **29.08 ms**, consistent across all four TP ranks.
The target range occupied 22.97 ms and the interval until the next target range
occupied 6.10 ms. Grouping kernels by launch correlation and order gives this
approximate GPU timeline:

| Phase within one steady step | GPU wall time | Evidence |
| --- | ---: | --- |
| Target input preparation and verification forward | 22.97 ms | `execute_context_0...` GPU range |
| Target logits, communication, and rejection sampling | 1.30 ms | Logits GEMM, NCCL, then `_rejection_kernel` |
| Draft context update and query setup | 0.57 ms | Post-update, context GEMM, KV writes, `_prepare_dflash_inputs_kernel` |
| Draft full-graph query, candidates, and selector | 4.21 ms | One `hipGraphLaunch` correlation group with 124 kernels |
| Remaining bookkeeping | About 0.02 ms | Two small kernels before the next target range |

The phase boundaries are inferred from kernel order, so the 1.30 ms and 0.57 ms
subdivisions are approximate. The target and full draft-graph boundaries are
directly visible in the trace.

Within the 22.97 ms target range, 256 `packed_kernel_0` launches account for
about **11.5 ms** of kernel execution. Their generic name prevents a definitive
source-op attribution from this trace alone. Target mapped all-reduce kernels
account for about 1.9 ms; all 16 segmented-attention stage/reduce pairs account
for about 0.16 ms. About 17.2 ms is named kernel execution and 5.8 ms is
inter-kernel or synchronization time within the target range; these are not
necessarily independent optimization opportunities.

Within the 4.21 ms draft graph, GEMM kernels account for about **2.8 ms**, the
five Triton prefix-attention launches about **0.68 ms**, and 20 grouped
convolutions about **0.029 ms**. Rejection and selector kernels are each only a
few microseconds. For this short prompt, draft GEMMs are a larger lead than
draft attention, and target verification dominates the complete step.

## Context-length comparison and root cause

A second offline trace repeated the same prompt to **3962 input tokens** with
`max_model_len=8192` and the same model, TP, attention, KV, and speculative
settings. A third trace used **15,952 input tokens** and `max_model_len=32768`
to fit them; its other settings were the same. Each trace came from a request
after an initial run of its prompt shape. The raw traces passed gzip validation
on all four ranks. PyTorch's optional stack postprocessing raised a UTF-8
decoding error when stopping the 3962-token profiler, but its GPU event timing
remains available. The 15,952-token trace used stack tracing disabled and
completed normally. Request wall times include prefill and profiling overhead;
the step timings below come directly from GPU events.

Median steady-step timings across TP ranks:

| GPU wall-time component | About 90 input tokens | 3962 input tokens | 15,952 input tokens |
| --- | ---: | ---: | ---: |
| Target preparation and verification forward | 22.97 ms | 22.96 ms | 23.28 ms |
| Target-to-next-target interval | 6.10 ms | 12.88 ms | 34.60 ms |
| Captured draft graph within that interval | 4.21 ms | 10.98 ms | 32.61 ms |
| Five draft prefix-attention kernels within the graph | 0.68 ms | 7.40 ms | 28.72 ms |
| 20 grouped-convolution kernels within the graph | 0.029 ms | 0.029 ms | 0.029 ms |
| **Complete target-start-to-target-start step** | **29.08 ms** | **35.82 ms** | **57.91 ms** |

From about 90 to 3962 input tokens, the full step grew by about 6.75 ms and
draft prefix attention grew by about 6.72 ms. Thus essentially all additional
time came from draft attention. Target segmented-attention stage/reduce time
rose only from about 0.16 ms to 0.24 ms across all 16 full-attention layers.
Between 3962 and 15,952 tokens, the full step grew another 22.09 ms and draft
prefix attention grew 21.32 ms; target segmented stage/reduce reached about
0.58 ms.
All four TP ranks reported nearly identical steady-step timings.

The reason is visible in [`_fwd_kernel`](../../vllm/v1/attention/ops/prefix_prefill.py):
its context loop starts at token zero and runs to `cur_batch_ctx_len` in
32-token tiles. It loads K/V and computes QK for each tile. Only afterward
does it apply the sliding-window mask. For the 3962-token input, it therefore
processes roughly 124 tiles per head even though a 2048-token window needs
only about 64 tiles. At 15,952 tokens it processes roughly 499 tiles where
roughly 64 could contain visible keys. The same unnecessary prefix scan grows
with context length beyond the window. This explains the measured growth
without assuming that the target's segmented kernel regressed.

At 15,952 tokens, about 87% of the prefix fallback's scanned context tiles are
outside the window. This identified the draft attention route as the first
optimization target. The [follow-up note](rdna4_dflash2_segmented_attention_speedup.md)
records the segmented sliding-window implementation and its measurements.

The baseline offline probes and traces were stored under
`/tmp/dflash2_offline_probe.py`, `/tmp/dflash2_profile/`,
`/tmp/dflash2_long_context_probe.py`, `/tmp/dflash2_profile_long/`,
`/tmp/dflash2_16k_probe.py`, and `/tmp/dflash2_profile_long16k/` on the
profiling host. These temporary artifacts are not committed with this note.
