# RDNA4 DFlash2: segmented sliding-window attention speedup

Date: 2026-09-23. Follow-up to the
[pre-change baseline investigation](rdna4_dflash2_spec_decode_investigation.md).
The workload uses `Qwen/Qwen3.8-27B-FP8` as the target and
`incoai/Qwen3.8-27B-DFlash2` as the draft, TP4 on four AMD Radeon AI PRO R9700
GPUs. There are seven proposed tokens and eight draft query positions per
speculative step. The target uses `ROCM_SEGMENTED_ATTN` throughout these
comparisons; the draft attention backend is the changed path.

## Progression and current result

| Stage | What changed or was measured | Result |
| --- | --- | --- |
| Original request, before the change | Draft fell back to full-prefix Triton attention | About 21 output tokens/s with speculation versus 58.5 without it, from the user's approximately 128K request logs |
| Baseline offline diagnosis at 15,952 tokens | Measured five draft prefix-attention calls per step | 28.72 ms attention, 32.61 ms draft graph, 57.91 ms complete GPU step |
| Segmented sliding-window path, static configuration | Routed the five non-causal draft layers to bounded segmented prefill | 0.041 ms attention, 3.60 ms draft graph, 29.14 ms complete GPU step at 15,952 tokens |
| SWA-aware startup autotuning | Tuned window and causality as part of the cache identity | 58 draft buckets tuned on all four TP ranks without failures; the batch-one eight-query kernel improved 1.033× over static in its isolated tuner |
| Controlled 127,998-token serving A/B | Same synthetic prompt and target configuration, with only the draft backend changed | 99.49 final output tokens/s with tuned segmented attention versus 11.16 with the original fallback; identical 1,024-token text |
| Real-world live request | User supplied two steady ten-second log intervals from the tuned server | 109.1 and 139.1 final output tokens/s at essentially the same 37.2–37.3 speculative rounds/s; acceptance varied |
| Reproducible original workload and model evaluation | Pending | The full original request and sampling settings have not been captured for a controlled A/B or evaluation |

The original request's 21 tokens/s and the controlled fallback's 11.16 tokens/s
come from **different prompts and acceptance behavior**. The 8.91× serving
speedup below compares only the two controlled synthetic-prompt runs. The
offline GPU step timings are a separate measurement from streamed API output.

## Segmented sliding-window implementation

[`segmented_attention.py`](../../vllm/v1/attention/ops/segmented_attention.py) now
accepts a normalized left-window extent and a causal flag. For each query tile,
the first possible key is derived from the first query position, clamped to
zero, and aligned down to the key tile size. Causal mode also bounds the end of
the range. The existing per-row mask remains in place, so the aligned first tile
and the different query offsets preserve exact window semantics. Non-causal
mode permits each of DFlash2's eight query positions to see the complete query
chunk while retaining the per-query left boundary.

The dispatcher now admits sliding-window and non-causal patterns when the other
segmented constraints are satisfied. Configuration selection uses the effective
attention span, `min(sequence length, window extent + query length)`, rather
than the 128K sequence length. The backend advertises non-causal support, and
the serving reference explicitly sets the draft's `attention_backend` to
`ROCM_SEGMENTED_ATTN`. The segmented backend remains opt-in.

The window passed by the ROCm implementation is already normalized: a configured
2048-token window arrives as a left extent of 2047. The kernel uses an inclusive
lower bound of `query position - 2047`, giving exactly 2048 visible positions in
causal mode. Tests cover causal and non-causal execution with BF16 and FP8 KV
caches using DFlash2's 8-query, 32-head, 8-KV-head, dimension-128 geometry.

Focused attention tests passed (15 cases), including BF16/FP8, causal and
non-causal windows, window extents 0 and 2047, tuner cache separation, and
startup routing. Python compilation and `git diff --check` also passed. A
standalone GPU tune of the batch-one, eight-query, 2055-position non-causal
draft bucket completed. These checks validate the new route and tuner wiring;
the controlled serving result below supplies the end-to-end measurement.

## Results after routing DFlash2 to segmented prefill

The patched run used `ROCM_SEGMENTED_ATTN` for both target and draft. The
baseline already used this backend for the target but used `ROCM_ATTN` for the
draft. Segmented autotuning was disabled during these offline comparisons; the
dispatcher used static configurations. The 15,952-token run is directly
comparable to the [baseline trace](rdna4_dflash2_spec_decode_investigation.md#context-length-comparison-and-root-cause).
A second patched run used a 127,998-token prompt with
`max_model_len=131072`. These are median GPU wall times between consecutive
target verification starts in steady speculative decoding, not request wall
times. The 128K trace contains four such intervals per rank.

| GPU wall-time component | 15,952 baseline | 15,952 segmented | Change at 15,952 | 127,998 segmented |
| --- | ---: | ---: | ---: | ---: |
| Target preparation and verification | 23.28 ms | 23.66 ms | +0.38 ms | 27.11 ms |
| Target-to-next-target interval | 34.60 ms | 5.48 ms | −29.12 ms | 5.48 ms |
| Captured draft graph | 32.61 ms | 3.60 ms | −29.01 ms; 9.1× faster | 3.60 ms |
| Five draft attention kernels | 28.72 ms | 0.041 ms | −28.68 ms; about 700× faster | 0.041 ms |
| 20 grouped-convolution kernels | 0.029 ms | 0.030 ms | +0.001 ms | 0.030 ms |
| **Complete speculative step** | **57.91 ms** | **29.14 ms** | **−28.77 ms; 1.99× faster** | **32.59 ms** |

At 15,952 tokens, the complete step improved by 28.77 ms, or 49.7%. Draft
attention accounts for 28.68 ms of that improvement. Its cost stayed flat
once the sliding window was full; the draft graph was 3.60 ms at both measured
long contexts. All four TP ranks agreed within measurement noise. We did **not**
capture a 127,998-token pre-change GPU trace, so the 32.59 ms figure is a
patched-context result, not a direct 128K before/after delta. The old prefix
kernel's linear scan predicts worse latency at 128K, but the controlled serving
A/B below is the measured 128K comparison.

The 128K trace exposes the next context-dependent cost: the target's 16
full-attention segmented stages rise from 0.57 ms total at 16K to 4.07 ms at
128K. This is expected because those target layers use full attention. Target
verification consequently grows from 23.66 to 27.11 ms, while draft time stays
flat. The profile request's 23.0 output tokens/s includes roughly 0.995 seconds
of cached-prefix preparation and should not be compared directly with the
steady serving generation-throughput metric; the 32.59 ms cycle is the isolated
steady decode measurement.

A representative 128K round decomposes as follows:

| Phase | GPU wall time | Main work |
| --- | ---: | --- |
| Target verification forward | 27.11 ms | Target model over eight verification positions |
| Target logits, communication, and rejection sampling | 1.25 ms | 1.05 ms logits GEMM, 0.10 ms NCCL, sampling kernels |
| Draft context update and query setup | 0.60 ms | State update, context GEMM, five KV writes, DFlash input preparation |
| Draft full graph | 3.59 ms | Five attention layers, draft GEMMs, grouped convolution, candidate selection |
| Final bookkeeping | 0.02 ms | Two small index kernels |
| **Complete round** | **32.59 ms** | Target start to next target start |

Within the 27.11 ms target range, `packed_kernel_0` accounts for 11.49 ms,
the 16 segmented stages and reductions account for 4.13 ms, and mapped
all-reduce accounts for 1.92 ms. Other named kernels account for about 3.72 ms;
the remaining 5.85 ms is spacing or synchronization between kernels. The
subdivision after target verification is inferred from ordered kernel launches;
the target range and draft graph boundaries are direct trace events.

## Other measured costs after the attention change

The grouped convolution remains about 0.030 ms per draft pass. Target
verification and host metadata preparation have larger measured costs. See
the [post-segmented bottleneck note](rdna4_dflash2_post_segmented_bottlenecks.md)
for the convolution roofline and target preparation traces.

## SWA autotuning and controlled 128K serving result

The segmented startup tuner now keys its in-memory and persistent tables by
left window extent and causality. It compiles, verifies, and measures candidate
kernels with those same modes, and the DFlash2 layer supplies its eight-query
limit. For a sliding window, the tuner bounds the representative attention
span by `left extent + query length`; it does not allocate a full 128K KV cache
for each draft candidate. This is a bounded candidate search, not a claim of
global optimality. The earlier offline GPU traces used static configurations.

With autotuning enabled, all four TP ranks completed 150 target full-attention
buckets and 58 draft SWA buckets with zero failures. The draft table identity
was eight query heads, two KV heads, dimension 128, page 1648, left extent
2047, and `causal=False`. For `(batch=1, queries=8, attention span=2055)`, the
winner kept the static 16-way split and changed the stage count from one to
two. The tuner measured 20.16 µs and a 1.033× paired speedup over static for
that isolated candidate; this percentage is not the serving speedup. Cold
tuning took 173 seconds for the target and 42 seconds for the draft across
four TP ranks. The fallback server later loaded the same target table in
0.01 seconds per rank.

Three API servers used TP4, FP8 KV, `ROCM_SEGMENTED_ATTN` for the target,
`max_model_len=131072`, `max_num_batched_tokens=4096`, `max_num_seqs=8`, and
temperature zero. Each received the same 127,998-token prompt (the sky-blue
sentence repeated 11,636 times), a 32-token warmup request to populate prefix
cache, then a streamed 1,024-token request with `ignore_eos=True`. Decode rate
is `(1024 - 1) / (last streamed token time - first streamed token time)`;
startup and prompt processing are excluded from that rate.

| Serving mode | Draft attention | First token | Streamed decode | Full request | Final output rate | Relative to fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Speculative, original fallback | `ROCM_ATTN` prefix prefill | 1.22 s | 91.64 s | 93.11 s | 11.16 tokens/s | 1.00× |
| Speculative, tuned segmented | `ROCM_SEGMENTED_ATTN` SWA | 1.16 s | 10.28 s | 11.47 s | 99.49 tokens/s | 8.91× |
| No speculation | — | 0.58 s | 18.69 s | 19.27 s | 54.74 tokens/s | 4.90× |

The two speculative runs held the target configuration and KV page size fixed
and produced identical 1,024-token text (SHA-256
`50b452964ed42e1ef0f7b77c6df24a8fcd640e818235b49e75dc9327054b0bdf`).
Tuned segmented draft attention was **8.91×** faster than the original draft
fallback and **1.82×** faster than no-spec decoding on this prompt. The
no-spec run had a different target KV page size (1568 instead of 1648) and a
different output hash, so it is a serving reference rather than a controlled
kernel or output-equivalence check. Acceptance length varied substantially
through the generated text, making short serving-log intervals unsuitable for
the A/B headline. These results establish a reproducible foothold; they do
not substitute for the original request or a model output evaluation.

The 99.49 tokens/s counts **final tokens streamed to the client**: accepted
draft tokens plus tokens supplied by target verification. It excludes rejected
draft proposals. Likewise, 3.60 ms is a complete draft GPU graph per round,
not end-to-end throughput; seven proposals divided by 3.60 ms would be about
1,900 proposed tokens/s if that graph ran alone.

## Next targets

1. **Capture the exact real request and evaluate output.** The user has now
   observed 109–139 final output tokens/s with the live tuned server. Save a
   reproducible prompt and sampling configuration, then measure both throughput
   and the model evaluation. The 99.49 tokens/s synthetic A/B remains the only
   controlled backend comparison.
2. **Sweep draft length on that request.** The latest seventh-position
   acceptance was 0.194 output tokens per round; removing it needs more than
   a 5.2% full-round speedup to pay off. The preceding interval needs only
   2.4%. Compare five, six, and seven proposals on the same prompt, measuring
   both rounds/s and accepted output/s; see the
   [live-round accounting](rdna4_dflash2_post_segmented_bottlenecks.md#live-request-acceptance-versus-round-speed).
3. **Explain the MLP down-projection gap and reduce weight traffic.** Direct
   rotating-weight measurements on the exact MLP shapes reach 602 GB/s for
   gate/up and 569 GB/s for down, versus the R9700's 640 GB/s peak. The
   [kernel roofline](rdna4_dflash2_post_segmented_bottlenecks.md#direct-mlp-kernel-measurement-and-roofline)
   leaves little modeled benefit for simple gate/up tile tuning. The full
   target trace's GDN down projection is 47.56 µs versus 39.18 µs in the
   isolated rotation benchmark; investigate that gap in the same serving
   context. Reducing weight bytes or increasing reuse has a larger potential
   benefit than moving closer to the existing one-read-per-weight roof.
4. **Measure and reduce repeated small launches.** A representative target
   range has 1,752 GPU kernels and 5.85 ms of inter-kernel spacing, with no
   single gap over 10 µs. Attribute repeated GDN, quantization, and pointwise
   launches, then test fusion against complete round time.
5. **Profile the tuned target's full-attention stages at 128K.** The earlier
   static-config trace put 16 stages and reductions at 4.13 ms, or 13% of the
   round. Measure their new tuned cost and compare the selected split counts
   and tiles before another kernel change.
6. **Reduce repeated GDN metadata work if it is on the critical path.** Ten
   per-group builders take about
   3.3–3.8 ms of host time on the less noisy TP ranks. Prototype shared
   batch-wide speculative metadata and a batched or fused group-specific
   state-index update, then measure the full step to see how much was on the
   critical path. The runner's block-table and slot-mapping kernels are
   already too small to prioritize. The [preparation trace](rdna4_dflash2_post_segmented_bottlenecks.md#target-verification-preparation-at-128k)
   records the scope and overlap caveat; the current GPU trace has no long
   matching idle interval.

The segmented profiling scripts and raw traces were stored under
`/tmp/dflash2_16k_segmented_probe.py`,
`/tmp/dflash2_profile_segmented16k_v4/`,
`/tmp/dflash2_128k_segmented_probe.py`,
`/tmp/dflash2_profile_segmented128k/`, and
`/tmp/full_spec_step_report.py` on the profiling host. The serving
comparison used `/tmp/dflash2_serving_client.py`; server logs and client
results are `/tmp/dflash2_serve_{segmented_tuned,fallback_tuned_target,no_spec_tuned_target}.log`
and `/tmp/dflash2_result_{segmented,fallback,no_spec}.jsonl`. These temporary
artifacts are not committed with this note.
