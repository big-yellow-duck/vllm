# RedHatAI/gemma-4-12B-it-FP8-Dynamic · TP2 · short-mtp-effect-segmented-after

Ordinary decode → MTP1. These are end-to-end serving results on two Radeon AI PRO R9700 GPUs (gfx1201), with TP2, BF16 activations, FP8 model weights and explicit FP8 KV cache in every arm.

Model revision: `2083adceb2a3df0465e6021e08a7196d21dbbb44`. Workload: eight turns, 2,048 new synthetic prompt tokens and 128 output tokens per turn; streams 1/2/4/8; seed 20260715; prefix caching enabled. 1 fresh server repetitions per arm. Speculation is the experimental change; the attention backend matches. The MTP source snapshot adds host support for attention layers that read the target KV cache without projected K/V. Compiled segmented kernel cache keys, configuration selection and the existing projected-K/V dispatch path are unchanged, as validated in the linked fix report.

Segmented startup autotuning: `True`. Main model limits: context 32,768, eight sequences, batched-token limit 8,192, GPU memory utilization 0.90. Backend choice was inspected on both workers; automatic selection uses no attention-backend flag.

Observed baseline backends by rank: `[{'ROCM_SEGMENTED_ATTN': 48}, {'ROCM_SEGMENTED_ATTN': 48}]`. Candidate: `[{'ROCM_SEGMENTED_ATTN': 52}, {'ROCM_SEGMENTED_ATTN': 52}]`.

Model, data definition, streams, request limits and seed match. Every repetition contains 8/16/32/64 successful turn requests and no errored or incomplete requests. The table and charts use means over every successful request, including warmup/cooldown, through the skill's summary and plotting helpers. Positive TPS changes and negative TTFT/E2E changes improve performance.

| Streams | Output tok/s, baseline → candidate | Output tok/s change | Mean TTFT (ms), baseline → candidate | Mean TTFT (ms) change | Mean E2E (s), baseline → candidate | Mean E2E (s) change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.42 → 61.77 | **+42.3%** | 423 → 432 | +2.0% | 2.96 → 2.08 | **−29.6%** |
| 2 | 38.17 → 51.46 | **+34.8%** | 597 → 562 | **−5.8%** | 3.36 → 2.51 | **−25.5%** |
| 4 | 31.66 → 40.40 | **+27.6%** | 1,098 → 901 | **−17.9%** | 4.07 → 3.25 | **−20.0%** |
| 8 | 23.43 → 28.05 | **+19.7%** | 1,864 → 1,146 | **−38.5%** | 5.54 → 4.69 | **−15.4%** |

Output tok/s changes range from +19.7% to +42.3% across the four stream counts.
Mean TTFT (ms) changes range from -38.5% to +2.0% across the four stream counts.
Mean E2E (s) changes range from -29.6% to -15.4% across the four stream counts.

Every short condition runs a separate, untimed copy of the complete short workload before measurement. Prefix cache is reset afterward and counters are sampled after a 12-second flush interval. The measured GuideLLM report still includes all of its successful warmup/cooldown requests. Primer reports are retained separately.

MTP cache-reader support and checks: [validation](../../../validation/shared-kv-fix/validation.json), [kernel/dispatch equivalence](../../../validation/shared-kv-fix/math-equivalence.json), [tuning-cache reuse provenance](../../../validation/shared-kv-fix/cache-migration.json). The failed initial MTP startup and the original source snapshots are retained; no measured requests from that failure enter this comparison.

Achieved mean concurrency for each repetition:

| Arm | Repeat | Streams | Mean active requests |
|---|---:|---:|---:|
| Ordinary decode | 1 | 1 | 1.000 |
| Ordinary decode | 1 | 2 | 1.998 |
| Ordinary decode | 1 | 4 | 3.996 |
| Ordinary decode | 1 | 8 | 7.990 |
| MTP1 | 1 | 1 | 0.999 |
| MTP1 | 1 | 2 | 1.995 |
| MTP1 | 1 | 4 | 3.992 |
| MTP1 | 1 | 8 | 7.864 |

MTP counters are measured by subtracting Prometheus snapshots immediately before and after the measured workload; the initial smoke and any separate primer are excluded. Speculation uses one token and the matching pinned Google assistant checkpoint with its published BF16 weights.

| Arm | Draft tokens | Accepted | Acceptance |
|---|---:|---:|---:|
| MTP1 | 9195 | 6083 | 66.16% |

This short synthetic MTP run measures serving behavior under fixed output lengths. It does not establish acceptance or speedups on natural conversation data. The matching short non-MTP controls provide a same-workload speculation comparison.

Startup and tuning are excluded from serving latency. The main comparison has two server repetitions; short exploratory controls have one. Pooled request means are descriptive and do not establish statistical significance. Prefix cache is reset before each complete GuideLLM sweep; the original workload retains cache state between its stream levels.
Compiler caches persist across repetitions. If present, `logs/segmented-serving-2026-10-16/cache-reuse.json` records the reuse of matching model/backend compiler caches and target-attention tuning tables for MTP startup; existing identity and workload guards still validate cache entries.

![Output TPS](plots/compare_tps_by_turn.png)

![TTFT](plots/compare_ttft_by_turn.png)

![End-to-end latency](plots/compare_e2e_by_turn.png)

Raw reports:

- [Ordinary decode, repetition 1](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-decode-rep1/agent-multiturn-benchmarks.json), [server setup](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-decode-rep1/setup.json), [worker inspection](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-decode-rep1/inspection.json)
  [Separate untimed primer](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-decode-rep1/primer-multiturn-benchmarks.json).
- [MTP1, repetition 1](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-mtp1-rep1/agent-multiturn-benchmarks.json), [server setup](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-mtp1-rep1/setup.json), [worker inspection](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-mtp1-rep1/inspection.json)
  [Separate untimed primer](../../../raw/gemma-4-12b-fp8-dynamic/tp2-segmented-after-short-mtp1-rep1/primer-multiturn-benchmarks.json).
