# Segmented attention: benchmark review

Completed: 20 TP2 serving conditions, 2,400 successful measured turn requests, no errors or incomplete requests, and nine two-case comparisons. Separate short primers contribute another 960 successful requests and are excluded from measured means and MTP counter deltas.

Hardware: two Radeon AI PRO R9700 (gfx1201), 32 GiB each. Serving: FP8 checkpoint weights, BF16 activations, explicit FP8 KV cache in every case; context 32,768, max sequences 8, batched tokens 8,192, prefix caching enabled, seed 20260715. These results do not include separate BF16-weight checkpoints.

## Main simple-agent workload

Eight turns, 2,048 new synthetic prompt tokens and 512 output tokens per turn. Two fresh-server repetitions per arm. Gemma compares automatic TRITON_ATTN against opt-in ROCM_SEGMENTED_ATTN; Qwen compares the previous and updated segmented implementations. Values are mean per-request output tokens/s over all successful records, including warmup/cooldown. They are end-to-end serving measurements.

| Model | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B-FP8 | 27.00 → 27.28 (**+1.0%**) | 24.66 → 24.77 (**+0.4%**) | 21.86 → 21.95 (**+0.4%**) | 18.49 → 18.49 (~0%) |
| RedHatAI/gemma-4-12B-it-FP8-Dynamic | 21.65 → 36.37 (**+68.0%**) | 19.71 → 35.34 (**+79.3%**) | 16.42 → 36.91 (**+124.8%**) | 11.72 → 33.19 (**+183.1%**) |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | 33.32 → 53.32 (**+60.0%**) | 29.49 → 49.26 (**+67.1%**) | 23.48 → 47.03 (**+100.3%**) | 16.56 → 41.33 (**+149.5%**) |

All four Qwen runs logged Triton JIT-monitor warnings during measurement, including both repetitions. The per-repetition tables retain every request and show substantial absolute timing variation; the second repetition is near parity. Persistent compiler caches do not establish fully warmed execution. Small pooled Qwen differences do not establish a repeatable speedup. Gemma's larger improvements also need the per-repetition context in the detailed reports.

## Short Gemma MTP controls

Eight turns and 128 output tokens per turn; matching Google BF16 assistant; one speculative token. Each condition uses a fresh server, a complete separate untimed matching primer, a prefix-cache reset and counter flush before measurement. One repetition per condition makes these exploratory synthetic results.

| Model | Streams | Default TPS: plain → MTP | Segmented TPS: plain → MTP | Segmented/default TPS with MTP |
|---|---:|---:|---:|---:|
| gemma12-fp8 | 1 | 18.76 → 24.59 (**+31.1%**) | 43.42 → 61.77 (**+42.3%**) | **2.51×** |
| gemma12-fp8 | 2 | 14.69 → 17.58 (**+19.6%**) | 38.17 → 51.46 (**+34.8%**) | **2.93×** |
| gemma12-fp8 | 4 | 10.56 → 11.69 (**+10.6%**) | 31.66 → 40.40 (**+27.6%**) | **3.46×** |
| gemma12-fp8 | 8 | 6.62 → 6.87 (**+3.8%**) | 23.43 → 28.05 (**+19.7%**) | **4.08×** |
| gemma26-fp8 | 1 | 29.17 → 35.97 (**+23.3%**) | 75.16 → 97.55 (**+29.8%**) | **2.71×** |
| gemma26-fp8 | 2 | 22.47 → 26.18 (**+16.5%**) | 62.79 → 79.34 (**+26.4%**) | **3.03×** |
| gemma26-fp8 | 4 | 16.00 → 17.08 (**+6.7%**) | 48.79 → 58.79 (**+20.5%**) | **3.44×** |
| gemma26-fp8 | 8 | 10.12 → 10.15 (**+0.3%**) | 35.67 → 41.15 (**+15.4%**) | **4.06×** |

Measured aggregate draft acceptance for gemma12-fp8: segmented-after: 66.16%; default: 65.17%.
Measured aggregate draft acceptance for gemma26-fp8: segmented-after: 67.47%; default: 67.87%.

At eight streams, Gemma 26B default MTP is near TPS parity (**+0.3%**), while mean E2E rises 14.61 → 15.40s (+5.4%). Mean per-request rates and mean latencies average different quantities; the per-turn plots show the variation. The detailed comparisons retain both metrics.


MTP startup exposed attention layers that read the target KV cache with no projected K/V. The segmented host path now supports those readers. The fix passed 20 GPU tests, 319 CPU contracts and repository hooks; all three compiled Triton cache keys and 6,144 existing projected-K/V dispatch checks match. The MTP source snapshot is separate from the preserved main-study snapshots. [Validation](validation/shared-kv-fix/validation.json) and [cache reuse provenance](validation/shared-kv-fix/cache-migration.json).

## Compressed FP8 autotuning

The comprehensive kernel study passed 1,704/1,704 numerical cases. Eligible FP8 candidate count fell 43.0%, and summed search time fell 55.9% (18,498.51 → 8,150.61 seconds across unique profiles with compiler caches prewarmed). This is not one model's startup time. BF16/normal-only search is unchanged.

Aggregate compressed/full tuned performance is near parity, but 98 per-shape losses reproduced in independent fresh-input retests. The compressed search is not regression-free. [Full autotune report, tail results and plots](autotune/report.md).

## Detailed comparisons

- [Qwen/Qwen3.8-27B-FP8: full-workload](serving/qwen3.8-27b-fp8/full-workload/comparison.md)
- [RedHatAI/gemma-4-12B-it-FP8-Dynamic: full-workload](serving/gemma-4-12b-fp8-dynamic/full-workload/comparison.md)
- [RedHatAI/gemma-4-12B-it-FP8-Dynamic: short-mtp-backends](serving/gemma-4-12b-fp8-dynamic/short-mtp-backends/comparison.md)
- [RedHatAI/gemma-4-12B-it-FP8-Dynamic: short-mtp-effect-default](serving/gemma-4-12b-fp8-dynamic/short-mtp-effect-default/comparison.md)
- [RedHatAI/gemma-4-12B-it-FP8-Dynamic: short-mtp-effect-segmented-after](serving/gemma-4-12b-fp8-dynamic/short-mtp-effect-segmented-after/comparison.md)
- [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic: full-workload](serving/gemma-4-26b-a4b-fp8-dynamic/full-workload/comparison.md)
- [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic: short-mtp-backends](serving/gemma-4-26b-a4b-fp8-dynamic/short-mtp-backends/comparison.md)
- [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic: short-mtp-effect-default](serving/gemma-4-26b-a4b-fp8-dynamic/short-mtp-effect-default/comparison.md)
- [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic: short-mtp-effect-segmented-after](serving/gemma-4-26b-a4b-fp8-dynamic/short-mtp-effect-segmented-after/comparison.md)

Each comparison includes TPS, TTFT and E2E charts, request-level means, workload checks, achieved concurrency and raw-result links. Positive TPS changes and negative TTFT/E2E changes improve performance. All 18 required TPS/TTFT charts were visually inspected; all 27 serving plots are included.

## Provenance and next work

Verified pushed checkpoint: `b7a87748e35bad93d4c4fb2e47edf711c14dfad4` on `rocm/segmented-attn-gemma4`. The benchmark snapshots and exact file hashes are preserved separately in [the manifest](provenance/manifest.json). [Final audit](validation/final-audit.json), [visual inspection](validation/visual-inspection.json), [comparison skill](method/SKILL.md).

Next scope: Gemma 4 multimodal prefix attention support on `rocm/segmented-attn-gemma4-mm-prefix`, based on the pushed checkpoint. These serving runs use language-model-only mode; multimodal support is separate follow-up work. [Masking observations and validation plan](provenance/mm-prefix-followup.json).

The raw serving JSONs, primers, server setup/inspection, logs, derived request pools, chart helpers and validation records are included in this directory. Historical runtime paths inside JSON records describe the original workspace. Model weights, compiler caches and source worktrees remain in that workspace.
