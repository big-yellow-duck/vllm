# QSA simple-agent benchmark: BF16 baseline vs RDNA4 Octave K3V4

## Result

This is a real TP2 serving comparison on the two gfx1201 R9700 GPUs, using the prepared local Qwen3.8-Flash-Next W4A16 Modular snapshot at revision `1e7b246d7185a7f725b9e1a74f3bd252a97cff0a`. Both arms ran the same `simple-agent-multiturn-prefix-cache` collection workload: seed `20260715`, eight turns, 2,048 synthetic prompt tokens and 512 generated tokens per turn, with concurrency 1/2/4/8 and 8/16/32/64 requests. Vision stayed enabled; the workload itself contains text only.

Settings matched across arms: TP2, BF16 activations, PLE CPU offload, 65,536 max context, 8,192 max batched tokens, 8 max sequences, and 3 GiB manual KV allocation per GPU. Both used VLLM_COMPILE and default FULL_AND_PIECEWISE graphs. The baseline explicitly used BF16 main KV and no attention-backend override; the Octave arm used `ROCM_OCTAVE` and `octave_k3v4`.

Both measured sweeps completed 120/120 requests with 0 errors and 0 incomplete requests. The measured traces reached 20,118 prompt tokens (BF16) and 20,053 (Octave). Metrics reported 61,440 generated tokens per arm and zero KV preemptions.

| Streams | Baseline output tok/s | Octave output tok/s | Change | TTFT p50 / p95, baseline → Octave (ms) | TPOT p50 / p95, baseline → Octave (ms) | E2E p50 / p95, baseline → Octave (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.99 | 35.21 | +25.8% | 1,046 / 1,327 → 2,173 / 3,073 | 35.7 / 36.3 → 28.2 / 30.2 | 18.30 / 18.60 → 14.56 / 15.48 |
| 2 | 50.45 | 40.50 | −19.7% | 1,393 / 2,627 → 2,914 / 5,406 | 39.7 / 40.7 → 49.6 / 51.7 | 20.35 / 20.86 → 25.42 / 27.10 |
| 4 | 112.77 | 64.92 | −42.4% | 3,167 / 4,251 → 7,164 / 10,206 | 34.7 / 38.0 → 62.7 / 68.1 | 18.06 / 19.47 → 32.12 / 34.90 |
| 8 | 101.95 | 106.01 | +4.0% | 6,169 / 25,708 → 9,523 / 18,933 | 63.4 / 125.8 → 74.5 / 85.4 | 32.48 / 64.62 → 39.21 / 43.70 |

At stream 1, Octave improved aggregate generated-token throughput, TPOT, and E2E latency. At streams 2 and 4, Octave was slower on all three. At stream 8, aggregate output throughput was nearly even (+4.0% for Octave); Octave's median E2E was slower, while its p95 E2E was lower. These results do not support a blanket speedup claim.

## Startup KV capacity and worker selection

Both servers reserved exactly `3,221,225,472` bytes per GPU. The capacities below are the actual vLLM startup reports, not estimates.

| Arm | Startup KV tokens | Max concurrency at 65,536 | Aligned block | Startup group layout | QSA blocks/rank and QSA tensor shape |
|---|---:|---:|---:|---|---|
| BF16 baseline | 201,850 | 3.08× | 1,600 | `(1600, 1600, 1600, 1600, 4, 1600)` | 154; `[154, 1, 1600, 512]`, BF16 |
| Octave K3V4 | 498,801 | 7.61× | 3,136 | `(3136, 3136, 3136, 3136, 3136, 3136, 3136, 4, 3136)` | 274; `[274, 1, 3136, 248]`, uint8 |

The worker inspection found 12 QSA layers on each rank. In both arms the layer registry reports `QWEN4_EXP_QSA_TRITON` and `Qwen4ExpQSAImpl`; the decisive implementation field differs: baseline `native_engine` is null, while Octave uses `TunedHipOctave` on every QSA layer. The Octave cache has 2.47× the reported token capacity at the same byte budget.

The same 8-way aggregate ~8K-token prefill stress passed before each primer: eight successful requests, zero errors/incomplete, 1,076 actual prompt tokens per request. Prefix cache was reset after the stress and after the primer, immediately before each measured sweep.

## Caveats

- Triton JIT warnings occurred during both measured windows: five GDN/PLE kernels at 2-stream warmup/measurement. Their latency effects are included in the request timing. No profiler or additional per-request timing instrumentation was enabled.
- Octave logged two HIP allocator warnings during its 8-stream stage (each rank failed a 163,577,856-byte allocation while reporting zero free bytes). The service continued, the sweep completed 120/120, and Guidellm recorded zero errors/incomplete. Treat these as recovered allocation pressure and retain them when interpreting high-concurrency latency.
- An earlier baseline-only attempt at 4.5 GiB KV/GPU was stopped by a fatal 158 MiB PLE activation allocation OOM at an 8,001-token prefill step. That attempt is preserved in `../attempt-01-baseline-4p5gib-oom/`. Both reported comparison arms use the final matched 3 GiB setting.
- Timings are one sequential baseline-then-Octave pair, not repeated trials. The measured workload is synthetic text; multimodal image input remained enabled in server configuration but was not exercised by this collection.

## Evidence files

- `report.json`: machine-readable setup, worker inspection, KV layout/capacity, per-stream p50/p95 metrics, prefix-cache/preemption counters, JIT warnings, and hashes.
- `../baseline/` and `../octave/`: raw Guidellm reports, workloads, commands, server logs, metrics snapshots, inspector output, and stress/primer evidence.
- `../run.py` and `../worker_probe.py`: benchmark orchestrator and read-only worker inspection extension.
- `../attempt-01-baseline-4p5gib-oom/`: preserved failed 4.5 GiB baseline attempt.
