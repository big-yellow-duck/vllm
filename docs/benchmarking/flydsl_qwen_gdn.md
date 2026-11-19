# FlyDSL Qwen GDN decode on RDNA4

The Qwen3.8-27B-FP8 monokernel runs causal convolution, gated delta recurrence,
gated RMSNorm, and dynamic group-128 FP8 activation quantization for one GDN
layer in one launch. Each workgroup owns a key head and its three value heads,
so it can reuse the convolved Q and K and update each cache location once.
The implementation follows the recurrence staging and packed FP8 epilogues
in the FlyDSL mono kernels. [Kimi-K3 recurrence](https://github.com/ROCm/FlyDSL/blob/802b3c92d6d4813b92a213b468959eb9dd520d85/kernels/monokernel/k3/kda_recurrence.py)

## Kernel latency

Measured independently on two Radeon PRO R9700 GPUs (gfx1201) with TP2
shard shapes, BF16 convolution and recurrent caches, and SD convolution
layout. Each graph contains 128 calls, and the table reports the median of
10 graph replays, divided by the number of calls. The stock sequence includes
the FP8 input quantization for the output projection.

| Decode tokens | GPU 0 stock → FlyDSL | GPU 1 stock → FlyDSL |
| ---: | ---: | ---: |
| 1 | 13.54 → 7.01 µs | 13.56 → 6.96 µs |
| 4 | 16.28 → 7.29 µs | 16.15 → 7.31 µs |
| 8 | 19.60 → 10.42 µs | 19.45 → 10.32 µs |
| 16 | 28.31 → 18.23 µs | 28.12 → 18.35 µs |

```bash
HIP_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_qwen_gdn_flydsl.py
HIP_VISIBLE_DEVICES=1 python benchmarks/kernels/benchmark_qwen_gdn_flydsl.py
```

## TP2 serving

Measured with the Qwen3.8-27B-FP8 checkpoint at
`c3a462cc3d38cf351f3b26cbd763469eeb74ef81`, PyTorch 2.13.0+rocm10.0.0,
Triton 3.8.0, FlyDSL `495bb86`, and AITER `ff887216`. Both workers used
`ROCM_ATTN`, Triton block-FP8 linears, 4 GiB of KV cache per rank, and graph
capture sizes 1, 2, 4 and 8.

The benchmark sends simultaneous streamed completions at each concurrency.
It generates 96 tokens per request with greedy sampling, warms each concurrency,
and reports medians over three runs. Throughput includes prefill. Inter-token
latency comes from the server's Prometheus histogram.

| Concurrent requests | Stock output tok/s | FlyDSL output tok/s | Stock ITL | FlyDSL ITL |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 26.20 | 33.56 | 37.42 ms | 29.01 ms |
| 4 | 113.71 | 113.79 | 33.55 ms | 32.92 ms |
| 8 | 205.59 | 207.63 | 35.46 ms | 34.89 ms |

During a request with one prefill and eight decode steps, the TP2 profiler
recorded 384 monokernel dispatches on each rank. The 48 GDN layers account for
192 stock dispatches per decode step and 48 fused dispatches, saving 144
launches per rank. Across the whole traced request, the kernel event count
dropped from 10,671 to 9,472 on rank 0 and from 10,674 to 9,474 on rank 1.

Two of three 64-token greedy smoke prompts matched exactly. The third
completion shared its first two tokens with the stock run. The cache-update
and FP8 output tests passed on both GPUs. The existing sigmoid-gated RMSNorm
bitwise test differed by one BF16 element on the clean upstream checkout.

Set `FLYDSL_GDN=0` for stock serving or `FLYDSL_GDN=1` for the monokernel in
the command below.

```bash
HIP_VISIBLE_DEVICES=0,1 \
VLLM_ROCM_USE_AITER=0 VLLM_GDN_DECODE_KERNEL=triton NCCL_PROTO=Simple \
VLLM_ROCM_USE_FLYDSL_QWEN_FP8="${FLYDSL_GDN:-1}" \
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --revision c3a462cc3d38cf351f3b26cbd763469eeb74ef81 \
  --served-model-name qwen38 --tensor-parallel-size 2 \
  --distributed-executor-backend mp --language-model-only \
  --dtype bfloat16 --attention-backend ROCM_ATTN --max-model-len 4096 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 4294967296 --no-enable-prefix-caching \
  --cudagraph-capture-sizes 1 2 4 8
```
