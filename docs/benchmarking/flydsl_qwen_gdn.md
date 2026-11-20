# FlyDSL Qwen GDN decode on RDNA4

The Qwen3.8-27B-FP8 monokernel runs causal convolution, gated delta recurrence,
gated RMSNorm, and dynamic group-128 FP8 activation quantization for one GDN
layer in one launch. Each workgroup owns a key head and its three value heads,
so it can reuse the convolved Q and K and update each cache location once.
The kernels live under `vllm/models/qwen3_5/amd/mono` and use FlyDSL's ROCm
buffer operations. The recurrence staging and packed FP8 epilogue follow
the FlyDSL mono kernels. [Kimi-K3 recurrence](https://github.com/ROCm/FlyDSL/blob/802b3c92d6d4813b92a213b468959eb9dd520d85/kernels/monokernel/k3/kda_recurrence.py)

## Kernel latency

Measured independently on two Radeon PRO R9700 GPUs (gfx1201) with TP2
shard shapes, BF16 convolution and recurrent caches, and SD convolution
layout. AITER was absent from the Python environment. Each graph contains 64
calls, and the table reports the median of 10 graph replays, divided by the
number of calls. The stock sequence includes
the FP8 input quantization for the output projection.

| Decode tokens | GPU 0 stock → FlyDSL | GPU 1 stock → FlyDSL |
| ---: | ---: | ---: |
| 1 | 13.81 → 7.27 µs | 13.50 → 7.05 µs |
| 4 | 16.56 → 7.59 µs | 16.10 → 7.32 µs |
| 8 | 19.85 → 11.12 µs | 19.30 → 10.84 µs |
| 16 | 28.42 → 18.90 µs | 27.92 → 18.86 µs |

```bash
HIP_VISIBLE_DEVICES=0 python benchmarks/kernels/benchmark_qwen_gdn_flydsl.py --graph-calls 64
HIP_VISIBLE_DEVICES=1 python benchmarks/kernels/benchmark_qwen_gdn_flydsl.py --graph-calls 64
```

## TP2 serving

The ordinary decode serving runs used `d06c122308` with the Qwen3.8-27B-FP8 checkpoint at
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

## DFlash2 verification

The verification monokernel keeps a recurrent-state tile in FP32 registers
across all offered tokens. It reads the previous round's accepted prefix,
rolls the convolution history forward, and writes the per-prefix snapshots
vLLM uses to resume after rejection. Its epilogue emits FP8 directly. AMD's
megakernel uses the same verify-then-commit pattern with compact replay
records in its own serving path.
[AMD's Qwen3.8 megakernel write-up](https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html#speculative-decoding-more-tokens-per-pass)

### Launch configuration

The sweep covered 1, 2, 4, and 8 requests and the same set of verification
lengths on both R9700s. Each verification length includes the target token.
With TP2 and BF16 state, 768 threads per workgroup gave the best results.
The selector uses two lanes per state row for one request, eight for four or
more requests with at most two positions each, and four for the remaining
shapes. It uses the configured length for ragged batches. TP1 and FP32 state
use four lanes per row.

| Requests | Positions per request | Fixed four lanes, GPU 0 / 1 | Selected lanes, GPU 0 / 1 |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 15.50 / 15.53 µs | 14.85 / 14.88 µs |
| 1 | 8 | 26.24 / 26.20 µs | 25.11 / 25.11 µs |
| 8 | 1 | 13.12 / 13.83 µs | 12.21 / 13.29 µs |
| 8 | 2 | 19.24 / 21.11 µs | 18.91 / 19.43 µs |

With seven draft tokens and the target token, the selected layouts give the
following per-layer latencies. The stock sequence includes convolution,
recurrence, gated RMSNorm, and dynamic group-128 FP8 quantization.

| Requests | GPU 0 stock → FlyDSL | GPU 1 stock → FlyDSL |
| ---: | ---: | ---: |
| 1 | 37.84 → 25.13 µs | 36.59 → 25.00 µs |
| 2 | 44.50 → 26.60 µs | 42.87 → 26.88 µs |
| 4 | 53.96 → 30.23 µs | 54.58 → 31.23 µs |
| 8 | 72.69 → 53.55 µs | 75.15 → 53.45 µs |

At 128 verification positions per batch the two paths were within 2%.
At 256, the fused kernel took about 12% longer. The serving path falls back to
the standard GDN core above 64 positions per batch or seven drafts per request.

```bash
for gpu in 0 1; do
  HIP_VISIBLE_DEVICES="$gpu" python benchmarks/kernels/benchmark_qwen_gdn_flydsl.py \
    --verify-positions 1 2 4 8 --batches 1 2 4 8 --graph-calls 64
done
```

### TP2 serving with segmented attention

Both the target and `z-lab/Qwen3.8-27B-DFlash2` used `ROCM_SEGMENTED_ATTN` on
both GPUs, with seven draft tokens per request and graph capture sizes 8, 16,
32, and 64. The target checkpoint, PyTorch, Triton, and FlyDSL versions
match the ordinary decode measurements. AITER was absent from both serving
runs. Each cell reports the median of three runs generating
128 tokens per request after warmup. Per-request decode throughput measures
the stream from the first to the last token. Aggregate throughput includes
prefill and request completion.

| Prompt tokens | Concurrent requests | Decode tok/s per request, stock → FlyDSL | Aggregate tok/s, stock → FlyDSL |
| ---: | ---: | ---: | ---: |
| 192 | 1 | 143.39 → 158.13 | 123.81 → 135.39 |
| 192 | 4 | 79.92 → 90.72 | 205.86 → 227.29 |
| 192 | 8 | 79.79 → 78.06 | 305.58 → 285.43 |
| 2,048 | 1 | 77.87 → 85.21 | 59.50 → 63.98 |
| 2,048 | 4 | 66.54 → 75.32 | 130.73 → 133.42 |
| 2,048 | 8 | 52.95 → 57.01 | 133.33 → 139.00 |

Mean acceptance length for 192-token prompts at concurrency eight changed
from 4.40 to 4.12 tokens per verification step.

The serving trace captured four verification steps. It recorded 192
monokernel launches on each rank, one per GDN layer per step. Total kernel
events dropped from 8,461 to 6,493 on rank 0 and from 8,458 to 6,492 on rank 1
(23% fewer). Both traces included the DFlash2 selector and segmented
attention kernels.

One of four 96-token greedy completions matched exactly. The other three
shared 33, 6, and 37 initial tokens with the stock completions. The
accepted-prefix cache and FP8 output tests passed on both GPUs.

On 96 GSM8K test questions sampled with Python random seed 42, stock answered
57 correctly and FlyDSL answered 60 correctly. Both runs used greedy generation
with thinking disabled, a 224-token output budget, and concurrency four.
[GSM8K dataset](https://github.com/openai/grade-school-math)

Use the serving command above with `--attention-backend ROCM_SEGMENTED_ATTN`,
`--cudagraph-capture-sizes 8 16 32 64`, and this additional argument.

```bash
--speculative-config '{
  "method": "dflash",
  "model": "z-lab/Qwen3.8-27B-DFlash2",
  "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
  "num_speculative_tokens": 7,
  "draft_tensor_parallel_size": 2,
  "attention_backend": "ROCM_SEGMENTED_ATTN"
}'
```

## Reduction and synchronization profiling

The baseline for this follow-up is `20676974df`. ROCprofiler thread traces
on the R9700s attributed 75% of sampled stalls at one request and eight
verification positions to vector-memory load waits. The compiled
cross-lane reductions already used DPP and VALU permutations. The
per-lane recurrent dot products used ordered FP32 sums. The workgroup
fences also emitted global-memory waits and cache invalidation at shared
memory handoffs.
[ROCprofiler thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.0.0/how-to/using-thread-trace.html)

The verifier uses scalar buffer loads for wave-uniform request metadata
and issues each snapshot-slot load before its dot products. That removes
the LDS lookup and lets the scheduler overlap the load with recurrence
arithmetic. Ordinary decode assigns separate producer waves to Q and K and
reads their LDS fragments as vectors.

Both kernels fence LDS at workgroup handoffs. The release and acquire
fences carry LLVM's local-address-space annotation, and kernel completion
orders the global cache stores before the next dispatch. Workgroup
rendezvous protect the cross-wave LDS dependencies. The gfx1201 ISA keeps
two workgroup rendezvous in decode and three in verification, with the
global cache invalidations removed from those handoffs.
[LLVM AMDGPU fence address spaces](https://llvm.org/docs/AMDGPUUsage.html#fence-and-address-spaces)

The final kernel uses the original FP32 sum order. Reassociation sped up the
isolated kernel and reduced DFlash2 acceptance from 5.20 to 4.31 tokens per
verification step on the four-request, 192-token workload. The final kernel
matched the previous monokernel bit for bit on its convolution history,
recurrent snapshots, FP8 values, and scales across 64 verification rounds
per GPU.

The latencies below compare warmed graphs over the same inputs on each GPU
with AITER absent. Each sample replays each graph 12 times with 128 calls per
graph. The table reports the median of seven samples per shape.

| Mode | Requests | Positions per request | GPU 0 before → after | GPU 1 before → after |
| --- | ---: | ---: | ---: | ---: |
| Decode | 1 | 1 | 6.14 → 6.05 µs | 6.18 → 6.01 µs |
| Decode | 8 | 1 | 10.30 → 10.26 µs | 10.80 → 10.49 µs |
| Verify | 1 | 1 | 7.24 → 5.79 µs | 7.29 → 5.80 µs |
| Verify | 1 | 8 | 24.42 → 20.51 µs | 24.39 → 20.34 µs |
| Verify | 4 | 8 | 33.27 → 27.38 µs | 33.65 → 27.61 µs |
| Verify | 8 | 2 | 19.60 → 16.13 µs | 19.75 → 16.34 µs |
| Verify | 8 | 8 | 58.64 → 47.39 µs | 59.17 → 47.84 µs |

The TP2 DFlash2 serving comparison used the same segmented-attention recipe,
prompt lengths, and throughput metrics as above.

| Prompt tokens | Concurrent requests | Decode tok/s per request, before → after | Aggregate tok/s, before → after |
| ---: | ---: | ---: | ---: |
| 192 | 1 | 159.54 → 157.80 | 137.28 → 135.04 |
| 192 | 4 | 90.55 → 90.72 | 226.99 → 227.39 |
| 192 | 8 | 74.88 → 77.63 | 271.55 → 284.56 |
| 2,048 | 1 | 85.11 → 85.13 | 63.80 → 63.88 |
| 2,048 | 4 | 73.29 → 73.43 | 133.12 → 133.25 |
| 2,048 | 8 | 56.06 → 57.17 | 137.38 → 137.38 |

All four 96-token greedy smoke completions matched the preceding
monokernel. Mean acceptance length was 5.20 tokens in both runs of the
four-request, 192-token workload. The 96-question GSM8K sample at concurrency
four scored 61 before and 59 after. The serving trace recorded 192 FlyDSL
verification launches on each rank, along with segmented attention and the
DFlash2 selector.
