# RDNA4 short-prefill Triton prototype

A standalone, two-kernel BF16 attention prototype for one request with HQ=12,
HKV=2, D=256, scale=1/16, causal attention, and physical page size 784.
It reads the ROCM_ATTN packed K/head-major V cache directly, combines query
positions and GQA heads in each matrix tile, splits the cached prefix, handles
fresh dense current K/V with a causal mask, and merges FP32 partial results.

The selected **Q2/C65536** configuration reaches **231.18 us / 90.7% of the
ideal roof** under read-only cache eviction, including both kernels. The
original **Q32/C8192** probe reaches **78.96 us / 34.1%**, so its 80% target
was not reached. See the [full report](../../../ATTENTION_TRITON_ROOFLINE_PROTOTYPE.md)
for matched baselines, the eviction-method change, numerical checks, and limits.

## Entry point

```python
from kernel import prepare_attention

run, output = prepare_attention(
    query,
    key_cache,
    value_cache,
    current_key,
    current_value,
    block_table,
    context=65536,
)
run()  # Warm up before graph capture.
```

`prepare_attention` selects only the two validated shapes:

| Q | Prefix C | Query tile | KV tile | QK reduction tile | Splits | Warps | Stages |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 2 | 65536 | 16 | 32 | 256 | 32 | 4 | 1 |
| 32 | 8192 | 32 | 64 | 64 | 8 | 4 | 2 |

Q is contiguous `[Q,12,256]`; current K/V are contiguous `[Q,2,256]`.
All are BF16. Packed K is `[pages,2,32,784,8]`; V is
`[pages,2,256,784]`. The cache's outer strides may include interleaved K/V
backing. The contiguous GPU block table has shape `[1,pages]` and int32
physical page indices. Inputs must be on the same GPU.

The factory allocates output and workspace once. Retain `run`, its inputs,
and `output` while replaying a graph; changing tensor contents is supported.
There are no GPU allocations or CPU/GPU synchronization inside the two-kernel call.
The configurable `make_call` is for experiments; it does not constitute a
validated general-purpose attention API.

## Reproduce in this workspace

```bash
cd /app/tps-rdna4-qwen38-tp2/results/triton-prefill-hour-20260915
source ./env.bash
python /app/vllm-perf-rdna4-prefill-autotune-splitkv/benchmarks/kernels/rdna4_prefill_prototype/validate.py --output reproduced-q2.json
python /app/vllm-perf-rdna4-prefill-autotune-splitkv/benchmarks/kernels/rdna4_prefill_prototype/validate.py --queries 32 --context 8192 --output reproduced-q32.json
python /app/vllm-perf-rdna4-prefill-autotune-splitkv/benchmarks/kernels/rdna4_prefill_prototype/robustness.py
./run-winner-profiles.bash
```

`env.bash` activates the requested
`/app/vllm-rdna4-flydsl_fp8-flydsl_ar/.venv` and pins the local source trees.
The kernel itself imports only Torch and Triton. Matched baseline validation
also requires the patched AITER and vLLM checkouts.

`validate.py` checks every output against FP32 attention, including shuffled
pages, poisoned cached tails, independently changed fresh K/V, and graph replay.
It then measures five rounds of 30 graph samples under read eviction, write
eviction, and reuse. All timed outputs are checked again after each round.
`tune.py` retains the launch-search interface; `experiments/` contains rejected
alternatives. Historical search JSONs and kernel snapshots are in the result
directory. `calibrate.py` and `check_eviction.py` diagnose eviction effects.

## Scope

This is an isolated benchmark prototype, not a vLLM backend integration.
It does not implement ragged batching, FP8, attention sinks, ALiBi, sliding
windows, or other unrepresented backend features. Production promotion needs
routing, workspace lifecycle integration, engine benchmarks, and model tests.
The write-eviction Q2 result is worse than AITER; the report preserves that
result and does not claim a win under every cache condition.
