# RDNA4 Octave serving smoke

The production backend is `ROCM_OCTAVE_RDNA4`. It uses the promoted native RDNA4
runtime packaged in `vllm/v1/attention/ops/rocm_octave_rdna4`, independently of
the RDNA3/CDNA3 kernels. An `octave_*` KV dtype selects it automatically on
RDNA4; the explicit backend flag is available for testing.

## Verified configuration

Qwen/Qwen3.8-27B-FP8, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, ran with TP2 on two Radeon AI PRO
R9700 (`gfx1201`) GPUs. Each rank has 12 query heads, two KV heads, head size
256 and 64 RoPE dimensions. Weights are FP8; activations are BF16; KV uses
`octave_k3v4`. The sequence limit is 32768, the batched-token limit 8192, and
the request limit four.

The test uses full decode graphs with capture sizes 1/2/4 and Torch compilation
disabled (`mode=0`). Both native source families match the measured standalone
release byte for byte. Core vLLM ROCm extensions were reused from the existing
environment; the RDNA4 libraries were compiled independently with `hipcc`.

## Results

- **37 adapter tests pass**, checking all three cache formats, both activation dtypes,
  both cache strides, decode/prefill/mixed batches, packed store reconstruction,
  untouched slots, graph replay after query changes, and automatic selection.
- All six model responses are correct: four concurrent arithmetic questions
  return `4`, `8`, `5`, `12`; two recall requests return `ORCHID-729`.
- The recall prompt has 5681 tokens. The second request reuses 3264 cached
  tokens, matching the hybrid model's attention/Mamba page size.
- Decode graph capture completes, and the server shuts down with GPU memory
  released. The existing output handler prints `EngineDeadError` during the
  intentional SIGTERM teardown after requests complete; both workers exit
  gracefully.

This is functional serving coverage and same-packed-KV numerical validation.
Serving throughput, 256K model-context validation and broad task quality remain
unmeasured. Some FP8 linear and GDN kernels use
default configurations or compile during the first requests; no latency claim
is made from this smoke run.

## Run in this workspace

From the new branch checkout, `pixi run` finds the surrounding ROCm workspace.
Using `python -m` from this directory ensures the server imports this checkout
while preserving the SDK's AMD SMI path.

```bash
cd /home/ellm/jeff/rocm_workspaces/10/gfx1201/vllm-rdna4-octave-kv
HIP_VISIBLE_DEVICES=0,1 NCCL_PROTO=Simple pixi run ../.venv/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B-FP8 \
    --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
    --tensor-parallel-size 2 --language-model-only --dtype bfloat16 \
    --attention-backend ROCM_OCTAVE_RDNA4 --kv-cache-dtype octave_k3v4 \
    --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --enable-prompt-tokens-details --generation-config vllm --seed 0 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

For an installed copy of this branch, use `vllm serve` as shown in the
[rendered design documentation](../../../../../docs/design/rocm_octave.md).
The first native build requires `hipcc` on `PATH`; generated libraries live in
`VLLM_CACHE_ROOT/octave_rdna4`. Concurrent TP workers share a build lock.

```bash
../.venv/bin/python -m pytest -q tests/v1/attention/test_rocm_octave_rdna4.py
```

The smoke test uses `RemoteOpenAIServer` with real model weights, checks greedy
responses and cache reuse, then stops the server. GPU tests in this workspace
were serialized under locks for physical GPUs 0 and 1; strict KFD queue/memory
checks remained enabled. The idle busy-counter threshold was 10% because GPU 1
reported 9% with no KFD clients and only 57 MiB reserved. These runs measure
correctness only.

See [validation](validation.json), [responses](responses.json),
[adapter tests](adapter-tests.log), and [server log](serving-test.log).
