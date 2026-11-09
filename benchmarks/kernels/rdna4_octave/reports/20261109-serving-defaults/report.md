# RDNA4 Octave: default compilation and vision

Qwen3.8-27B-FP8 TP2 passes the serving regression with the full vision model and
vLLM's default Torch compilation and graph configuration. The earlier smoke
command's reduced graph settings and language-only flag were conveniences for
that initial test; the RDNA4 runtime needs neither. No runtime changes were
required after the serving adapter in commit `a00ac1f3e2`.

## Results

- **37 adapter checks pass**, preserving the cache-store, numerical, mixed-batch,
  graph-replay and automatic-selection checks.
- **17/17 model responses are correct**, including **11 image responses**.
- The engine resolves `VLLM_COMPILE` with `FULL_AND_PIECEWISE`. Logs confirm
  piecewise graph captures at sizes 1/2/4/8 and full decode captures at 1/2/4.
- The vision encoder loads and uses its default Torch SDPA attention backend.
  Every image response reports nonzero image tokens.
- Repeated text recall reuses 3264 cached tokens on a 5681-token prompt.

| Image check | Observed answer | Image tokens |
| --- | --- | ---: |
| Solid red PNG, 224×224 | red | 64 |
| Solid blue PNG, 448×336, same prompt | blue | 140 |
| Red replay after blue | red | 64 |
| Green circle PNG | circle | 81 |
| Green square PNG, same prompt | square | 81 |
| Two images, blue then red | blue, red | 204 |
| Full-resolution JPEG photograph | Pigeon | 6144 |
| Four concurrent image requests | red, blue, circle, square | 64/140/81/81 |

The paired inputs verify that answers follow the image content. The replay,
concurrent requests and two-image ordering also exercise multimodal cache and
batch handling. Four concurrent arithmetic requests and two long-prefix recall
requests run on the same server. Client retries are disabled by the existing
test helper.

Fixtures: [red PNG](red.png), [blue PNG](blue.png), [circle PNG](circle.png),
[square PNG](square.png), and the existing [pigeon JPEG](../../../../../tests/v1/ec_connector/integration/hato.jpg).
Complete [responses](responses.json), [validation and source hashes](validation.json),
[adapter log](adapter-tests.log) and [server log](serving-test.log) are preserved.

This validates basic execution and image content on the tested fixture set.
Broader vision/task accuracy, serving throughput, 256K model context and gfx1200
execution remain separate evaluations. The earlier standalone kernel campaign
retains its own broad geometry/performance results.

## Tested launch command

From this workspace's branch checkout:

```bash
cd /home/ellm/jeff/rocm_workspaces/10/gfx1201/vllm-rdna4-octave-kv
HIP_VISIBLE_DEVICES=0,1 NCCL_PROTO=Simple pixi run ../.venv/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B-FP8 \
    --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
    --tensor-parallel-size 2 --dtype bfloat16 \
    --attention-backend ROCM_OCTAVE_RDNA4 --kv-cache-dtype octave_k3v4 \
    --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --enable-prompt-tokens-details --generation-config vllm --seed 0
```

For an installed copy of this branch, use `vllm serve` with the same model and
flags. The command retains default compilation, graph sizes and vision support.
The native HIP JIT still requires `hipcc` for its first build; generated
libraries are reused under `VLLM_CACHE_ROOT/octave_rdna4`.

```bash
../.venv/bin/python -m pytest -q tests/v1/attention/test_rocm_octave_rdna4.py
```

The smoke test stops the server and verifies GPU memory release afterward.
The output handler logs `EngineDeadError` during intentional SIGTERM teardown,
after all requests have completed and both workers exit gracefully. Hardware
tests were serialized on physical GPUs 0 and 1 using the workspace GPU locks.
