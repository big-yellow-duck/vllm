# RDNA4 segmented attention: 8192-query support and persistent cache

Validation date: 2026-09-24.

The serving measurements below were obtained on `perf/rdna4-optimized-stack`.
The segmented-attention changes were ported independently to
`perf/rdna4-prefill-autotune-splitkv` from commit
[`03f61647b1`](https://github.com/big-yellow-duck/vllm/commit/03f61647b14bfeb08375a0ab7caf083fa9a25775).
The throughput numbers are not measurements of the independent feature branch;
the combined stack also contains other RDNA4 optimizations.

Segmented attention now supports query chunks through 8192 tokens. The startup
tuner includes that query bucket, plus 32768- and 131072-token context buckets
for verification and prefill. These intermediate contexts remain eligible when
the model's 262144-token maximum is pruned by the KV-memory budget.

When either the target or draft explicitly selects `ROCM_SEGMENTED_ATTN`,
`VllmConfig` sets `TRITON_CACHE_DIR` before workers start. Its default is
`$VLLM_CACHE_ROOT/rocm_segmented_attention/triton`; an explicit
`TRITON_CACHE_DIR` is preserved. This stack uses `InductorStandaloneAdaptor`
(PyTorch 2.13, `VLLM_USE_STANDALONE_COMPILE=True`), whose existing cache
initialization already preserves this environment variable. No compiler-interface
patch is needed or retained. The legacy `InductorAdaptor` has different cache
redirection behavior and is not the compiler used in this validation.
Target and draft kernels, including Triton autotuning artifacts, consequently
use the same persistent location across engine starts. The segmented launch
winner tables remain JSON files under `$VLLM_CACHE_ROOT/rocm_segmented_attention`.
Changing kernel/tuner source intentionally invalidates old winner tables.

## Serving configuration

The available machine exposes **two** AMD Radeon AI PRO R9700 GPUs. This is TP2,
not the previous TP4 result. The user approved the cached target
`Qwen/Qwen3.8-27B-FP8` and draft `z-lab/Qwen3.8-27B-DFlash2`.
The remaining settings match
[`serve-qwen3.8-w7b-fp8.bash`](../../serve-qwen3.8-w7b-fp8.bash):

- `max_num_seqs=8`, `gpu_memory_utilization=0.92`, TP2.
- `max_num_batched_tokens=8192`, FP8 KV, `load_format=fastsafetensors`.
- Segmented attention for both target and draft; DFlash with five proposals.
- Qwen3 reasoning parser, Qwen3 XML tool parser, automatic tool choice,
  and multimodal encoder TP mode `data`.
- Explicit maximum context length: 131072.

The first attempt retained the model-default 262144-token context. It completed
tuning and graph capture but failed the final capacity check: 5.66 GiB KV was
required per GPU, versus 4.18 GiB available. The serving script now caps the
context at 131072, enough for the 128000-token input and 1024-token output.

## Validation method

The smoke client sends exactly 128000 input token IDs and requests 1024 output
tokens with temperature zero, seed 42, and `ignore_eos=True`. The synthetic
prompt repeats a sky/grass sentence and ends by asking for an explanation of
sky color and plant growth. This is a reproducible smoke workload, not a broad
model-quality evaluation or a reproduction of the user's original prompt.

The client saves streamed output token IDs and text, API usage, and Prometheus
counter snapshots before and after each request. Decode throughput is
`(output tokens - first chunk tokens) / (last token time - first token time)`.
This handles speculative chunks containing multiple tokens. Request throughput
includes prompt processing; decode throughput excludes it.

Acceptance is computed from request counter deltas:
`1 + accepted draft tokens / speculative rounds`. The one is the target-supplied
token. With five proposals, the value must be between one and six. The client
checks that per-position accepted counts sum to total accepted drafts, accepted
counts do not exceed proposals, streamed token count equals API usage, and
accepted tokens plus rounds reconcile with the final output count, allowing
for the initial prefill token and final-round truncation.

Artifacts and the runnable serving/client scripts are under
`/workspace/segmented-validation/` on the validation machine.

## Focused checks

The focused attention/workspace/cache suite passed 92 cases initially. Its only
failure was an old test treating 4097 queries as unsupported; after moving that
boundary to 8193, the case passed. Both 8192-query FP8 cases (head dimensions
128 and 256) matched the dense reference within the existing tolerance.
Two workload-generation tests passed after adding intermediate long contexts.
Four cache cases cover target selection, draft-only selection, explicit cache
overrides, and successive target/draft/restart standalone compiler initialization.
Ruff check, Ruff format, and `git diff --check` passed.

## Startup tuning

The initial 262144-context startup tuned 123 target workloads in 441.37 seconds and 42 draft workloads
in 93.40 seconds, with zero failures. Work was divided between two TP ranks.
The target's `(batch=1, queries=8192, context=131072)` bucket was measured and
saved successfully. An earlier incomplete tuning run overlapped GPU tests;
its records were moved aside and were not used for these measurements.

With `max_model_len=131072`, the same 123 target buckets were remeasured in
83.86 seconds and the 42 draft buckets in 9.14 seconds, reusing compiled Triton
kernels. Both tables again completed with zero failures.

## Initial serving measurements

| Request | Input tokens | Output tokens | First token | Decode tokens/s | Entire request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold prompt | 128000 | 1024 | 61.766 s | 83.226 | 74.058 s |
| Cached prefix | 128000 | 1024 | 1.381 s | 83.123 | 13.689 s |

Both requests recorded 263 speculative rounds, 1315 proposed tokens, and 763
accepted draft tokens. Mean acceptance length was **3.90114**
(`1 + 763 / 263`), with **58.0228%** draft-token acceptance. Position counts
were `[227, 188, 147, 116, 85]`; their sum is exactly 763. Accounting gives
`1 + 263 + 763 = 1027`, with three tokens truncated by the 1024-token output
limit. API usage and streamed token IDs both report exactly 1024 output tokens.
The cold and cached requests have identical output token IDs, SHA-256
`76b30130df0670e97b1d116badcb6d4bb6d1a6d58395e6cf9f1f0503fdbf5961`
(using the client's JSON token-list serialization). Output text starts with an
explanation of Rayleigh scattering and then discusses plant growth.

The startup warning about an unidentified draft KV group did not prevent
speculative generation. These request counters demonstrate real draft proposals
and accepted tokens under standard greedy rejection sampling; no synthetic
acceptance settings were enabled. No separate non-speculative output-equivalence
or broad quality evaluation is claimed.

## Engine restart cache verification

A full stop/start with identical serving settings loaded all 123 target records
and all 42 draft records on both TP ranks, with `missing=0`, `tuned=0`, and
`failed=0`. Each target load took 0.01–0.02 seconds; each draft load took 0.01
seconds. The tuning JSON files were byte-identical before and after restart.
The target retained `max_query_len=8192`, `causal=True`; the draft retained
`max_query_len=6`, `causal=False`. Six draft query positions correspond to five
proposals plus the conditioning position.

The reused target compilation took 2.32 seconds; draft head and selector
compilation reuse took 0.07 and 0.02 seconds. These are logged compilation
measurements, not complete engine startup times. See `server-restart.log`,
`cache-before-restart.json`, and `cache-after-restart.json` in the artifact
directory for the cache evidence.

## Final measurements after restart

| Request | Input tokens | Output tokens | First token | Decode tokens/s | Entire request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold prompt after restart | 128000 | 1024 | 54.966 s | 82.966 | 67.297 s |
| Cached prefix after restart | 128000 | 1024 | 1.381 s | **82.952** | 13.714 s |

All four requests produced identical output token IDs and identical request-wide
acceptance counters: 263 rounds, 1315 proposals, 763 accepted draft tokens,
**3.90114 mean acceptance length**, and **58.0228%** draft-token acceptance.
The final cached request delivered 74.667 tokens/s including its 1.381-second
first-token delay. The headline 82.952 tokens/s measures final streamed decode
output, excluding prompt processing and rejected proposals.

First requests still emitted JIT-monitor warnings for
`_segmented_attention_stage` and `_prepare_dflash_inputs_kernel`; this change
establishes stable cache placement and winner-table reuse, not exhaustive
startup coverage of every runtime specialization. The measured restart did not
retune either segmented attention table.

These results are on **2 × R9700, TP2, five speculative tokens**, so they do not
reproduce or supersede the earlier 100+ tokens/s observations on four GPUs.
The test server was stopped after validation; the patched serving script is
ready to launch it again.

## Config-only cache setup verification

The final patch leaves `compiler_interface.py` identical to the repository
baseline. Cache setup is confined to `VllmConfig`; the active standalone
compiler already honors `TRITON_CACHE_DIR`.

A fresh engine start with this simpler patch loaded 123 target and 42 draft
records on each GPU, with zero retuning and zero failures. All persistent
winner-table files were byte-identical to the previous validation. Four cache
regression cases and Ruff checks passed.

Repeating the 128000-input/1024-output smoke test measured 82.869 decode tokens/s
for the cold prompt and **82.913 decode tokens/s** with the cached prefix.
Cached first-token latency was 1.379 seconds and total request time was 13.718
seconds. Both requests reproduced the earlier token IDs and acceptance counts:
263 rounds, 763 accepted drafts, and **3.90114 mean acceptance length**.
Artifacts are `server-config-only.log`, `config-only-{cold,warm}-request.json`,
and `cache-config-only.json` under `/workspace/segmented-validation/`.
The test server was stopped afterward.

## Independent feature branch validation

The port to `perf/rdna4-prefill-autotune-splitkv` passed 83 focused tests covering
the dedicated segmented-attention suite, cache configuration, and workspace
reservation. The tests imported this branch's checkout at
`/workspace/vllm-rdna4-prefill`; they did not import the combined stack's Python
modules. The segmented kernel, tuner, backend, and dedicated test suite match
`03f61647b1`. No FlyDSL GEMM or all-reduce changes were included in the port.
