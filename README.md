# RDNA4 Octave KV validation

Source: `big-yellow-duck/vllm`, branch `rocm/rdna4-octave-kv-pr`, commit
`362b0485e827d872e4b263340ca68de2bd05dfc8`. The upstream PR contains two
signed-off commits over `be47189f37f6162991366ff6b91539f6e9d1f432`.
This branch contains evidence only; no generated binaries or model weights.

## Validation of the proposed commit

The November 16 focused ROCm run passed **101 tests** in 68.90 seconds.
All applicable repository pre-commit hooks passed. `git diff --check` passed.
A merge-tree check against upstream main at
`0067b98f7e62c774ea4acd300eea743aed832aaf` had no conflicts; this is a merge
check, not runtime validation of the combined tree.

Reproduce from a prepared ROCm environment with the proposed source imported:

```bash
uv run --no-project .venv/bin/python -m pytest -q \
  tests/kernels/attention/test_rocm_octave.py \
  tests/models/qwen4_exp/test_qsa_amd.py
uv run --no-project .venv/bin/pre-commit run \
  --from-ref be47189f37 --to-ref HEAD
```

The local run used the workspace's Pixi activation for HIP headers and
AMD-SMI discovery, set `PYTHONPATH` to the proposed checkout and
`$ROCM_PATH/share/amd_smi`, and held the GEAK GPU lock. An initial run without
the AMD-SMI path skipped every test; it is excluded from the passing result.

The November 12 serving runs imported this exact proposed commit on both
workers. Hardware was two Radeon AI PRO R9700 GPUs (gfx1201), PyTorch
2.13.0+rocm10.0.0, HIP 7.15.26333, AMD clang 23, TP2 and BF16 activations.
Common serving settings were a 65,536 model limit, 8,192 batched tokens,
8 sequences, 3 GiB KV per GPU, default FULL_AND_PIECEWISE graphs, and vision
enabled. QSA used PLE CPU offload.

| Model / cache | Reported KV tokens | GSM8K |
| --- | ---: | ---: |
| Flash-Next BF16 | 201,850 | 124/128 |
| Flash-Next Octave K3V4 | 498,801 | 124/128 |
| Qwen3.8-27B-FP8 Octave K3V4 | 300,980 | Text/vision/retrieval smoke only |

The QSA capacity ratio is **2.4711×**. GSM8K used five shots, temperature 0,
seed 42, thinking disabled, concurrency 4 and 1,024 maximum output tokens.
Both QSA arms had zero API errors, zero token-limit finishes and zero invalid
extractions; 126/128 extracted predictions matched. Both also completed
eight concurrent requests with zero errors or incomplete requests, at 1,076
actual prompt tokens each. Text, red/blue image recognition and long
retrieval smoke passed for dense and QSA serving.

The QSA checkpoint was
`ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular` at
`1e7b246d7185a7f725b9e1a74f3bd252a97cff0a`. Its implicit unit PLE scale was
supplied in a local sidecar. Dense serving used `Qwen/Qwen3.8-27B-FP8`.
`serving/` preserves setup commands, worker inspection, smoke responses,
capacity and scored prediction summaries. `summary.json` collects counts.

The prior build logs cover gfx1200 and gfx1201; gfx1200 has **no runtime
validation**. Package staging and an sdist included both HIP sources.
The existing MI300 attention job selects the Octave test file, but it
cannot exercise RDNA4 dispatch. RDNA4 validation was local.

## Historical results, from earlier commits

These measurements provide context; they were not rerun on the proposed
commit and are not performance claims for it.

`historical/dense-gsm8k.md` records the full 1,319-question GSM8K evaluation
at `659ac4e339e2fcb192ee39e147ffcb3f48cfe052`: BF16 1,273 correct, FP8
1,275 and Octave K3V4 1,275. Token-limit finishes were 3/5/5 and remained
in the scores; no API errors or invalid extractions occurred. The file
documents matched settings, model revision, excluded failed attempts and
the exact scoring protocol. These observations do not prove a general
quality improvement.

`historical/qsa-throughput.md` and its JSON record a single sequential
BF16/Octave pair at `e36f0ca0d353ffa6c4c4ae0e039d91472e032474` with the
same QSA checkpoint and 3 GiB/GPU budget. The synthetic agent workload used
eight turns, 2,048 prompt tokens and 512 output tokens per turn.

| Concurrent streams | BF16 output tok/s | Octave output tok/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 27.99 | 35.21 | +25.8% |
| 2 | 50.45 | 40.50 | -19.7% |
| 4 | 112.77 | 64.92 | -42.4% |
| 8 | 101.95 | 106.01 | +4.0% |

Both arms completed 120/120 requests without errors or preemptions. Both
had JIT warnings during the measured window; Octave had recovered allocator
warnings at eight streams. This was one pair, without repeat statistics.
The results do not support a blanket throughput improvement.

The subsequent experimental branch `rocm/rdna4-octave-qsa-tuning-review`
at `3e63df69c5e5d29651a74337175d4c7a2ac72927` is excluded. Its main sweep
lost 39.3% at one stream versus the original Octave implementation, and the
reverse-order repeat reproduced a 39.4% loss. Isolated kernel wins did not
establish a general serving win.

## Review notes

The patch preserves the existing backend/CLI, wire formats and causal
metadata. RDNA4 code is isolated from RDNA3/CDNA3 kernels. Geometry reuses
the existing Python format definition. The two native implementations
retain a specialized D256/R64 dense path and a general geometry/QSA path.
Their size and first-use compilation are the main maintenance tradeoffs.

`hipcc` is needed on cold start; source is packaged and compilation occurs
before graph capture. Dense split scratch can reserve up to 2 GiB/GPU
outside the explicit KV budget. Sliding windows, ALiBi, sinks, soft caps,
unequal QK/V geometry and fused output quantization are unsupported.
No end-to-end RDNA4 MTP or dedicated sync-check serving result is claimed.

The required repository `pr-checklist` was applied: design fit, correctness,
CI selection, docs, hooks, provenance, duplicate work, and limitations.
Human line-by-line review remains necessary before marking the draft ready.

Duplicate checks on November 16 read RFC #59773 and its comments and searched
open PRs for `octave`, `octave rdna4`, `gfx1200`, `gfx1201` and
`59773 in:body`. No open PR implements these independent gfx12 kernels.
Related #65696 enables gfx1151 using RDNA3 kernels; #67635 refactors existing
dispatch and #67660 fixes prefill device selection. The RFC also describes
planned gfx1200 work by another contributor. This draft identifies the
overlap without claiming those changes are integrated.

AI assistance: OpenAI Codex assisted with implementation, validation,
result collection and drafting. No claim of completed human review is made.

## Additional BF16/FP8 serving comparisons

The [dense comparison report](dense-comparison/README.md) adds audited historical
BF16 Triton and FP8 segmented Triton serving results, packed-cache size calculations,
ordinary Triton FP8 capacity evidence, per-request measurements and a summary chart.
These are earlier development measurements; no new GPU benchmark is claimed.
