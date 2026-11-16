# GSM8K accuracy: Qwen/Qwen3.8-27B-FP8 KV cache comparison

## Results

| KV cache | Attention backend | Correct / total | Exact match | Δ vs BF16 | Invalid extraction | API errors | `length` finishes |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 (`auto`) | TRITON_ATTN | 1,273 / 1,319 | 96.5125% | baseline | 0 | 0 | 3 |
| FP8 | TRITON_ATTN | 1,275 / 1,319 | 96.6641% | +0.1516 pp | 0 | 0 | 5 |
| Octave K3V4 | ROCM_OCTAVE_RDNA4 | 1,275 / 1,319 | 96.6641% | +0.1516 pp | 0 | 0 | 5 |

Every arm scored all 1,319 questions. API errors and invalid answer extraction were both zero. Outputs finishing at the 1,024-token limit were retained and scored by the same exact-match extraction as other outputs; their counts are shown above.

Paired question results:

| Comparison | First correct → second wrong | First wrong → second correct | Same correctness status | Extracted predictions differ |
|---|---:|---:|---:|---:|
| BF16 → FP8 | 3 | 5 | 1,311 / 1,319 | 11 |
| BF16 → Octave | 6 | 8 | 1,305 / 1,319 | 21 |
| FP8 → Octave | 8 | 8 | 1,303 / 1,319 | 21 |

“Same correctness status” counts questions both answered correctly or both answered incorrectly; it does not mean the extracted predictions are equal. Prediction disagreements use the extracted integer, with invalid extraction represented separately (none occurred).

Full request start/end times (UTC) and client elapsed time:

| Arm | Started | Finished | Elapsed |
|---|---|---|---:|
| FP8 | 2026-11-10 06:22:34Z | 2026-11-10 06:40:49Z | 1,094.8 s |
| Octave | 2026-11-10 06:51:47Z | 2026-11-10 07:15:00Z | 1,393.2 s |
| BF16 retry | 2026-11-10 07:20:24Z | 2026-11-10 07:42:06Z | 1,302.7 s |

## Protocol and provenance

- Model: `Qwen/Qwen3.8-27B-FP8`, pinned to revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` for all arms.
- Evaluation: existing vLLM GSM8K harness prompt/scorer in `vllm-rdna4-octave-kv/tests/evals/gsm8k/gsm8k_eval.py`; official cached train/test JSONL, five training examples and the full 1,319-question test set. Score is exact match of the scorer’s extracted last integer.
- Request format: OpenAI chat completions with `temperature=0`, `seed=42`, model chat template, and `chat_template_kwargs={"enable_thinking": false}`. `top_p` and `top_k` were omitted; pinned generation config supplies top_p 0.95 and top_k 20. Eight concurrent requests.
- Pilot: first eight prompts at 256 output tokens had one `length` finish. Repeating those eight with 1,024 tokens had zero length finishes. All three full arms therefore used 1,024 output tokens. The protocol selects the harness-supported chat mode with thinking disabled and concurrency eight, instead of the evaluator function’s default completion mode and unrestricted client concurrency; it reuses the same prompt builder and answer scorer.
- Matched serving settings: TP2 on GPUs 0 and 1; BF16 model activations; max model length 65,536; max batched tokens 8,192; exactly 8,393,906,176 KV-cache bytes per GPU. Vision remained enabled. Default Torch compilation and FULL_AND_PIECEWISE CUDA graphs remained enabled. Worker inspection verified these settings on both ranks and recorded the resolved backend/cache dtype.
- Runtime: Torch `2.13.0+rocm10.0.0`, ROCm `7.15.26333`, gfx1201. All servers loaded `vllm-rdna4-octave-kv` source from HEAD `659ac4e339e2fcb192ee39e147ffcb3f48cfe052`. The valid FP8 run used the original launcher with dual GPU locks and the AMD-SMI Python path. After the Octave compiler-header failure, the Octave run and BF16 retry used the established locked Pixi wrapper, which supplies the HIP C++ headers. The Octave HIP kernel preflight passed before its full run. Live inspection recorded matching model, Torch/ROCm build versions, generation settings, and the shared serving configuration across all three arms; the launcher-environment correction is preserved in the diagnostic history.
- All valid arms used the same fixed client `run_accuracy.py`, SHA-256 `76376eca7366b9334f5a44b35451202c7ae54e05c50ef7075a88ae4005108c13`. It acquires a semaphore before starting each HTTP request timeout. A CPU-only local check with six requests, concurrency two, 150ms request duration, and a 250ms timeout passed all six although aggregate queue time exceeded the timeout.
- Valid order: FP8 → Octave K3V4 → BF16 retry. Exact request rows, outputs, extraction, finish reason, token count, and error are retained in each arm’s `full.json`; worker configuration, metrics and server logs are alongside them. `paired-comparison.json` contains aggregate and paired details; `compare_results.py` regenerates it and per-arm runner hashes.

## Excluded attempt and limitations

The first BF16 attempt is retained at `arms/bf16/initial-timeout-attempt.json` as a client diagnostic only. An aiohttp total timeout originally included time queued behind the eight-connection limit, causing 710 tasks to time out before being sent. Those timeout results are excluded from all accuracy counts; the corrected semaphore ordering was CPU-validated and the full BF16 arm was rerun.

The initial Octave server launch also failed before evaluation because a custom launcher bypassed Pixi’s HIP C++ standard-header environment. That failure is preserved under `arms/octave-compile-failure/`; no responses from it were scored. The successful retry used the standard locked Pixi wrapper and a passing HIP compile preflight.

These are single greedy runs on one model revision and the full 1,319-question GSM8K test set. Small differences (two additional correct answers for FP8/Octave versus BF16) should be read as observed scores, not as proof of a general accuracy improvement. The benchmark uses integer extraction and does not assess solution reasoning beyond the extracted answer. The FP8 model emits a startup warning that its quantization metadata does not provide KV scales, so vLLM uses scale 1.0; this is part of the tested implementation.
