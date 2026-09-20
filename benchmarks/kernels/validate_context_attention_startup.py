# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 ROCm attention startup, persistent tuning, and output validation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

if os.environ.get("CONTEXT_TUNING_FORBID") == "1":
    from vllm.v1.attention.ops import (
        prefix_prefill_tuning,
        segmented_prefill_tuning,
    )

    def forbidden(*args, **kwargs):
        raise RuntimeError("Warm engine attempted to benchmark context attention")

    prefix_prefill_tuning._tune_workload = forbidden
    segmented_prefill_tuning._tune_workload = forbidden


if os.environ.get("CONTEXT_VALIDATE_NUMERICS") == "1":
    import torch

    from vllm.v1.attention.ops import (
        chunked_prefill_paged_decode,
        prefix_prefill,
        prefix_prefill_tuning,
    )

    original_context = prefix_prefill.context_attention_fwd

    @torch.inference_mode()
    def checked_context(*args, **kwargs):
        result = original_context(*args, **kwargs)
        if kwargs.get("_launch_config") is not None:
            return result
        output = kwargs["o"]
        reference = torch.empty_like(output)
        baseline_kwargs = dict(
            kwargs, o=reference, _launch_config=prefix_prefill_tuning._DEFAULT
        )
        original_context(*args, **baseline_kwargs)
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
        difference = output.float() - reference.float()
        relative_l2 = difference.norm() / reference.float().norm()
        print(
            f"NUMERICS rank_device={output.device} Q={kwargs['max_input_len']} "
            f"max_abs={difference.abs().max().item():.8f} "
            f"relative_l2={relative_l2.item():.8f}",
            flush=True,
        )
        return result

    prefix_prefill.context_attention_fwd = checked_context
    chunked_prefill_paged_decode.context_attention_fwd = checked_context


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tp-size", type=int, default=int(os.environ.get("TP_SIZE", "2"))
    )
    args = parser.parse_args()
    backend = os.environ.get("ATTENTION_BACKEND", "ROCM_ATTN")
    scheduler_args = {}
    if value := os.environ.get("MAX_MODEL_LEN"):
        scheduler_args["max_model_len"] = int(value)
    if value := os.environ.get("MAX_BATCHED_TOKENS"):
        scheduler_args["max_num_batched_tokens"] = int(value)
    if value := os.environ.get("MAX_NUM_SEQS"):
        scheduler_args["max_num_seqs"] = int(value)
    start = time.monotonic()
    llm = LLM(
        model="Qwen/Qwen3.8-27B-FP8",
        tensor_parallel_size=args.tp_size,
        language_model_only=True,
        load_format="fastsafetensors",
        linear_backend="triton",
        attention_backend=backend,
        kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "auto"),
        gpu_memory_utilization=0.90,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        **scheduler_args,
    )
    startup = time.monotonic() - start
    results = []
    max_model_len = scheduler_args.get("max_model_len")
    lengths = (
        (33, 129, 513)
        if max_model_len is None
        else sorted(
            {max(1, min(length, max_model_len - 8)) for length in (33, 129, 513)}
        )
    )
    for length in lengths:
        prompt = TokensPrompt(prompt_token_ids=[1000 + i % 9000 for i in range(length)])
        outputs = llm.generate(
            [prompt],
            SamplingParams(
                temperature=0.0, ignore_eos=True, max_tokens=8, detokenize=False
            ),
            use_tqdm=False,
        )
        results.append(
            {"input_len": length, "token_ids": outputs[0].outputs[0].token_ids}
        )
    cache_name = (
        "rocm_segmented_attention"
        if backend == "ROCM_SEGMENTED_ATTN"
        else "rocm_context_attention"
    )
    caches = {}
    for path in (Path(os.environ["VLLM_CACHE_ROOT"]) / cache_name).glob("*.json"):
        caches[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns,
            "records": len(json.loads(path.read_text())["records"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "backend": backend,
                "startup_s": startup,
                "outputs": results,
                "caches": caches,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"VALIDATION startup={startup:.2f}s output={args.output}", flush=True)


if __name__ == "__main__":
    main()
