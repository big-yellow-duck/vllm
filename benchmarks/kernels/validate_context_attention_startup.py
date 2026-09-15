# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 engine startup, persistent tuning cache, and greedy output validation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

if os.environ.get("CONTEXT_TUNING_FORBID") == "1":
    from vllm.v1.attention.ops import prefix_prefill_tuning

    def forbidden(*args, **kwargs):
        raise RuntimeError("Warm engine attempted to benchmark context attention")

    prefix_prefill_tuning._tune_workload = forbidden


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
    args = parser.parse_args()
    start = time.monotonic()
    llm = LLM(
        model="Qwen/Qwen3.8-27B-FP8",
        tensor_parallel_size=2,
        language_model_only=True,
        load_format="fastsafetensors",
        linear_backend="triton",
        attention_backend="ROCM_ATTN",
        kv_cache_dtype="auto",
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        max_num_batched_tokens=int(os.environ.get("MAX_BATCHED_TOKENS", "8192")),
        max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "32")),
        gpu_memory_utilization=0.90,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    startup = time.monotonic() - start
    results = []
    for length in (33, 129, 513):
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
    caches = {}
    for path in (Path(os.environ["VLLM_CACHE_ROOT"]) / "rocm_context_attention").glob(
        "*.json"
    ):
        caches[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns,
            "records": len(json.loads(path.read_text())["records"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"startup_s": startup, "outputs": results, "caches": caches}, indent=2
        )
        + "\n"
    )
    print(f"VALIDATION startup={startup:.2f}s output={args.output}", flush=True)


if __name__ == "__main__":
    main()
