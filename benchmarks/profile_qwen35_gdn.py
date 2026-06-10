# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile Qwen3.5 GDN layers on one 32K prefill + decode request.

This script intentionally runs one unprofiled warmup request before enabling
the vLLM torch profiler, so model/warmup kernels are excluded from the trace.
It also monkey-patches the Python GDN core boundary to record tensor metadata
and add profiler ranges around each GDN core call.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


GDN_CALLS: list[dict[str, Any]] = []


def _tensor_meta(t: torch.Tensor | None, include_values: bool = False) -> dict[str, Any] | None:
    if t is None:
        return None
    meta: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "stride": list(t.stride()),
        "is_contiguous": t.is_contiguous(),
    }
    if include_values and t.numel() <= 128:
        meta["values"] = t.detach().cpu().tolist()
    return meta


def _phase_from_metadata(attn_metadata: Any) -> str:
    if attn_metadata is None:
        return "warmup"
    if getattr(attn_metadata, "num_prefills", 0) > 0:
        if getattr(attn_metadata, "num_decodes", 0) > 0:
            return "mixed_prefill_decode"
        return "prefill"
    if getattr(attn_metadata, "num_decodes", 0) > 0:
        return "decode"
    if getattr(attn_metadata, "num_spec_decodes", 0) > 0:
        return "spec_decode"
    return "unknown"


def install_gdn_patch(max_records: int) -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    original_forward_core = QwenGatedDeltaNetAttention._forward_core
    original_decode_non_spec = QwenGatedDeltaNetAttention._forward_core_decode_non_spec

    def patched_forward_core(self, mixed_qkv, b, a, core_attn_out):
        forward_context = qwen_gdn_linear_attn.get_forward_context()
        attn_metadata = None
        if forward_context.attn_metadata is not None:
            attn_metadata = forward_context.attn_metadata[self.prefix]
        phase = _phase_from_metadata(attn_metadata)
        range_name = f"gdn_core/{phase}/{self.prefix}"
        if len(GDN_CALLS) < max_records:
            GDN_CALLS.append(
                {
                    "kind": "core",
                    "phase": phase,
                    "layer": self.prefix,
                    "gdn_prefill_backend": getattr(self, "gdn_prefill_backend", None),
                    "mixed_qkv": _tensor_meta(mixed_qkv),
                    "b": _tensor_meta(b),
                    "a": _tensor_meta(a),
                    "core_attn_out": _tensor_meta(core_attn_out),
                    "metadata": _metadata_summary(attn_metadata),
                }
            )
        with torch.profiler.record_function(range_name):
            return original_forward_core(self, mixed_qkv, b, a, core_attn_out)

    def patched_decode_non_spec(
        self, mixed_qkv, b, a, core_attn_out, attn_metadata
    ):
        range_name = f"gdn_decode_non_spec/{self.prefix}"
        if len(GDN_CALLS) < max_records:
            GDN_CALLS.append(
                {
                    "kind": "decode_non_spec_fastpath",
                    "phase": "decode",
                    "layer": self.prefix,
                    "mixed_qkv": _tensor_meta(mixed_qkv),
                    "b": _tensor_meta(b),
                    "a": _tensor_meta(a),
                    "core_attn_out": _tensor_meta(core_attn_out),
                    "metadata": _metadata_summary(attn_metadata),
                }
            )
        with torch.profiler.record_function(range_name):
            return original_decode_non_spec(
                self, mixed_qkv, b, a, core_attn_out, attn_metadata
            )

    QwenGatedDeltaNetAttention._forward_core = patched_forward_core
    QwenGatedDeltaNetAttention._forward_core_decode_non_spec = patched_decode_non_spec


def _metadata_summary(attn_metadata: Any) -> dict[str, Any] | None:
    if attn_metadata is None:
        return None
    fields = [
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "num_spec_decodes",
        "num_spec_decode_tokens",
        "num_actual_tokens",
    ]
    out = {name: int(getattr(attn_metadata, name)) for name in fields}
    for name in [
        "has_initial_state",
        "non_spec_query_start_loc",
        "prefill_query_start_loc",
        "chunk_offsets",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
        "prefill_has_initial_state",
    ]:
        out[name] = _tensor_meta(getattr(attn_metadata, name, None), include_values=True)
    chunk_indices = getattr(attn_metadata, "chunk_indices", None)
    out["chunk_indices"] = _tensor_meta(chunk_indices)
    if chunk_indices is not None and chunk_indices.numel() > 0:
        out["chunk_indices_head"] = chunk_indices[: min(8, chunk_indices.shape[0])].cpu().tolist()
    return out


def _summarize_calls() -> dict[str, Any]:
    phases = Counter(call["phase"] for call in GDN_CALLS)
    layers = sorted({call["layer"] for call in GDN_CALLS})
    first_by_phase: dict[str, dict[str, Any]] = {}
    for call in GDN_CALLS:
        first_by_phase.setdefault(call["phase"], call)
    return {
        "num_recorded_calls": len(GDN_CALLS),
        "calls_by_phase": dict(phases),
        "layers_seen": layers,
        "first_call_by_phase": first_by_phase,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/Competitions/PRA26/Qwen3.5-9B")
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=2)
    parser.add_argument("--warmup-input-len", type=int, default=128)
    parser.add_argument("--warmup-output-len", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32776)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gdn-prefill-backend", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to benchmark_results/gdn_profile_<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or f"benchmark_results/gdn_profile_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "torch_profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VLLM_USE_V1", "1")
    install_gdn_patch(args.max_records)

    from vllm import LLM, SamplingParams

    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir.resolve()),
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_dump_cuda_time_total": True,
        "torch_profiler_use_gzip": True,
    }

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        profiler_config=profiler_config,
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )

    rng = np.random.default_rng(args.seed)
    warmup_prompt = {
        "prompt_token_ids": rng.integers(10000, size=args.warmup_input_len).tolist()
    }
    measured_prompt = {
        "prompt_token_ids": rng.integers(10000, size=args.input_len).tolist()
    }

    warmup_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.warmup_output_len,
        detokenize=False,
    )
    params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=False,
    )

    print("Running unprofiled warmup request...")
    llm.generate([warmup_prompt], sampling_params=warmup_params, use_tqdm=False)
    torch.cuda.synchronize()
    GDN_CALLS.clear()

    print("Running profiled 32K request...")
    start = time.perf_counter()
    llm.start_profile()
    outputs = llm.generate([measured_prompt], sampling_params=params, use_tqdm=False)
    llm.stop_profile()
    torch.cuda.synchronize()
    latency_s = time.perf_counter() - start

    generated = len(outputs[0].outputs[0].token_ids)
    metadata_path = out_dir / "gdn_metadata.json"
    summary = {
        "args": vars(args),
        "latency_s_including_profiler_overhead": latency_s,
        "generated_tokens": generated,
        "torch_profiler_dir": str(trace_dir.resolve()),
        "gdn_summary": _summarize_calls(),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(GDN_CALLS, f, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote GDN metadata to {metadata_path}")


if __name__ == "__main__":
    main()
