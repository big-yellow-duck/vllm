# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone benchmark for the Qwen3.5 GDN core operator path."""

from __future__ import annotations

import argparse
import json
import time
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec


H = 16
HV = 32
K = 128
V = 128
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
NUM_HIDDEN_LAYERS = 32
NUM_GDN_LAYERS = 24


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def set_gdn_dims(
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel: int,
    num_hidden_layers: int,
    num_gdn_layers: int,
) -> None:
    global H, HV, K, V, CONV_KERNEL, KEY_DIM, VALUE_DIM, CONV_DIM
    global NUM_HIDDEN_LAYERS, NUM_GDN_LAYERS
    H = num_k_heads
    HV = num_v_heads
    K = head_k_dim
    V = head_v_dim
    CONV_KERNEL = conv_kernel
    KEY_DIM = H * K
    VALUE_DIM = HV * V
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM
    NUM_HIDDEN_LAYERS = num_hidden_layers
    NUM_GDN_LAYERS = num_gdn_layers


def load_gdn_dims(args) -> dict:
    config_path = args.config_json
    if config_path is None:
        candidate = Path(args.model) / "config.json"
        config_path = str(candidate) if candidate.exists() else None

    if config_path is None:
        return {
            "source": "defaults",
            "num_k_heads": H,
            "num_v_heads": HV,
            "head_k_dim": K,
            "head_v_dim": V,
            "conv_kernel": CONV_KERNEL,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_gdn_layers": NUM_GDN_LAYERS,
        }

    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    text_config = raw.get("text_config", raw)
    layer_types = text_config.get("layer_types") or []
    num_hidden_layers = int(text_config.get("num_hidden_layers", len(layer_types) or 0))
    num_gdn_layers = sum(
        1 for layer_type in layer_types if layer_type == "linear_attention"
    )
    if num_gdn_layers == 0 and num_hidden_layers:
        full_attention_interval = int(text_config.get("full_attention_interval", 0))
        if full_attention_interval > 0:
            num_gdn_layers = (
                num_hidden_layers - num_hidden_layers // full_attention_interval
            )

    return {
        "source": str(config_path),
        "hidden_size": int(text_config.get("hidden_size", 0)),
        "num_k_heads": int(text_config["linear_num_key_heads"]),
        "num_v_heads": int(text_config["linear_num_value_heads"]),
        "head_k_dim": int(text_config["linear_key_head_dim"]),
        "head_v_dim": int(text_config["linear_value_head_dim"]),
        "conv_kernel": int(text_config["linear_conv_kernel_dim"]),
        "num_hidden_layers": num_hidden_layers,
        "num_gdn_layers": num_gdn_layers,
        "mamba_ssm_dtype": text_config.get("mamba_ssm_dtype"),
        "dtype": text_config.get("dtype"),
    }


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    context_lens = [
        batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        for i in range(batch_spec.batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=int(seq_lens_cpu.max()),
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def create_bench_vllm_config(
    model_name: str,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> VllmConfig:
    model_config = ModelConfig(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        max_model_len=max_model_len,
    )
    model_config.hf_config.update(
        {
            "linear_num_key_heads": H,
            "linear_num_value_heads": HV,
            "linear_key_head_dim": K,
            "linear_value_head_dim": V,
            "linear_conv_kernel_dim": CONV_KERNEL,
        }
    )
    cache_config = CacheConfig(block_size=BLOCK_SIZE, cache_dtype="auto")
    cache_config.num_gpu_blocks = 4096
    cache_config.num_cpu_blocks = 0
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def tensor_meta(t: torch.Tensor | None, include_values: bool = False):
    if t is None:
        return None
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "stride": list(t.stride()),
        "is_contiguous": t.is_contiguous(),
    }
    if include_values and t.numel() <= 128:
        out["values"] = t.detach().cpu().tolist()
    return out


def build_vllm_config(args):
    cfg = create_bench_vllm_config(
        model_name=args.model,
        max_model_len=max(args.prefill_tokens + 8, args.decode_context + 8),
        max_num_seqs=max(args.batch_size, args.decode_batch_size),
        max_num_batched_tokens=max(args.prefill_tokens, args.decode_batch_size),
    )
    cfg.additional_config = {"gdn_prefill_backend": args.gdn_prefill_backend}
    return cfg


def build_layer(vllm_config, pool_size: int, state_dtype: torch.dtype):
    device = torch.device("cuda")
    conv_state_shape, ssm_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, num_spec=0
        )
    )
    conv_state = torch.randn(
        pool_size,
        *conv_state_shape,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.01
    ssm_state = torch.randn(
        pool_size,
        *ssm_state_shape,
        dtype=state_dtype,
        device=device,
    ) * 0.01
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.conv_kernel_size = CONV_KERNEL
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.activation = "silu"
    layer.A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    layer.dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    conv_weight = torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    ) * 0.01
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.01
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_non_spec",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def build_metadata(vllm_config, seq_lens: list[int], query_lens: list[int]):
    device = torch.device("cuda")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1,),),
            dtypes=(torch.float16,),
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch, BLOCK_SIZE, device)
    with set_current_vllm_config(vllm_config):
        return builder.build(common_prefix_len=0, common_attn_metadata=common)


def run_core(layer, meta, mixed_qkv, b, a, core_attn_out, use_custom_op: bool):
    ctx = types.SimpleNamespace(
        attn_metadata={PREFIX: meta},
        no_compile_layers={PREFIX: layer},
    )
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        if use_custom_op:
            qwen_gdn_linear_attn.qwen_gdn_attention_core(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                layer_name=PREFIX,
            )
        else:
            layer._forward_core(mixed_qkv, b, a, core_attn_out)


def bench_phase(
    name: str,
    layer,
    meta,
    num_tokens: int,
    warmups: int,
    iters: int,
    profile: bool,
    use_custom_op: bool,
):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    mixed_qkv = torch.randn(num_tokens, CONV_DIM, dtype=dtype, device=device) * 0.01
    b = torch.randn(num_tokens, HV, dtype=dtype, device=device) * 0.01
    a = torch.randn(num_tokens, HV, dtype=dtype, device=device) * 0.01
    core_attn_out = torch.empty(num_tokens, HV, V, dtype=dtype, device=device)

    for _ in range(warmups):
        run_core(layer, meta, mixed_qkv, b, a, core_attn_out, use_custom_op)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        run_core(layer, meta, mixed_qkv, b, a, core_attn_out, use_custom_op)
    end_evt.record()
    torch.cuda.synchronize()
    avg_ms = start_evt.elapsed_time(end_evt) / iters

    profile_table = None
    trace_file = None
    if profile:
        trace_file = f"{name}_trace.json.gz"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function(f"standalone_gdn_core/{name}"):
                run_core(layer, meta, mixed_qkv, b, a, core_attn_out, use_custom_op)
        profile_table = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=80
        )
        prof.export_chrome_trace(trace_file)

    return {
        "avg_cuda_event_ms": avg_ms,
        "inputs": {
            "mixed_qkv": tensor_meta(mixed_qkv),
            "b": tensor_meta(b),
            "a": tensor_meta(a),
            "core_attn_out": tensor_meta(core_attn_out),
        },
        "metadata": metadata_summary(meta),
        "profile_table": profile_table,
        "trace_file": trace_file,
    }


def metadata_summary(meta):
    out = {
        "num_prefills": meta.num_prefills,
        "num_prefill_tokens": meta.num_prefill_tokens,
        "num_decodes": meta.num_decodes,
        "num_decode_tokens": meta.num_decode_tokens,
        "num_actual_tokens": meta.num_actual_tokens,
        "has_initial_state": tensor_meta(meta.has_initial_state, include_values=True),
        "non_spec_query_start_loc": tensor_meta(
            meta.non_spec_query_start_loc, include_values=True
        ),
        "prefill_query_start_loc": tensor_meta(
            meta.prefill_query_start_loc, include_values=True
        ),
        "chunk_indices": tensor_meta(meta.chunk_indices),
        "chunk_offsets": tensor_meta(meta.chunk_offsets, include_values=True),
        "non_spec_state_indices_tensor": tensor_meta(
            meta.non_spec_state_indices_tensor, include_values=True
        ),
        "prefill_state_indices": tensor_meta(
            meta.prefill_state_indices, include_values=True
        ),
        "prefill_has_initial_state": tensor_meta(
            meta.prefill_has_initial_state, include_values=True
        ),
    }
    if meta.chunk_indices is not None:
        out["chunk_indices_head"] = meta.chunk_indices[:8].detach().cpu().tolist()
    return out


def required_pool_size(*metas) -> int:
    max_idx = 0
    for meta in metas:
        for name in [
            "non_spec_state_indices_tensor",
            "prefill_state_indices",
            "spec_state_indices_tensor",
        ]:
            tensor = getattr(meta, name, None)
            if tensor is not None and tensor.numel() > 0:
                max_idx = max(max_idx, int(tensor.max().item()))
    return max_idx + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/Competitions/PRA26/Qwen3.5-9B")
    parser.add_argument(
        "--config-json",
        default=None,
        help=(
            "Path to a HuggingFace config.json. If omitted, uses "
            "<model>/config.json when it exists; otherwise falls back to "
            "Qwen3.5-9B defaults."
        ),
    )
    parser.add_argument("--prefill-tokens", type=int, default=32768)
    parser.add_argument("--decode-context", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-batch-size", type=int, default=1)
    parser.add_argument(
        "--state-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument("--gdn-prefill-backend", default="auto")
    parser.add_argument("--prefill-iters", type=int, default=3)
    parser.add_argument("--decode-iters", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--via-gdn-custom-op",
        action="store_true",
        help=(
            "Route benchmark calls through qwen_gdn_attention_core. This is "
            "useful for validating source-level shape logging."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to benchmark_results/gdn_core_<timestamp>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gdn_config = load_gdn_dims(args)
    set_gdn_dims(
        num_k_heads=gdn_config["num_k_heads"],
        num_v_heads=gdn_config["num_v_heads"],
        head_k_dim=gdn_config["head_k_dim"],
        head_v_dim=gdn_config["head_v_dim"],
        conv_kernel=gdn_config["conv_kernel"],
        num_hidden_layers=gdn_config["num_hidden_layers"],
        num_gdn_layers=gdn_config["num_gdn_layers"],
    )
    out_dir = Path(
        args.output_dir
        or f"benchmark_results/gdn_core_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    torch.manual_seed(0)
    state_dtype = torch.float32 if args.state_dtype == "float32" else torch.bfloat16
    vllm_config = build_vllm_config(args)

    prefill_seq_lens = [args.prefill_tokens] * args.batch_size
    prefill_query_lens = [args.prefill_tokens] * args.batch_size
    decode_seq_lens = [args.decode_context + 1] * args.decode_batch_size
    decode_query_lens = [1] * args.decode_batch_size
    prefill_meta = build_metadata(vllm_config, prefill_seq_lens, prefill_query_lens)
    decode_meta = build_metadata(vllm_config, decode_seq_lens, decode_query_lens)
    pool_size = required_pool_size(prefill_meta, decode_meta)

    layer = build_layer(vllm_config, pool_size=pool_size, state_dtype=state_dtype)
    torch.cuda.synchronize()

    results = {
        "device": torch.cuda.get_device_name(),
        "state_dtype": str(state_dtype),
        "config": gdn_config,
        "gdn_dims": {
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_gdn_layers": NUM_GDN_LAYERS,
            "num_k_heads": H,
            "num_v_heads": HV,
            "head_k_dim": K,
            "head_v_dim": V,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "conv_dim": CONV_DIM,
            "conv_kernel": CONV_KERNEL,
        },
        "prefill": bench_phase(
            "prefill",
            layer,
            prefill_meta,
            args.prefill_tokens * args.batch_size,
            args.warmups,
            args.prefill_iters,
            args.profile,
            args.via_gdn_custom_op,
        ),
        "decode": bench_phase(
            "decode",
            layer,
            decode_meta,
            args.decode_batch_size,
            args.warmups,
            args.decode_iters,
            args.profile,
            args.via_gdn_custom_op,
        ),
    }

    for phase in ("prefill", "decode"):
        table = results[phase].pop("profile_table")
        if table is not None:
            table_path = out_dir / f"{phase}_profiler_table.txt"
            table_path.write_text(table, encoding="utf-8")
            results[phase]["profile_table"] = str(table_path)
        trace_file = results[phase]["trace_file"]
        if trace_file is not None:
            src = cwd / trace_file
            dst = out_dir / trace_file
            if src.exists():
                src.replace(dst)
                results[phase]["trace_file"] = str(dst)

    output_json = out_dir / "summary.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote standalone GDN benchmark to {output_json}")


if __name__ == "__main__":
    main()
