# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Time Qwen block-FP8 GDN decode on RDNA4 with equal-size graph replays."""

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)


def graph_us(run: Callable[[], None], calls: int) -> float:
    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            run()
    graph.replay()
    torch.accelerator.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(10):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(times)


@torch.inference_mode()
def run_case(tokens: int, tp_size: int, calls: int) -> None:
    from vllm.models.qwen3_5.amd.flydsl_gdn_decode import flydsl_gdn_decode
    from vllm.models.qwen3_5.common.fused_gdn_gated_norm import fused_gdn_gated_rms_norm

    key_heads, value_heads = 16 // tp_size, 48 // tp_size
    head_dim = 128
    channels = (2 * key_heads + value_heads) * head_dim
    width = value_heads * head_dim
    eps = 1e-6
    torch.manual_seed(8)

    def rand(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, device="cuda") * scale).to(dtype)

    packed = rand(tokens, channels + width, scale=0.5)
    ba = rand(tokens, 2 * value_heads, scale=0.2)
    b, a = ba.chunk(2, dim=-1)
    conv_weight = rand(channels, 4, scale=0.3)
    conv = rand(tokens + 1, 3, channels, scale=0.25).transpose(1, 2)
    state = rand(tokens + 1, value_heads, head_dim, head_dim, scale=0.03)
    conv_ref, state_ref = conv.clone(), state.clone()
    indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32)
    a_log = rand(value_heads, dtype=torch.float32, scale=0.2) - 3
    bias = rand(value_heads, scale=0.2)
    norm_weight = 1 + rand(head_dim, scale=0.1)
    qkv, gate = packed.split((channels, width), dim=-1)
    convolved = torch.empty_like(qkv)
    core = torch.empty_like(gate)
    q = torch.empty_like(gate, dtype=torch.float8_e4m3fn)
    scales = torch.empty((tokens, value_heads), device="cuda", dtype=torch.float32)

    def baseline() -> None:
        causal_conv1d_update(
            qkv,
            conv_ref,
            conv_weight,
            None,
            "silu",
            conv_state_indices=indices,
            out=convolved,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=convolved,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=bias,
            scale=head_dim**-0.5,
            initial_state=state_ref,
            out=core.view(tokens, 1, value_heads, head_dim),
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        fused_gdn_gated_rms_norm(
            core, gate, norm_weight, core, eps, value_heads, head_dim, True, "silu"
        )
        per_token_group_quant_fp8(core, head_dim, use_ue8m0=False)

    def fused() -> None:
        flydsl_gdn_decode(
            packed,
            ba,
            conv_weight,
            conv,
            state,
            indices,
            a_log,
            bias,
            norm_weight,
            q,
            scales,
            eps,
            tokens,
        )

    baseline_us = graph_us(baseline, calls)
    fused_us = graph_us(fused, calls)
    print(
        f"TP{tp_size} tokens={tokens:2d}  stock={baseline_us:6.2f} us  "
        f"FlyDSL={fused_us:6.2f} us  speedup={baseline_us / fused_us:.2f}x"
    )


def main() -> None:
    from vllm.platforms import current_platform

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--tp-size", type=int, choices=[1, 2], default=2)
    parser.add_argument("--graph-calls", type=int, default=128)
    args = parser.parse_args()
    if not current_platform.is_rocm():
        parser.error("requires ROCm")
    from vllm.platforms.rocm import on_rdna4

    if not on_rdna4():
        parser.error("requires RDNA4")
    if args.graph_calls < 1 or any(tokens < 1 for tokens in args.batches):
        parser.error("batch sizes and graph calls must be positive")
    for tokens in args.batches:
        run_case(tokens, args.tp_size, args.graph_calls)


if __name__ == "__main__":
    main()
