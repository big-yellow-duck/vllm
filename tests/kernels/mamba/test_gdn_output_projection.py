# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract for `QwenGatedDeltaNetAttention._output_projection`.

Every `forward_*` already allocates `core_attn_out` and `z` as
`(N, H, D)`. The helper now norms those tensors as-is and flattens once
for `out_proj`; it used to be rank-agnostic via `reshape(z_shape_og)`.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_output_projection_norms_per_head_and_flattens(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    num_tokens, num_heads, head_dim = 3, 4, 8
    hidden = num_heads * head_dim

    layer = types.SimpleNamespace()
    layer.norm = RMSNormGated(
        head_dim,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device="cpu",
        dtype=dtype,
    )
    layer.out_proj = lambda x: (x, None)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )

    core_attn_out = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    out = layer._output_projection(core_attn_out, z)

    assert core_attn_out.shape == (num_tokens, num_heads, head_dim)
    assert z.shape == (num_tokens, num_heads, head_dim)
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype


@pytest.mark.parametrize("num_tokens", [1, 37])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused gated RMS norm Triton kernel requires CUDA or ROCm",
)
@torch.inference_mode()
def test_fused_gated_rms_norm_strided_z_bitwise(
    num_tokens: int, norm_before_gate: bool, activation: str
) -> None:
    """The Qwen3.5 fused gated RMS norm reads z from its packed strided view
    of the qkvz projection and matches `rmsnorm_fn` on the copied/reshaped
    z bitwise."""
    from vllm.models.qwen3_5.common.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        rmsnorm_fn,
    )

    torch.manual_seed(0)
    dtype = torch.bfloat16
    num_heads, head_dim = 4, 128
    qkvz_row = 2 * 2 * 128 + 2 * num_heads * head_dim
    qkvz = torch.randn(num_tokens, qkvz_row, dtype=dtype, device="cuda")
    z_view = qkvz[:, qkvz_row - num_heads * head_dim :]
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    weight = torch.randn(head_dim, dtype=torch.float32, device="cuda")
    eps = 1e-6

    ref = rmsnorm_fn(
        x,
        weight,
        None,
        z=z_view.reshape(num_tokens, num_heads, head_dim),
        eps=eps,
        group_size=None,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    out = torch.empty(num_tokens, num_heads * head_dim, dtype=dtype, device="cuda")
    fused_gdn_gated_rms_norm(
        x=x.view(num_tokens, -1),
        z=z_view,
        weight=weight,
        out=out,
        eps=eps,
        num_v_heads=num_heads,
        head_v_dim=head_dim,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )

    torch.testing.assert_close(
        out.view(num_tokens, num_heads, head_dim), ref, rtol=0, atol=0
    )


# 48/24/12 value heads per rank: TP1/2/4 of a 48-head model.
@pytest.mark.parametrize("num_heads", [48, 24, 12])
@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.skipif(
    not (
        (current_platform.is_cuda() and current_platform.has_device_capability(89))
        or (
            current_platform.is_rocm()
            and (
                current_platform.is_device_capability(95)
                or current_platform.is_device_capability(125)
            )
        )
    ),
    reason="the FP8 (e4m3fn) output of the fused gated RMS norm requires SM89+ "
    "or CDNA4+ (gfx950, gfx1250)",
)
@torch.inference_mode()
def test_fused_gated_rms_norm_fp8_matches_chain(
    num_tokens: int, num_heads: int
) -> None:
    """The FP8 output (fused out_proj input quant) equals the bf16 kernel
    followed by static scaled_fp8_quant."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen3_5.common.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )

    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    dtype = torch.bfloat16
    head_dim = 128
    qkvz_row = 2 * (num_heads // 3) * 128 + 2 * num_heads * head_dim
    qkvz = torch.randn(num_tokens, qkvz_row, generator=g, dtype=dtype, device="cuda")
    z_view = qkvz[:, qkvz_row - num_heads * head_dim :]
    x = torch.randn(
        num_tokens, num_heads * head_dim, generator=g, dtype=dtype, device="cuda"
    )
    weight = (torch.rand(head_dim, generator=g, device="cuda") + 0.5).to(dtype)
    # Small enough that the largest outputs saturate at +-448.
    scale = torch.tensor([0.004], device="cuda")
    kwargs = dict(
        z=z_view,
        weight=weight,
        eps=1e-6,
        num_v_heads=num_heads,
        head_v_dim=head_dim,
        norm_before_gate=True,
        activation="silu",
    )

    ref = torch.empty_like(x)
    fused_gdn_gated_rms_norm(x=x, out=ref, **kwargs)
    ref_q, _ = ops.scaled_fp8_quant(ref, scale, group_shape=(-1, -1))
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    fused_gdn_gated_rms_norm(x=x, out=q, scale=scale, **kwargs)

    assert ref_q.float().abs().max().item() == 448.0

    # Arch-dependent compilation flips occasional bf16 rounding boundaries;
    # codes must be within one FP8 step (+0 == -0).
    def codes(t: torch.Tensor) -> torch.Tensor:
        b = t.view(torch.uint8).int()
        return torch.where(b >= 128, -(b & 0x7F), b)

    dist = (codes(q) - codes(ref_q)).abs()
    assert dist.max().item() <= 1
    assert (dist > 0).float().mean().item() <= 1e-4


@pytest.mark.parametrize(
    "num_tokens,key_heads,state_dtype,dim_first",
    [
        (1, 8, torch.bfloat16, False),
        (7, 8, torch.bfloat16, True),
        (4, 16, torch.float32, False),
    ],
)
@torch.inference_mode()
def test_flydsl_gdn_decode_cache_and_fp8_output(
    num_tokens: int, key_heads: int, state_dtype: torch.dtype, dim_first: bool
) -> None:
    if not current_platform.is_rocm():
        pytest.skip("FlyDSL GDN decode requires ROCm")
    from vllm.platforms.rocm import on_rdna4

    if not on_rdna4():
        pytest.skip("FlyDSL GDN decode requires RDNA4")
    pytest.importorskip("flydsl")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.qwen3_5.amd.mono.gdn_decode import flydsl_gdn_decode
    from vllm.models.qwen3_5.common.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    head_dim, width = 128, 4
    value_heads = 3 * key_heads
    channels = (2 * key_heads + value_heads) * head_dim
    out_width = value_heads * head_dim
    generator = torch.Generator(device="cuda").manual_seed(8)
    slots = num_tokens + 2

    def rand(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(shape, device="cuda", generator=generator)
        return (t * scale).to(dtype)

    conv = rand(slots, channels, width - 1, scale=0.25)
    if not dim_first:
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    state = rand(slots, value_heads, head_dim, head_dim, dtype=state_dtype, scale=0.03)
    conv_ref, state_ref = conv.clone(), state.clone()
    indices = torch.arange(num_tokens, 0, -1, device="cuda", dtype=torch.int32)
    if num_tokens > 1:
        indices[-1] = 0
    conv_weight = rand(channels, width, scale=0.3)
    a_log = rand(value_heads, dtype=torch.float32, scale=0.2) - 3
    bias = rand(value_heads, scale=0.2)
    norm_weight = 1 + rand(head_dim, scale=0.1)
    q = torch.empty((num_tokens, out_width), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((num_tokens, value_heads), device="cuda", dtype=torch.float32)
    core = torch.empty_like(q, dtype=torch.bfloat16)
    eps = 1e-6

    for _ in range(3):
        qkvz = rand(num_tokens, channels + out_width, scale=0.5)
        ba = rand(num_tokens, 2 * value_heads, scale=0.2)
        b, a = ba.chunk(2, dim=-1)
        qkv, z = qkvz.split((channels, out_width), dim=-1)
        convolved = causal_conv1d_update(
            qkv,
            conv_ref,
            conv_weight,
            None,
            "silu",
            conv_state_indices=indices,
            out=torch.empty_like(qkv),
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=convolved,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=bias,
            scale=head_dim**-0.5,
            initial_state=state_ref,
            out=core.view(num_tokens, 1, value_heads, head_dim),
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        fused_gdn_gated_rms_norm(
            core, z, norm_weight, core, eps, value_heads, head_dim, True, "silu"
        )
        q_ref, scale_ref = per_token_group_quant_fp8(core, head_dim, use_ue8m0=False)
        flydsl_gdn_decode(
            qkvz,
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
            num_tokens,
        )
        torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)
        state_error = state.float() - state_ref.float()
        assert state_error.norm() < 0.003 * state_ref.float().norm()
        assert state_error.abs().max() < 0.006
        torch.testing.assert_close(state[0], state_ref[0], rtol=0, atol=0)
        torch.testing.assert_close(scales, scale_ref, rtol=0.02, atol=1e-10)
        actual = q.float().view(num_tokens, value_heads, head_dim) * scales.unsqueeze(
            -1
        )
        expected = q_ref.float().view(
            num_tokens, value_heads, head_dim
        ) * scale_ref.unsqueeze(-1)
        # Small differences before quantization can flip an FP8 rounding boundary.
        assert (actual - expected).norm() < 0.025 * expected.norm()
        if num_tokens > 1:
            assert actual[-1].count_nonzero().item() == 0
