# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-path tests for fused Qwen3.5 GDN MTP decode."""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_rocm()
    or (current_platform.is_cuda() and current_platform.has_device_capability(80))
):
    pytest.skip(
        reason="Fused GDN MTP decode requires ROCm or CUDA compute capability 8.0+.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
SPEC_TOKENS = NUM_SPEC + 1
H = 1
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
EPS = 1e-6


class _TestGatedNorm:
    def __init__(self, weight: torch.Tensor, activation: str) -> None:
        self.weight = weight
        self.bias = None
        self.eps = EPS
        self.group_size = None
        self.norm_before_gate = True
        self.activation = activation

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x,
            self.weight,
            None,
            z=z,
            eps=self.eps,
            norm_before_gate=True,
            activation=self.activation,
        )


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.additional_config = {
        "gdn_prefill_backend": "triton" if current_platform.is_rocm() else "cutedsl"
    }
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_layer(
    vllm_config,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_gate_activation: str,
):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=K,
        value_dim=HV * V,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_TestGatedNorm(norm_weight, output_gate_activation),
        layer_norm_epsilon=EPS,
        gdn_decode_kernel="cuda",
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    layer.gdn_prefill_backend = layer.chunk_gated_delta_rule.gdn_prefill_backend
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "split_ba",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@pytest.mark.parametrize(
    "num_v_heads,expected",
    [
        pytest.param(2, True, id="ratio1"),
        pytest.param(4, True, id="ratio2"),
        pytest.param(6, True, id="ratio3"),
        pytest.param(8, True, id="ratio4"),
        pytest.param(16, True, id="ratio8"),
        pytest.param(5, False, id="non-integral-ratio"),
        pytest.param(10, False, id="unsupported-ratio5"),
    ],
)
def test_fused_mtp_head_ratio_guard(num_v_heads: int, expected: bool) -> None:
    if not qwen_gdn_linear_attn.supports_fused_mtp_decode():
        pytest.skip("fused GDN decode MTP op is not built")

    layer = types.SimpleNamespace(
        num_k_heads=2,
        num_v_heads=num_v_heads,
        kv_cache=(None, torch.empty(1, dtype=torch.float32, device="cuda")),
        gdn_decode_kernel="cuda",
    )
    attn_metadata = types.SimpleNamespace(
        spec_state_indices_tensor=torch.ones(
            1, SPEC_TOKENS, dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )

    assert (
        QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode(
            cast(QwenGatedDeltaNetAttention, layer),
            cast(GDNAttentionMetadata, attn_metadata),
        )
        is expected
    )


@torch.inference_mode()
def test_fused_forward_uses_packed_entrypoint() -> None:
    """Fused mode keeps projected QKVZ and BA packed through the model op."""
    device = torch.device("cuda")
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=device)
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_fused_gdn_spec_decode=True,
        _fused_gdn_fp8=None,
        _ba_stream=None,
        norm=types.SimpleNamespace(
            weight=torch.empty(V, dtype=torch.bfloat16, device=device)
        ),
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        in_proj_qkvz=lambda _: (mixed_qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        out_proj=lambda x: (x, None),
    )
    layer.forward_cuda = types.MethodType(
        QwenGatedDeltaNetAttention.forward_cuda, layer
    )

    def packed_op(
        actual_qkvz: torch.Tensor,
        actual_ba: torch.Tensor,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        assert actual_qkvz is mixed_qkvz
        assert actual_ba is ba
        assert layer_name == _encode_layer_name(PREFIX)
        output.fill_(1)

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed_mock,
        patch.object(torch.ops.vllm, "qwen_gdn_attention_core") as triton_mock,
    ):
        output = layer.forward_cuda(hidden_states)

    packed_mock.assert_called_once()
    triton_mock.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "seq_lens,query_lens,device_query_lens,draft_tokens,expected_fused_calls",
    [
        pytest.param([128], [SPEC_TOKENS], None, [NUM_SPEC], 1, id="pure-mtp"),
        pytest.param(
            [128, 128, 128],
            [3, 3, 3],
            [1, 4, 4],
            [2, 2, 2],
            1,
            id="adaptive-ragged-mtp",
        ),
        pytest.param(
            [128] * 8,
            [1] * 8,
            [4, 4, 0, 0, 0, 0, 0, 0],
            [NUM_SPEC] * 8,
            1,
            id="adaptive-padded-ragged-mtp",
        ),
        pytest.param(
            [128, 96],
            [SPEC_TOKENS, 64],
            None,
            [NUM_SPEC, -1],
            0,
            id="mixed-mtp-falls-back",
        ),
        pytest.param([96], [64], None, [-1], 0, id="pure-prefill"),
        pytest.param([128], [1], None, [-1], 0, id="pure-decode"),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_model_path_matches_reference(
    seq_lens: list[int],
    query_lens: list[int],
    device_query_lens: list[int] | None,
    draft_tokens: list[int],
    expected_fused_calls: int,
    output_gate_activation: str,
) -> None:
    """Fused MTP and its mixed/prefill/decode fallbacks match the reference."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    if device_query_lens is not None:
        # Simulate adaptive verification token reallocation
        assert sum(device_query_lens) == sum(query_lens)
        device_query_start_loc = torch.zeros(
            len(device_query_lens) + 1, dtype=torch.int32, device=device
        )
        torch.cumsum(
            torch.tensor(device_query_lens, dtype=torch.int32, device=device),
            dim=0,
            out=device_query_start_loc[1:],
        )
        common = common.replace(query_start_loc=device_query_start_loc)
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        )

    state_indices = [
        indices
        for indices in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if indices is not None and indices.numel() > 0
    ]
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = 0.01 * torch.randn(
        pool_size, *temporal_state_shape, dtype=torch.float32, device=device
    )
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkv = 0.1 * torch.randn(
        num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)
    output_gate = 0.1 * torch.randn(
        num_tokens, HV, V, dtype=torch.bfloat16, device=device
    )
    mixed_qkvz = torch.cat((mixed_qkv, output_gate.flatten(1)), dim=-1)
    ba = torch.cat((b, a), dim=-1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    reference_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    reference_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        reference_layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b,
            a=a,
            core_attn_out=reference_out,
        )
    reference_out = reference_layer.norm(reference_out, output_gate)

    fused_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    context.no_compile_layers = {PREFIX: fused_layer}
    # Poison: the packed entrypoint writes every actual row, so no NaN survives.
    fused_out = torch.full_like(output_gate, torch.nan)
    fused_mock = MagicMock(wraps=qwen_gdn_linear_attn.fused_mtp_decode_impl())
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
        patch.object(
            qwen_gdn_linear_attn, "fused_mtp_decode_impl", return_value=fused_mock
        ),
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            fused_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    assert fused_mock.call_count == expected_fused_calls
    torch.testing.assert_close(
        fused_layer.kv_cache[0], reference_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(fused_out, reference_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        reference_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize(
    "state_dtype,dim_first", [(torch.bfloat16, False), (torch.float32, True)]
)
@torch.inference_mode()
def test_flydsl_gdn_verify_preserves_accepted_prefix(
    state_dtype: torch.dtype, dim_first: bool
) -> None:
    if not current_platform.is_rocm():
        pytest.skip("FlyDSL GDN verification requires ROCm")
    from vllm.platforms.rocm import on_rdna4

    if not on_rdna4():
        pytest.skip("FlyDSL GDN verification requires RDNA4")
    pytest.importorskip("flydsl")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.qwen3_5.amd.mono.gdn_verify import flydsl_gdn_spec_decode
    from vllm.models.qwen3_5.common.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    key_heads, value_heads, cap = 2, 6, 8
    qk_width, v_width = key_heads * K, value_heads * V
    conv_width = 2 * qk_width + v_width
    lengths = (8, 3, 0)
    tokens = sum(lengths)

    def rand(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator, device=device) * scale).to(
            dtype
        )

    # The second request starts from its eighth snapshot and writes three new
    # snapshots. The padding row has no state and the other slots stay intact.
    indices = torch.tensor(
        [list(range(8, 0, -1)), list(range(16, 8, -1)), [0] * cap],
        dtype=torch.int32,
        device=device,
    )
    starts = torch.tensor([0, 8, tokens, tokens], dtype=torch.int32, device=device)
    untouched = torch.tensor([0, 9, 10, 11, 12, 13], device=device)
    conv = rand(17, conv_width, cap + CONV_KERNEL - 2, scale=0.25)
    if not dim_first:
        conv = conv.transpose(1, 2).contiguous().transpose(1, 2)
    state = rand(17, value_heads, V, K, scale=0.03, dtype=state_dtype)
    conv_ref, state_ref = conv.clone(), state.clone()
    weight = rand(conv_width, CONV_KERNEL, scale=0.3)
    norm = 1 + rand(V, scale=0.1)
    log_a = rand(value_heads, dtype=torch.float32, scale=0.2) - 3
    bias = rand(value_heads, scale=0.2)
    q = torch.empty((tokens, v_width), dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty((tokens, value_heads), dtype=torch.float32, device=device)

    for accepted_prefix in ([5, 8, 1], [3, 1, 1], [8, 2, 1]):
        accepted = torch.tensor(accepted_prefix, dtype=torch.int32, device=device)
        qkvz = rand(tokens, conv_width + v_width, scale=0.5)
        ba = rand(tokens, 2 * value_heads, scale=0.2)
        b, a = ba.chunk(2, dim=1)
        qkv, gate = qkvz.split((conv_width, v_width), dim=1)
        x = causal_conv1d_update(
            qkv,
            conv_ref,
            weight,
            None,
            "silu",
            conv_state_indices=indices[:, 0],
            num_accepted_tokens=accepted,
            query_start_loc=starts,
            max_query_len=cap,
            out=torch.empty_like(qkv),
        )
        query, key, value = x.split((qk_width, qk_width, v_width), dim=1)
        out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=log_a,
            a=a,
            b=b,
            dt_bias=bias,
            q=query.reshape(1, tokens, key_heads, K),
            k=key.reshape(1, tokens, key_heads, K),
            v=value.reshape(1, tokens, value_heads, V),
            initial_state=state_ref,
            cu_seqlens=starts,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )
        out = out.reshape(tokens, v_width)
        fused_gdn_gated_rms_norm(
            out, gate, norm, out, EPS, value_heads, V, True, "silu"
        )
        ref, ref_scales = per_token_group_quant_fp8(out, V, use_ue8m0=False)
        flydsl_gdn_spec_decode(
            qkvz,
            ba,
            weight,
            conv,
            state,
            indices,
            starts,
            accepted,
            log_a,
            bias,
            norm,
            q,
            scales,
            EPS,
        )

        torch.testing.assert_close(conv, conv_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            state[untouched], state_ref[untouched], atol=0, rtol=0
        )
        state_error = state.float() - state_ref.float()
        assert state_error.norm() < 0.001 * state_ref.float().norm()
        actual = q.float().reshape(tokens, value_heads, V) * scales.unsqueeze(-1)
        expected = ref.float().reshape(tokens, value_heads, V) * ref_scales.unsqueeze(
            -1
        )
        assert (actual - expected).norm() < 0.01 * expected.norm()
