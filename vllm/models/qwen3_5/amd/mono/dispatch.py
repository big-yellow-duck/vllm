# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM dispatch for the RDNA4 Qwen FP8 GDN kernels."""

from typing import TYPE_CHECKING

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .common import CONV_WIDTH, HEAD_DIM
from .gdn_decode import flydsl_gdn_decode
from .gdn_norm import gated_norm_quant
from .gdn_verify import flydsl_gdn_spec_decode

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

_MAX_MONO_DECODE_TOKENS = 16
_MAX_MONO_VERIFY_TOKENS = 64
_MAX_VERIFY_TOKENS_PER_REQUEST = 8


def _try_mono_decode(
    layer: "QwenGatedDeltaNetAttention",
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    metadata: GDNAttentionMetadata,
) -> bool:
    tokens = metadata.num_actual_tokens
    if (
        tokens <= 0
        or metadata.num_prefills
        or metadata.num_spec_decode_tokens > _MAX_MONO_VERIFY_TOKENS
        or not (qkvz.is_contiguous() and ba.is_contiguous())
    ):
        return False
    conv_state, state = layer.kv_cache[:2]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    weight = layer.conv1d.weight
    if (
        conv_state.dtype != torch.bfloat16
        or conv_state.shape[1] != weight.shape[0]
        or conv_state.shape[-1] < CONV_WIDTH - 1
        or state.dtype not in (torch.bfloat16, torch.float32)
        or state.shape[1:] != (scales.shape[1], HEAD_DIM, HEAD_DIM)
        or state.stride()[1:] != (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or layer.conv1d.bias is not None
        or layer.A_log.dtype != torch.float32
        or layer.dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or layer.activation not in ("silu", "swish")
    ):
        return False

    if metadata.spec_sequence_masks is not None:
        requests = metadata.num_spec_decodes
        indices = metadata.spec_state_indices_tensor
        starts = metadata.spec_query_start_loc
        accepted = metadata.num_accepted_tokens
        if (
            metadata.num_decodes
            or requests <= 0
            or indices is None
            or indices.ndim != 2
            or indices.shape[0] < requests
            or not 2 <= indices.shape[1] <= _MAX_VERIFY_TOKENS_PER_REQUEST
            or indices.stride(1) != 1
            or indices.dtype != torch.int32
            or conv_state.shape[-1] < CONV_WIDTH + indices.shape[1] - 2
            or starts is None
            or starts.dtype != torch.int32
            or starts.shape[0] <= requests
            or not starts.is_contiguous()
            or accepted is None
            or accepted.dtype != torch.int32
            or accepted.shape[0] < requests
            or not accepted.is_contiguous()
        ):
            return False
        flydsl_gdn_spec_decode(
            qkvz,
            ba,
            weight.view(-1, CONV_WIDTH),
            conv_state,
            state,
            indices[:requests],
            starts[: requests + 1],
            accepted[:requests],
            layer.A_log,
            layer.dt_bias,
            layer.norm.weight,
            quantized,
            scales,
            layer.norm.eps,
            query_length=metadata.uniform_spec_sequence_length,
        )
        return True

    indices = metadata.non_spec_state_indices_tensor
    if (
        not 0 < tokens == metadata.num_decodes <= _MAX_MONO_DECODE_TOKENS
        or indices is None
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
        or indices.numel() < tokens
    ):
        return False
    flydsl_gdn_decode(
        qkvz,
        ba,
        weight.view(-1, CONV_WIDTH),
        conv_state,
        state,
        indices,
        layer.A_log,
        layer.dt_bias,
        layer.norm.weight,
        quantized,
        scales,
        layer.norm.eps,
        tokens,
    )
    return True


@eager_break_during_capture
def _qwen_gdn_flydsl_fp8(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    context = get_forward_context()
    prefix = _resolve_layer_name(layer_name)
    layer = context.no_compile_layers[prefix]
    attn_metadata = context.attn_metadata
    metadata = attn_metadata.get(prefix) if isinstance(attn_metadata, dict) else None
    if isinstance(metadata, GDNAttentionMetadata) and _try_mono_decode(
        layer, qkvz, ba, quantized, scales, metadata
    ):
        return

    tokens, heads = scales.shape
    width = heads * HEAD_DIM
    qkv, gate = qkvz.split([qkvz.shape[-1] - width, width], dim=-1)
    b, a = layer.split_ba(ba)
    core = qkvz.new_empty((tokens, heads, HEAD_DIM))
    layer._forward_core(qkv, b, a, core)
    gated_norm_quant(
        core.flatten(1), gate, layer.norm.weight, quantized, scales, layer.norm.eps
    )


direct_register_custom_op(
    op_name="qwen_gdn_flydsl_fp8",
    op_func=_qwen_gdn_flydsl_fp8,
    mutates_args=["qkvz", "quantized", "scales"],
)


def fused_gdn_fp8(
    layer: "QwenGatedDeltaNetAttention", qkvz: torch.Tensor, ba: torch.Tensor
) -> QuantizedActivation | None:
    if (
        get_input_quant_key(layer.out_proj) != kFp8Dynamic128Sym
        or qkvz.dtype != torch.bfloat16
        or ba.dtype != torch.bfloat16
        or layer.norm.weight.dtype not in (torch.bfloat16, torch.float32)
        or layer.norm.bias is not None
        or layer.norm.activation not in ("silu", "swish")
        or not layer.norm.norm_before_gate
    ):
        return None
    heads = layer.num_v_heads // layer.tp_size
    tokens = qkvz.shape[0]
    data = torch.empty(
        (tokens, heads * HEAD_DIM), dtype=torch.float8_e4m3fn, device=qkvz.device
    )
    scales = torch.empty((tokens, heads), dtype=torch.float32, device=qkvz.device)
    torch.ops.vllm.qwen_gdn_flydsl_fp8(
        qkvz, ba, data, scales, _encode_layer_name(layer.prefix)
    )
    return QuantizedActivation(data, scales, qkvz.dtype, data.shape, kFp8Dynamic128Sym)
