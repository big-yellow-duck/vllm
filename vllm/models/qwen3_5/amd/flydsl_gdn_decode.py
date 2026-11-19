# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RDNA4 Qwen GDN decode with an FP8 epilogue."""

from functools import lru_cache
from typing import TYPE_CHECKING

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Float32, Int32, Int64, Stream, T

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

from .flydsl_gdn_norm import (
    _HEAD_DIM,
    _VECTOR_WIDTH,
    _WAVE_SIZE,
    _gated_norm_fp8,
    gated_norm_quant,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

_CONV_WIDTH = 4
_VALUES_PER_KEY = 3
_MAX_MONO_DECODE_TOKENS = 16


def _sum_lanes(value, width):
    for shift in range_constexpr(width.bit_length() - 2, -1, -1):
        value = value + value.shuffle_xor(1 << shift, _WAVE_SIZE)
    return value


def _exp(value):
    return Float32(rocdl.exp2(T.f32, (value * 1.4426950408889634).ir_value()))


def _sigmoid(value):
    denominator = Float32(1.0) + _exp(-value)
    return Float32(rocdl.rcp(T.f32, denominator.ir_value()))


def _rsqrt(value):
    return Float32(rocdl.rsq(T.f32, value.ir_value()))


def _load(resource, offset, dtype):
    return bo.buffer_load(resource, offset, vec_width=1, dtype=dtype)


def _load_vec(resource, offset, dtype):
    return fx.Vector(
        bo.buffer_load(resource, offset, vec_width=_VECTOR_WIDTH, dtype=dtype)
    ).to(Float32)


def _conv_silu(x_buf, row, weight, channel, cache, channel_stride, time_stride):
    offset = channel * channel_stride
    h0 = fx.BFloat16(_load(cache, offset, T.bf16))
    h1 = fx.BFloat16(_load(cache, offset + time_stride, T.bf16))
    h2 = fx.BFloat16(_load(cache, offset + 2 * time_stride, T.bf16))
    x = fx.BFloat16(_load(x_buf, row + channel, T.bf16))
    weights = _load_vec(weight, channel * _CONV_WIDTH, T.bf16)
    inputs = fx.Vector.from_elements([h0, h1, h2, x], Float32)
    # The Triton convolution rounds each BF16 product before FP32 accumulation.
    products = (inputs * weights).to(fx.BFloat16).to(Float32)
    result = products[0] + products[1] + products[2] + products[3]
    bo.buffer_store(h1, cache, offset)
    bo.buffer_store(h2, cache, offset + time_stride)
    bo.buffer_store(x, cache, offset + 2 * time_stride)
    return fx.BFloat16(result * _sigmoid(result))


@lru_cache(maxsize=32)
def _build_decode(
    device: int,
    key_heads: int,
    conv_channel_stride: int,
    conv_time_stride: int,
    fp32_state: bool,
    fp32_bias: bool,
    fp32_norm: bool,
    norm_eps: float,
):
    value_heads = key_heads * _VALUES_PER_KEY
    qk_width = 2 * key_heads * _HEAD_DIM
    conv_width = qk_width + value_heads * _HEAD_DIM
    projection_width = conv_width + value_heads * _HEAD_DIM
    # Each block owns one key head and its three value heads. This gives the
    # Q/K convolution history a single writer and lets its consumers share LDS.
    heads_per_block = _VALUES_PER_KEY
    threads_per_head = 256
    k_lanes = 8
    threads = heads_per_block * threads_per_head
    rows_per_step = threads_per_head // k_lanes
    partials = _HEAD_DIM // _WAVE_SIZE
    vector_steps = _HEAD_DIM // (k_lanes * _VECTOR_WIDTH)
    state_element_size = 4 if fp32_state else 2

    @fx.struct
    class Shared:
        q: fx.Array[Float32, _HEAD_DIM, 16]
        k: fx.Array[Float32, _HEAD_DIM, 16]
        q_squares: fx.Array[Float32, partials, 16]
        k_squares: fx.Array[Float32, partials, 16]
        v: fx.Array[Float32, heads_per_block * _HEAD_DIM, 16]
        output: fx.Array[Float32, heads_per_block * _HEAD_DIM, 16]
        decay: fx.Array[Float32, heads_per_block, 16]
        beta: fx.Array[Float32, heads_per_block, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def qwen_gdn_mono_decode(
        qkvz: Int64,
        ba: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state: Int64,
        indices: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        quantized: Int64,
        scales: Int64,
        conv_slot_stride: Int64,
        state_slot_stride: Int64,
    ):
        state_type = T.f32 if fp32_state else T.bf16
        bias_type = T.f32 if fp32_bias else T.bf16
        norm_type = T.f32 if fp32_norm else T.bf16
        token = gpu.block_idx.x
        tid = gpu.thread_idx.x
        local_head = tid // threads_per_head
        head_thread = tid % threads_per_head
        head = gpu.block_idx.y * heads_per_block + local_head
        key_head = head // _VALUES_PER_KEY
        group = token * value_heads + head
        head_offset = local_head * _HEAD_DIM
        buf = bo.create_buffer_resource_from_addr
        slot = Int64(_load(buf(indices), token, T.i32))
        if slot > 0:
            shared = fx.SharedAllocator().allocate(Shared).peek()
            x_buf, w_buf = buf(qkvz), buf(conv_weight)
            conv_buf = buf(conv_state + slot * conv_slot_stride * 2)
            state_buf = buf(state + slot * state_slot_stride * state_element_size)
            input_base = token * projection_width

            if tid < _HEAD_DIM:
                q_channel = key_head * _HEAD_DIM + tid
                k_channel = q_channel + key_heads * _HEAD_DIM
                q = Float32(
                    _conv_silu(
                        x_buf,
                        input_base,
                        w_buf,
                        q_channel,
                        conv_buf,
                        conv_channel_stride,
                        conv_time_stride,
                    )
                )
                k = Float32(
                    _conv_silu(
                        x_buf,
                        input_base,
                        w_buf,
                        k_channel,
                        conv_buf,
                        conv_channel_stride,
                        conv_time_stride,
                    )
                )
                fx.ptr_store(q, shared.q.ptr + tid)
                fx.ptr_store(k, shared.k.ptr + tid)
                q_square = _sum_lanes(q * q, _WAVE_SIZE)
                k_square = _sum_lanes(k * k, _WAVE_SIZE)
                if tid % _WAVE_SIZE == 0:
                    wave = tid // _WAVE_SIZE
                    fx.ptr_store(q_square, shared.q_squares.ptr + wave)
                    fx.ptr_store(k_square, shared.k_squares.ptr + wave)

            if head_thread < _HEAD_DIM:
                channel = qk_width + head * _HEAD_DIM + head_thread
                v = _conv_silu(
                    x_buf,
                    input_base,
                    w_buf,
                    channel,
                    conv_buf,
                    conv_channel_stride,
                    conv_time_stride,
                )
                fx.ptr_store(Float32(v), shared.v.ptr + head_offset + head_thread)
            if head_thread == 0:
                b_offset = token * (2 * value_heads) + head
                b = Float32(_load(buf(ba), b_offset, T.bf16))
                a = Float32(_load(buf(ba), b_offset + value_heads, T.bf16))
                bias = Float32(_load(buf(dt_bias), head, bias_type))
                log_rate = Float32(_load(buf(a_log), head, T.f32))
                delta = a + bias
                softplus = delta
                if delta <= Float32(20.0):
                    softplus = Float32(fmath.log(Float32(1.0) + _exp(delta)))
                fx.ptr_store(
                    _exp(-_exp(log_rate) * softplus), shared.decay.ptr + local_head
                )
                fx.ptr_store(_sigmoid(b), shared.beta.ptr + local_head)
            gpu.barrier()

            q_square, k_square = Float32(0.0), Float32(0.0)
            for wave in range_constexpr(partials):
                q_square += fx.ptr_load(shared.q_squares.ptr + wave)
                k_square += fx.ptr_load(shared.k_squares.ptr + wave)
            q_scale = _rsqrt(q_square + 1e-6) * _HEAD_DIM**-0.5
            k_scale = _rsqrt(k_square + 1e-6)
            k_lane = head_thread % k_lanes
            q_parts, k_parts, offsets = [], [], []
            for part in range_constexpr(vector_steps):
                offset = (part * k_lanes + k_lane) * _VECTOR_WIDTH
                offsets.append(offset)
                q_parts.append(
                    fx.Vector.from_elements(
                        [
                            fx.ptr_load(shared.q.ptr + offset + i)
                            for i in range(_VECTOR_WIDTH)
                        ],
                        Float32,
                    )
                    * q_scale
                )
                k_parts.append(
                    fx.Vector.from_elements(
                        [
                            fx.ptr_load(shared.k.ptr + offset + i)
                            for i in range(_VECTOR_WIDTH)
                        ],
                        Float32,
                    )
                    * k_scale
                )
            decay = fx.ptr_load(shared.decay.ptr + local_head)
            beta = fx.ptr_load(shared.beta.ptr + local_head)
            for step in fx.range(_HEAD_DIM // rows_per_step, unroll=1):
                row = step * rows_per_step + head_thread // k_lanes
                state_base = (head * _HEAD_DIM + row) * _HEAD_DIM
                states = []
                dot = Float32(0.0)
                for part in range_constexpr(vector_steps):
                    state_vec = (
                        _load_vec(state_buf, state_base + offsets[part], state_type)
                        * decay
                    )
                    states.append(state_vec)
                    dot += (state_vec * k_parts[part]).reduce(fx.ReductionOp.ADD)
                v = fx.ptr_load(shared.v.ptr + head_offset + row)
                residual = (v - _sum_lanes(dot, k_lanes)) * beta
                result = Float32(0.0)
                for part in range_constexpr(vector_steps):
                    updated = states[part] + k_parts[part] * residual
                    result += (updated * q_parts[part]).reduce(fx.ReductionOp.ADD)
                    if not fp32_state:
                        updated = updated.to(fx.BFloat16)
                    bo.buffer_store(updated, state_buf, state_base + offsets[part])
                result = Float32(fx.BFloat16(_sum_lanes(result, k_lanes)))
                if k_lane == 0:
                    fx.ptr_store(result, shared.output.ptr + head_offset + row)
            gpu.barrier()

            if head_thread < _WAVE_SIZE:
                col = head_thread * _VECTOR_WIDTH
                values = fx.Vector.from_elements(
                    [
                        fx.ptr_load(shared.output.ptr + head_offset + col + i)
                        for i in range(_VECTOR_WIDTH)
                    ],
                    Float32,
                )
                z_offset = input_base + conv_width + head * _HEAD_DIM + col
                gates = _load_vec(x_buf, z_offset, T.bf16)
                weights = _load_vec(buf(norm_weight), col, norm_type)
                packed, scale = _gated_norm_fp8(
                    values, gates, weights, Float32(norm_eps)
                )
                bo.buffer_store(
                    packed, buf(quantized), group * _WAVE_SIZE + head_thread
                )
                if head_thread == 0:
                    bo.buffer_store(scale, buf(scales), group)
        else:
            if head_thread < _WAVE_SIZE:
                bo.buffer_store(
                    Int32(0), buf(quantized), group * _WAVE_SIZE + head_thread
                )
                if head_thread == 0:
                    bo.buffer_store(Float32(1e-10 / 448.0), buf(scales), group)

    @flyc.jit
    def launch(
        qkvz: Int64,
        ba: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state: Int64,
        indices: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        quantized: Int64,
        scales: Int64,
        conv_slot_stride: Int64,
        state_slot_stride: Int64,
        tokens: Int32,
        stream: Stream,
    ):
        qwen_gdn_mono_decode(
            qkvz,
            ba,
            conv_weight,
            conv_state,
            state,
            indices,
            a_log,
            dt_bias,
            norm_weight,
            quantized,
            scales,
            conv_slot_stride,
            state_slot_stride,
        ).launch(
            grid=(tokens, value_heads // heads_per_block, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    with torch.accelerator.device_index(device):
        return flyc.compile(launch, *([0] * 13), 1, Stream(None), compile_only=True)


def flydsl_gdn_decode(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    norm_eps: float,
    num_tokens: int,
) -> None:
    """Advance the convolution and recurrent states and emit group FP8."""
    key_heads = scales.shape[1] // _VALUES_PER_KEY
    launch = _build_decode(
        qkvz.get_device(),
        key_heads,
        conv_state.stride(1),
        conv_state.stride(2),
        state.dtype == torch.float32,
        dt_bias.dtype == torch.float32,
        norm_weight.dtype == torch.float32,
        norm_eps,
    )
    launch(
        qkvz.data_ptr(),
        ba.data_ptr(),
        conv_weight.data_ptr(),
        conv_state.data_ptr(),
        state.data_ptr(),
        indices.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        norm_weight.data_ptr(),
        quantized.data_ptr(),
        scales.data_ptr(),
        conv_state.stride(0),
        state.stride(0),
        num_tokens,
        torch.cuda.current_stream(qkvz.device),
    )


def _try_mono_decode(
    layer: "QwenGatedDeltaNetAttention",
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    metadata: GDNAttentionMetadata,
) -> bool:
    tokens = metadata.num_actual_tokens
    indices = metadata.non_spec_state_indices_tensor
    if (
        metadata.num_prefills
        or metadata.spec_sequence_masks is not None
        or not 0 < tokens == metadata.num_decodes <= _MAX_MONO_DECODE_TOKENS
        or indices is None
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
        or indices.numel() < tokens
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
        or conv_state.shape[-1] != _CONV_WIDTH - 1
        or state.dtype not in (torch.bfloat16, torch.float32)
        or state.shape[1:] != (scales.shape[1], _HEAD_DIM, _HEAD_DIM)
        or state.stride()[1:] != (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or layer.conv1d.bias is not None
        or layer.A_log.dtype != torch.float32
        or layer.dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or layer.activation not in ("silu", "swish")
    ):
        return False
    flydsl_gdn_decode(
        qkvz,
        ba,
        weight.view(-1, _CONV_WIDTH),
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
    width = heads * _HEAD_DIM
    qkv, gate = qkvz.split([qkvz.shape[-1] - width, width], dim=-1)
    b, a = layer.split_ba(ba)
    core = qkvz.new_empty((tokens, heads, _HEAD_DIM))
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
        (tokens, heads * _HEAD_DIM), dtype=torch.float8_e4m3fn, device=qkvz.device
    )
    scales = torch.empty((tokens, heads), dtype=torch.float32, device=qkvz.device)
    torch.ops.vllm.qwen_gdn_flydsl_fp8(
        qkvz, ba, data, scales, _encode_layer_name(layer.prefix)
    )
    return QuantizedActivation(data, scales, qkvz.dtype, data.shape, kFp8Dynamic128Sym)
