# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RDNA4 Qwen GDN decode with an FP8 epilogue."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Float32, Int32, Int64, Stream, T

from .common import (
    HEAD_DIM,
    VALUES_PER_KEY,
    VECTOR_WIDTH,
    WAVE_SIZE,
    buffer_ptr,
    conv_result,
    exp,
    lds_barrier,
    load_conv_weights,
    load_fp32_vec,
    rsqrt,
    sigmoid,
    sum_lanes,
)
from .gdn_norm import _gated_norm_fp8


def _conv_silu(x_buf, row, weight, channel, cache, channel_stride, time_stride):
    offset = channel * channel_stride
    h0 = fx.ptr_load(cache + offset)
    h1 = fx.ptr_load(cache + offset + time_stride)
    h2 = fx.ptr_load(cache + offset + 2 * time_stride)
    x = fx.ptr_load(x_buf + row + channel)
    weights = load_conv_weights(weight, channel)
    result = conv_result(h0, h1, h2, x, weights)
    fx.ptr_store(h1, cache + offset)
    fx.ptr_store(h2, cache + offset + time_stride)
    fx.ptr_store(x, cache + offset + 2 * time_stride)
    return result


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
    value_heads = key_heads * VALUES_PER_KEY
    qk_width = 2 * key_heads * HEAD_DIM
    conv_width = qk_width + value_heads * HEAD_DIM
    projection_width = conv_width + value_heads * HEAD_DIM
    # Each block owns one key head and its three value heads. This gives the
    # Q/K convolution history a single writer and lets its consumers share LDS.
    heads_per_block = VALUES_PER_KEY
    threads_per_head = 256
    k_lanes = 8
    threads = heads_per_block * threads_per_head
    rows_per_step = threads_per_head // k_lanes
    partials = HEAD_DIM // WAVE_SIZE
    vector_steps = HEAD_DIM // (k_lanes * VECTOR_WIDTH)
    state_element_size = 4 if fp32_state else 2

    @fx.struct
    class Shared:
        qk: fx.Array[Float32, 2 * HEAD_DIM, 16]
        qk_squares: fx.Array[Float32, 2 * partials, 16]
        v: fx.Array[Float32, heads_per_block * HEAD_DIM, 16]
        output: fx.Array[Float32, heads_per_block * HEAD_DIM, 16]
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
        key_head = head // VALUES_PER_KEY
        group = token * value_heads + head
        head_offset = local_head * HEAD_DIM
        slot = Int64(rocdl.s_buffer_load(buffer_ptr(indices, T.i32), token))
        q_ptr = buffer_ptr(quantized, T.i32) + group * WAVE_SIZE + head_thread
        scale_ptr = buffer_ptr(scales, T.f32) + group
        if slot > 0:
            shared = fx.SharedAllocator().allocate(Shared).peek()
            x_buf = buffer_ptr(qkvz, T.bf16)
            w_buf = buffer_ptr(conv_weight, T.bf16)
            conv_buf = buffer_ptr(conv_state + slot * conv_slot_stride * 2, T.bf16)
            state_buf = buffer_ptr(
                state + slot * state_slot_stride * state_element_size, state_type
            )
            input_base = token * projection_width

            # Give Q and K separate producer waves and each channel one writer.
            if tid < 2 * HEAD_DIM:
                qk_head = key_head + tid // HEAD_DIM * key_heads
                channel = qk_head * HEAD_DIM + tid % HEAD_DIM
                qk = Float32(
                    _conv_silu(
                        x_buf,
                        input_base,
                        w_buf,
                        channel,
                        conv_buf,
                        conv_channel_stride,
                        conv_time_stride,
                    )
                )
                fx.ptr_store(qk, shared.qk.ptr + tid)
                square_sum = sum_lanes(qk * qk, WAVE_SIZE)
                if tid % WAVE_SIZE == 0:
                    fx.ptr_store(square_sum, shared.qk_squares.ptr + tid // WAVE_SIZE)

            if head_thread < HEAD_DIM:
                channel = qk_width + head * HEAD_DIM + head_thread
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
                ba_ptr = buffer_ptr(ba, T.bf16)
                b = Float32(fx.ptr_load(ba_ptr + b_offset))
                a = Float32(fx.ptr_load(ba_ptr + b_offset + value_heads))
                bias = Float32(fx.ptr_load(buffer_ptr(dt_bias, bias_type) + head))
                log_rate = fx.ptr_load(buffer_ptr(a_log, T.f32) + head)
                delta = a + bias
                softplus = delta
                if delta <= Float32(20.0):
                    softplus = Float32(fmath.log(Float32(1.0) + exp(delta)))
                fx.ptr_store(
                    exp(-exp(log_rate) * softplus), shared.decay.ptr + local_head
                )
                fx.ptr_store(sigmoid(b), shared.beta.ptr + local_head)
            lds_barrier()

            q_square, k_square = Float32(0.0), Float32(0.0)
            for wave in range_constexpr(partials):
                q_square += fx.ptr_load(shared.qk_squares.ptr + wave)
                k_square += fx.ptr_load(shared.qk_squares.ptr + partials + wave)
            q_scale = rsqrt(q_square + 1e-6) * HEAD_DIM**-0.5
            k_scale = rsqrt(k_square + 1e-6)
            k_lane = head_thread % k_lanes
            q_parts, k_parts, offsets = [], [], []
            for part in range_constexpr(vector_steps):
                offset = (part * k_lanes + k_lane) * VECTOR_WIDTH
                offsets.append(offset)
                q_parts.append(load_fp32_vec(shared.qk.ptr + offset) * q_scale)
                k_parts.append(
                    load_fp32_vec(shared.qk.ptr + HEAD_DIM + offset) * k_scale
                )
            decay = fx.ptr_load(shared.decay.ptr + local_head)
            beta = fx.ptr_load(shared.beta.ptr + local_head)
            for step in fx.range(HEAD_DIM // rows_per_step, unroll=1):
                row = step * rows_per_step + head_thread // k_lanes
                state_base = (head * HEAD_DIM + row) * HEAD_DIM
                states = []
                dot = Float32(0.0)
                for part in range_constexpr(vector_steps):
                    state_vec = load_fp32_vec(state_buf + state_base + offsets[part])
                    state_vec *= decay
                    states.append(state_vec)
                    dot += (state_vec * k_parts[part]).reduce(fx.ReductionOp.ADD)
                v = fx.ptr_load(shared.v.ptr + head_offset + row)
                residual = (v - sum_lanes(dot, k_lanes)) * beta
                result = Float32(0.0)
                for part in range_constexpr(vector_steps):
                    updated = states[part] + k_parts[part] * residual
                    result += (updated * q_parts[part]).reduce(fx.ReductionOp.ADD)
                    if not fp32_state:
                        updated = updated.to(fx.BFloat16)
                    fx.ptr_store(updated, state_buf + state_base + offsets[part])
                result = Float32(fx.BFloat16(sum_lanes(result, k_lanes)))
                if k_lane == 0:
                    fx.ptr_store(result, shared.output.ptr + head_offset + row)
            lds_barrier()

            if head_thread < WAVE_SIZE:
                col = head_thread * VECTOR_WIDTH
                values = load_fp32_vec(shared.output.ptr + head_offset + col)
                z_offset = input_base + conv_width + head * HEAD_DIM + col
                gates = load_fp32_vec(x_buf + z_offset)
                weights = load_fp32_vec(buffer_ptr(norm_weight, norm_type) + col)
                packed, scale = _gated_norm_fp8(
                    values, gates, weights, Float32(norm_eps)
                )
                fx.ptr_store(packed, q_ptr)
                if head_thread == 0:
                    fx.ptr_store(scale, scale_ptr)
        else:
            if head_thread < WAVE_SIZE:
                fx.ptr_store(Int32(0), q_ptr)
                if head_thread == 0:
                    fx.ptr_store(Float32(1e-10 / 448.0), scale_ptr)

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
    key_heads = scales.shape[1] // VALUES_PER_KEY
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
