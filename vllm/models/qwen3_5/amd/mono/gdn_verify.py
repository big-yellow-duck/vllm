# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-launch RDNA4 Qwen GDN verification with a dynamic FP8 epilogue."""

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


def _select_k_lanes(requests: int, query_length: int) -> int:
    """Choose the number of lanes sharing each recurrent-state row."""
    if requests == 1:
        return 2
    if requests >= 4 and query_length <= 2:
        return 8
    return 4


@lru_cache(maxsize=32)
def _build_verify(
    device: int,
    key_heads: int,
    max_tokens: int,
    channel_stride: int,
    time_stride: int,
    fp32_state: bool,
    fp32_bias: bool,
    fp32_norm: bool,
    norm_eps: float,
    k_lanes: int,
):
    value_heads = key_heads * VALUES_PER_KEY
    qk_width = 2 * key_heads * HEAD_DIM
    conv_width = qk_width + value_heads * HEAD_DIM
    input_width = conv_width + value_heads * HEAD_DIM
    threads_per_head = 256
    threads = VALUES_PER_KEY * threads_per_head
    rows_per_step = threads_per_head // k_lanes
    vector_steps = HEAD_DIM // (k_lanes * VECTOR_WIDTH)
    qk_channels = 2 * HEAD_DIM
    value_channels = VALUES_PER_KEY * HEAD_DIM
    local_channels = qk_channels + value_channels
    state_bytes = 4 if fp32_state else 2

    @fx.struct
    class Shared:
        projected: fx.Array[Float32, max_tokens * local_channels, 16]
        decay: fx.Array[Float32, max_tokens * VALUES_PER_KEY, 16]
        beta: fx.Array[Float32, max_tokens * VALUES_PER_KEY, 16]
        output: fx.Array[Float32, max_tokens * value_channels, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def qwen_gdn_mono_verify(
        qkvz: Int64,
        ba: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state: Int64,
        indices: Int64,
        starts: Int64,
        accepted: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        quantized: Int64,
        scales: Int64,
        conv_slot_stride: Int64,
        state_slot_stride: Int64,
        index_stride: Int64,
    ):
        req, key_head = gpu.block_idx.x, gpu.block_idx.y
        tid = gpu.thread_idx.x
        # Every lane in the wave reads the same request metadata.
        start_buf = buffer_ptr(starts, T.i32)
        begin = rocdl.s_buffer_load(start_buf, req)
        count = rocdl.s_buffer_load(start_buf, req + 1) - begin
        index_buf = buffer_ptr(indices, T.i32)
        index_row = Int32(req) * Int32(index_stride)
        conv_slot = Int64(rocdl.s_buffer_load(index_buf, index_row))
        prefix = rocdl.s_buffer_load(buffer_ptr(accepted, T.i32), req)
        valid_prefix = (prefix > 0) & (prefix <= max_tokens)
        source_column = valid_prefix.select(prefix - 1, Int32(0))
        source = Int64(rocdl.s_buffer_load(index_buf, index_row + source_column))
        if (
            (count > 0)
            & (count <= max_tokens)
            & (conv_slot > 0)
            & valid_prefix
            & (source > 0)
        ):
            shared = fx.SharedAllocator().allocate(Shared).peek()
            x_buf = buffer_ptr(qkvz, T.bf16)
            ba_buf = buffer_ptr(ba, T.bf16)
            w_buf = buffer_ptr(conv_weight, T.bf16)
            conv_buf = buffer_ptr(conv_state + conv_slot * conv_slot_stride * 2, T.bf16)
            local_head = tid // threads_per_head
            head = key_head * VALUES_PER_KEY + local_head
            head_thread = tid % threads_per_head
            head_offset = local_head * HEAD_DIM

            # Each lane owns one convolution channel and reads its
            # accepted history before shifting the rolling window.
            if tid < local_channels:
                within_head = tid % HEAD_DIM
                part = tid // HEAD_DIM
                qk_channel = (part * key_heads + key_head) * HEAD_DIM
                v_channel = qk_width + (key_head * VALUES_PER_KEY + part - 2) * HEAD_DIM
                channel = (part < 2).select(qk_channel, v_channel) + within_head
                cache_offset = channel * channel_stride
                history = cache_offset + (prefix - 1) * time_stride
                h0 = fx.ptr_load(conv_buf + history)
                h1 = fx.ptr_load(conv_buf + history + time_stride)
                h2 = fx.ptr_load(conv_buf + history + 2 * time_stride)
                weights = load_conv_weights(w_buf, channel)
                fx.ptr_store(h1, conv_buf + cache_offset)
                fx.ptr_store(h2, conv_buf + cache_offset + time_stride)
                for t, h in fx.range(Int32(0), count, Int32(1), init=[h0, h1, h2]):
                    t = Int32(t)
                    x = fx.ptr_load(x_buf + (begin + t) * input_width + channel)
                    value = conv_result(h[0], h[1], h[2], x, weights)
                    fx.ptr_store(
                        Float32(value),
                        shared.projected.ptr + t * local_channels + tid,
                    )
                    fx.ptr_store(x, conv_buf + cache_offset + (t + 2) * time_stride)
                    yield [h[1], h[2], x]

            if tid < count * VALUES_PER_KEY:
                t = tid // VALUES_PER_KEY
                gate_head = key_head * VALUES_PER_KEY + tid % VALUES_PER_KEY
                offset = (begin + t) * (2 * value_heads) + gate_head
                b = Float32(fx.ptr_load(ba_buf + offset))
                a = Float32(fx.ptr_load(ba_buf + offset + value_heads))
                bias_type = T.f32 if fp32_bias else T.bf16
                bias = Float32(fx.ptr_load(buffer_ptr(dt_bias, bias_type) + gate_head))
                log_rate = fx.ptr_load(buffer_ptr(a_log, T.f32) + gate_head)
                delta = a + bias
                softplus = delta
                if delta <= Float32(20.0):
                    softplus = Float32(fmath.log(Float32(1.0) + exp(delta)))
                fx.ptr_store(exp(-exp(log_rate) * softplus), shared.decay.ptr + tid)
                fx.ptr_store(sigmoid(b), shared.beta.ptr + tid)
            lds_barrier()

            # Normalize each token's Q and K once for all three V heads.
            lane, wave = tid % WAVE_SIZE, tid // WAVE_SIZE
            if wave < count * 2:
                pos = wave // 2 * local_channels + wave % 2 * HEAD_DIM
                ptr = shared.projected.ptr + pos + lane * VECTOR_WIDTH
                qk = load_fp32_vec(ptr)
                sq = sum_lanes((qk * qk).reduce(fx.ReductionOp.ADD), WAVE_SIZE)
                scale = rsqrt(sq + 1e-6)
                scale *= (wave % 2 == 0).select(Float32(HEAD_DIM**-0.5), Float32(1.0))
                scaled = qk * scale
                for i in range_constexpr(VECTOR_WIDTH):
                    fx.ptr_store(scaled[i], ptr + i)
            lds_barrier()

            state_type = T.f32 if fp32_state else T.bf16
            src_buf = buffer_ptr(
                state + source * state_slot_stride * state_bytes, state_type
            )
            k_lane = head_thread % k_lanes
            offsets = [
                (part * k_lanes + k_lane) * VECTOR_WIDTH for part in range(vector_steps)
            ]
            for step in fx.range(HEAD_DIM // rows_per_step, unroll=1):
                row = Int32(step) * rows_per_step + head_thread // k_lanes
                state_offset = (head * HEAD_DIM + row) * HEAD_DIM
                tile = fx.Vector.from_elements(
                    [
                        v
                        for offset in offsets
                        for v in load_fp32_vec(src_buf + state_offset + offset)
                    ],
                    Float32,
                )
                # Keep the FP32 state resident across verification.
                # The engine chooses one of these snapshots next round.
                for t, tiles in fx.range(Int32(0), count, Int32(1), init=[tile]):
                    t = Int32(t)
                    # Start the uniform snapshot-slot load before the dot products.
                    dest = Int64(rocdl.s_buffer_load(index_buf, index_row + t))
                    x_base = shared.projected.ptr + t * local_channels
                    q = fx.Vector.from_elements(
                        [
                            v
                            for offset in offsets
                            for v in load_fp32_vec(x_base + offset)
                        ],
                        Float32,
                    )
                    k = fx.Vector.from_elements(
                        [
                            v
                            for offset in offsets
                            for v in load_fp32_vec(x_base + HEAD_DIM + offset)
                        ],
                        Float32,
                    )
                    v = fx.ptr_load(x_base + qk_channels + head_offset + row)
                    param = t * VALUES_PER_KEY + local_head
                    h = fx.Vector(tiles[0]) * fx.ptr_load(shared.decay.ptr + param)
                    dot = sum_lanes((h * k).reduce(fx.ReductionOp.ADD), k_lanes)
                    delta = (v - dot) * fx.ptr_load(shared.beta.ptr + param)
                    h += k * delta
                    o = sum_lanes((h * q).reduce(fx.ReductionOp.ADD), k_lanes)
                    if k_lane == 0:
                        fx.ptr_store(
                            Float32(fx.BFloat16(o)),
                            shared.output.ptr + t * value_channels + head_offset + row,
                        )
                    if dest > 0:
                        dst = buffer_ptr(
                            state + dest * state_slot_stride * state_bytes, state_type
                        )
                        cast = h if fp32_state else h.to(fx.BFloat16)
                        vectors = cast.reshape((vector_steps, VECTOR_WIDTH))
                        for part in range_constexpr(vector_steps):
                            fx.ptr_store(
                                vectors[part, None], dst + state_offset + offsets[part]
                            )
                    yield [h]
            lds_barrier()

            # A wave emits one token's complete gated FP8 value head.
            t = head_thread // WAVE_SIZE
            if t < count:
                token = begin + t
                col = lane * VECTOR_WIDTH
                group = token * value_heads + head
                values = load_fp32_vec(
                    shared.output.ptr + t * value_channels + head_offset + col
                )
                gates = load_fp32_vec(
                    x_buf + token * input_width + conv_width + head * HEAD_DIM + col,
                )
                norm_type = T.f32 if fp32_norm else T.bf16
                weights = load_fp32_vec(buffer_ptr(norm_weight, norm_type) + col)
                packed, scale = _gated_norm_fp8(
                    values, gates, weights, Float32(norm_eps)
                )
                fx.ptr_store(
                    packed, buffer_ptr(quantized, T.i32) + group * WAVE_SIZE + lane
                )
                if lane == 0:
                    fx.ptr_store(scale, buffer_ptr(scales, T.f32) + group)

    @flyc.jit
    def launch(
        qkvz: Int64,
        ba: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state: Int64,
        indices: Int64,
        starts: Int64,
        accepted: Int64,
        a_log: Int64,
        dt_bias: Int64,
        norm_weight: Int64,
        quantized: Int64,
        scales: Int64,
        conv_slot_stride: Int64,
        state_slot_stride: Int64,
        index_stride: Int64,
        requests: Int32,
        stream: Stream,
    ):
        qwen_gdn_mono_verify(
            qkvz,
            ba,
            conv_weight,
            conv_state,
            state,
            indices,
            starts,
            accepted,
            a_log,
            dt_bias,
            norm_weight,
            quantized,
            scales,
            conv_slot_stride,
            state_slot_stride,
            index_stride,
        ).launch(grid=(requests, key_heads, 1), block=(threads, 1, 1), stream=stream)

    with torch.accelerator.device_index(device):
        return flyc.compile(launch, *([0] * 16), 1, Stream(None), compile_only=True)


def flydsl_gdn_spec_decode(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
    accepted: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    norm_eps: float,
    *,
    query_length: int | None = None,
) -> None:
    """Verify ordered token groups and publish each potential accepted prefix."""
    requests, capacity = indices.shape
    key_heads = scales.shape[1] // VALUES_PER_KEY
    k_lanes = 4
    if key_heads == 8 and state.dtype == torch.bfloat16:
        k_lanes = _select_k_lanes(requests, query_length or capacity)
    launch = _build_verify(
        qkvz.get_device(),
        key_heads,
        capacity,
        conv_state.stride(1),
        conv_state.stride(2),
        state.dtype == torch.float32,
        dt_bias.dtype == torch.float32,
        norm_weight.dtype == torch.float32,
        norm_eps,
        k_lanes,
    )
    launch(
        qkvz.data_ptr(),
        ba.data_ptr(),
        conv_weight.data_ptr(),
        conv_state.data_ptr(),
        state.data_ptr(),
        indices.data_ptr(),
        starts.data_ptr(),
        accepted.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        norm_weight.data_ptr(),
        quantized.data_ptr(),
        scales.data_ptr(),
        conv_state.stride(0),
        state.stride(0),
        indices.stride(0),
        requests,
        torch.cuda.current_stream(qkvz.device),
    )
