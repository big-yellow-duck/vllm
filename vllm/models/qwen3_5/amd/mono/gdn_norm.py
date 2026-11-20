# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused GDN gated RMSNorm and group-128 FP8 activation quantization."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Float32, Int32, Int64, Stream, T

from .common import HEAD_DIM, VECTOR_WIDTH, WAVE_SIZE, buffer_ptr, load_fp32_vec

_FP8_MAX = 448.0


def _gated_norm_fp8(values, gates, weights, eps):
    square_sum = (values * values).reduce(fx.ReductionOp.ADD)
    for shift in range_constexpr(4, -1, -1):
        square_sum += square_sum.shuffle_xor(1 << shift, WAVE_SIZE)
    variance = square_sum * (1.0 / HEAD_DIM) + eps
    rstd = Float32(rocdl.rsq(T.f32, variance.ir_value()))

    normalized = []
    absmax = Float32(1e-10)
    for i in range_constexpr(VECTOR_WIDTH):
        exponent = gates[i] * -1.4426950408889634
        exp = Float32(rocdl.exp2(T.f32, exponent.ir_value()))
        denominator = Float32(1.0) + exp
        sigmoid = Float32(rocdl.rcp(T.f32, denominator.ir_value()))
        value = values[i] * rstd * weights[i] * (gates[i] * sigmoid)
        # Match the BF16 intermediate consumed by the unfused quantizer.
        rounded = Float32(fx.BFloat16(value))
        normalized.append(rounded)
        absmax = Float32(fx.max(absmax, fmath.absf(rounded)))
    for shift in range_constexpr(4, -1, -1):
        peer = absmax.shuffle_xor(1 << shift, WAVE_SIZE)
        absmax = Float32(fx.max(absmax, peer))
    scale = absmax * (1.0 / _FP8_MAX)
    reciprocal = Float32(rocdl.rcp(T.f32, scale.ir_value()))
    fp8 = [value * reciprocal for value in normalized]
    lo = Int32(rocdl.cvt_pk_fp8_f32(T.i32, fp8[0], fp8[1], Int32(0), False))
    packed = Int32(rocdl.cvt_pk_fp8_f32(T.i32, fp8[2], fp8[3], lo, True))
    return packed, scale


@lru_cache(maxsize=8)
def _build_gated_norm_quant(device: int, fp32_weight: bool, waves_per_block: int):
    threads = waves_per_block * WAVE_SIZE

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def qwen_gdn_gated_norm_quant(
        x: Int64,
        gate: Int64,
        weight: Int64,
        quantized: Int64,
        scales: Int64,
        stride_x: Int32,
        stride_gate: Int32,
        heads: Int32,
        groups: Int32,
        eps: Float32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % WAVE_SIZE
        group = gpu.block_idx.x * waves_per_block + tid // WAVE_SIZE
        if group < groups:
            token = group // heads
            column = (group % heads) * HEAD_DIM + lane * VECTOR_WIDTH
            values = load_fp32_vec(buffer_ptr(x, T.bf16) + token * stride_x + column)
            gates = load_fp32_vec(
                buffer_ptr(gate, T.bf16) + token * stride_gate + column
            )
            weight_type = T.f32 if fp32_weight else T.bf16
            weights = load_fp32_vec(
                buffer_ptr(weight, weight_type) + lane * VECTOR_WIDTH
            )

            packed, scale = _gated_norm_fp8(values, gates, weights, eps)
            fx.ptr_store(
                packed, buffer_ptr(quantized, T.i32) + group * WAVE_SIZE + lane
            )
            if lane == 0:
                fx.ptr_store(scale, buffer_ptr(scales, T.f32) + group)

    @flyc.jit
    def launch(
        x: Int64,
        gate: Int64,
        weight: Int64,
        quantized: Int64,
        scales: Int64,
        stride_x: Int32,
        stride_gate: Int32,
        heads: Int32,
        groups: Int32,
        eps: Float32,
        stream: Stream,
    ):
        qwen_gdn_gated_norm_quant(
            x,
            gate,
            weight,
            quantized,
            scales,
            stride_x,
            stride_gate,
            heads,
            groups,
            eps,
        ).launch(
            grid=((groups + waves_per_block - 1) // waves_per_block, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    with torch.accelerator.device_index(device):
        return flyc.compile(
            launch, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1e-6, Stream(None), compile_only=True
        )


def gated_norm_quant(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    eps: float,
) -> None:
    if not x.numel():
        return
    heads = x.shape[1] // HEAD_DIM
    launch = _build_gated_norm_quant(x.get_device(), weight.dtype == torch.float32, 1)
    launch(
        x.data_ptr(),
        gate.data_ptr(),
        weight.data_ptr(),
        quantized.data_ptr(),
        scales.data_ptr(),
        x.stride(0),
        gate.stride(0),
        heads,
        x.shape[0] * heads,
        eps,
        torch.cuda.current_stream(x.device),
    )
