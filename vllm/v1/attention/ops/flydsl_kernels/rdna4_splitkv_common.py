# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared constants and helpers for RDNA4 SplitKV kernels."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

HEAD_DIM = 256
WAVE_SIZE = 32
LOG2E = 1.4426950408889634


def _dequant_fp8x8(raw, scale, output_type, *, is_fp8fnuz: bool):
    """Decode eight packed FP8 values with four gfx1201 packed conversions."""
    if const_expr(is_fp8fnuz):
        scale = scale * fx.Float32(0.5)
    words = raw.bitcast(fx.Int32)
    values = []
    for word_index in range_constexpr(2):
        for word_half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_pk_f32_fp8(
                    fx.Vector.make_type(2, fx.Float32),
                    words[word_index],
                    bool(word_half),
                )
            )
            values.append(pair[0] * scale)
            values.append(pair[1] * scale)
    return fx.Vector.from_elements(values, dtype=fx.Float32).to(output_type)


def _flat_view(tensor: fx.Tensor) -> fx.Tensor:
    return fx.make_view(fx.get_iter(tensor), fx.make_layout(1 << 30, 1))


def _wave_reduce(value, mode: str):
    result = value
    for offset in (16, 8, 4, 2, 1):
        peer = fx.gpu.shuffle_xor(result, offset, WAVE_SIZE)
        result = fx.max(result, peer) if const_expr(mode == "max") else result + peer
    return result
