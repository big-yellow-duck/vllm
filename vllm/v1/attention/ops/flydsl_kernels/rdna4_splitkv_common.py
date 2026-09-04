# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared constants and helpers for RDNA4 SplitKV kernels."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

HEAD_DIM = 256
TILE_TOKENS = 64
PARTITIONS = 4
WAVE_SIZE = 32
NUM_WAVES = 8
BLOCK_THREADS = WAVE_SIZE * NUM_WAVES
LOG2E = 1.4426950408889634
NATIVE_BF16_LDS_BYTES = 14_336


def _dequant_fp8x8(raw, scale, output_type, *, is_fp8fnuz: bool):
    """Decode packed FP8 with the scalar conversion supported by gfx1201 wave32."""
    if const_expr(is_fp8fnuz):
        scale = scale * fx.Float32(0.5)
    words = raw.bitcast(fx.Int32)
    values = []
    for word_index in range_constexpr(2):
        for byte_index in range_constexpr(4):
            value = fx.rocdl.cvt_f32_fp8(
                words[word_index],
                byte_sel=byte_index,
            )
            values.append(value * scale)
    return fx.Vector.from_elements(values, dtype=fx.Float32).to(output_type)


def _flat_view(tensor: fx.Tensor) -> fx.Tensor:
    return fx.make_view(fx.get_iter(tensor), fx.make_layout(1 << 30, 1))


def _wave_reduce(value, mode: str):
    result = value
    for offset in (16, 8, 4, 2, 1):
        peer = result.shuffle_xor(offset, WAVE_SIZE)
        result = fx.max(result, peer) if const_expr(mode == "max") else result + peer
    return result
