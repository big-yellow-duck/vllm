# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL helpers shared by the offline prefill prototypes."""

import flydsl.expr as fx
from flydsl.expr import const_expr

LOG2E = 1.4426950408889634
WAVE_SIZE = 32


def _flat_view(tensor: fx.Tensor) -> fx.Tensor:
    return fx.make_view(fx.get_iter(tensor), fx.make_layout(1 << 30, 1))


def _wave_reduce(value, mode: str):
    result = value
    for offset in (16, 8, 4, 2, 1):
        peer = fx.gpu.shuffle_xor(result, offset, WAVE_SIZE)
        result = fx.max(result, peer) if const_expr(mode == "max") else result + peer
    return result
