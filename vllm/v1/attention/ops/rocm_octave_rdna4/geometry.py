# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.rocm_octave import HADAMARD_BLOCK, OctaveFormat


@dataclass(frozen=True)
class OctaveGeometry:
    """Map logical Q/K/V dimensions onto the existing Octave wire layouts."""

    head_dim: int
    rope_dim: int

    def __post_init__(self) -> None:
        if not 1 <= self.head_dim <= 1024 or not 0 <= self.rope_dim <= self.head_dim:
            raise ValueError(
                f"Unsupported RDNA4 Octave geometry: D={self.head_dim}, "
                f"RoPE={self.rope_dim}"
            )
        if self._blocked_dims()[0] > 1024:
            raise ValueError("Padded Octave RoPE/NoPE dimensions exceed 1024")

    def _blocked_dims(self) -> tuple[int, int]:
        rope = round_up(self.rope_dim, HADAMARD_BLOCK)
        nope = round_up(self.head_dim - self.rope_dim, HADAMARD_BLOCK)
        return rope + nope, rope

    def layout(self, cache_dtype: str) -> OctaveFormat:
        if cache_dtype == "octave_k3v3_compact":
            head_dim = max(HADAMARD_BLOCK, 1 << (self.head_dim - 1).bit_length())
            rope_dim = 0
        else:
            head_dim, rope_dim = self._blocked_dims()
        return OctaveFormat.from_cache_dtype(cache_dtype, head_dim, rope_dim)
