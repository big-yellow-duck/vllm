# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical attention dimensions and compatible physical Octave wire layouts."""

from dataclasses import dataclass

CACHE_CODES = {"octave_k3v3": 3, "octave_k3v4": 4, "octave_k3v3_compact": 19}


def round64(n):
    return (n + 63) // 64 * 64


@dataclass(frozen=True)
class OctaveGeometry:
    head_dim: int = 256
    rope_dim: int = 64

    def __post_init__(self):
        if not isinstance(self.head_dim, int) or not 1 <= self.head_dim <= 1024:
            raise ValueError("head_dim must be an integer in 1..1024")
        if (
            not isinstance(self.rope_dim, int)
            or not 0 <= self.rope_dim <= self.head_dim
        ):
            raise ValueError("rope_dim must be an integer in 0..head_dim")
        if self.block_head_dim > 1024:
            raise ValueError("Padded RoPE/NoPE segments exceed 1024 storage dimensions")

    @property
    def block_head_dim(self):
        return round64(self.rope_dim) + round64(self.head_dim - self.rope_dim)

    def layout(self, cache_dtype):
        if cache_dtype not in CACHE_CODES:
            raise ValueError(f"Unknown Octave cache dtype: {cache_dtype}")
        return OctaveLayout(self, cache_dtype)


@dataclass(frozen=True)
class OctaveLayout:
    geometry: OctaveGeometry
    cache_dtype: str

    @property
    def compact(self):
        return self.cache_dtype == "octave_k3v3_compact"

    @property
    def v_bits(self):
        return 4 if self.cache_dtype == "octave_k3v4" else 3

    @property
    def head_dim(self):
        return self.geometry.head_dim

    @property
    def rope_dim(self):
        return self.geometry.rope_dim

    @property
    def storage_head_dim(self):
        if self.compact:
            return max(64, 1 << (self.head_dim - 1).bit_length())
        return self.geometry.block_head_dim

    @property
    def storage_rope_dim(self):
        return 0 if self.compact else round64(self.rope_dim)

    @property
    def num_k_scales(self):
        return 1 if self.compact else self.storage_head_dim // 64

    @property
    def off_k_nope(self):
        return self.storage_rope_dim // 2

    @property
    def off_v(self):
        return (
            self.off_k_nope + (self.storage_head_dim - self.storage_rope_dim) * 3 // 8
        )

    @property
    def off_scales(self):
        return self.off_v + self.storage_head_dim * self.v_bits // 8

    @property
    def slot_bytes(self):
        raw = self.off_scales + 2 * (self.num_k_scales + 1)
        align = 4 if self.compact else 8
        return (raw + align - 1) // align * align

    @property
    def kernel_code(self):
        return CACHE_CODES[self.cache_dtype]

    @property
    def needs_padding(self):
        return self.storage_head_dim != self.head_dim

    @property
    def logical_indices(self):
        if self.compact:
            return tuple(range(self.head_dim))
        r, sr = self.rope_dim, self.storage_rope_dim
        return tuple(range(r)) + tuple(range(sr, sr + self.head_dim - r))

    def to_storage(self, x):
        """Reference-only embedding; HIP fuses this mapping into tensor loads."""
        if x.shape[-1] != self.head_dim:
            raise ValueError("Logical head dimension does not match the layout")
        if not self.needs_padding:
            return x
        out = x.new_zeros((*x.shape[:-1], self.storage_head_dim))
        if self.compact:
            out[..., : self.head_dim] = x
        else:
            r, sr = self.rope_dim, self.storage_rope_dim
            out[..., :r] = x[..., :r]
            out[..., sr : sr + self.head_dim - r] = x[..., r:]
        return out

    def from_storage(self, x):
        if x.shape[-1] != self.storage_head_dim:
            raise ValueError("Storage head dimension does not match the layout")
        if not self.needs_padding:
            return x
        if self.compact:
            return x[..., : self.head_dim]
        import torch

        r, sr = self.rope_dim, self.storage_rope_dim
        return torch.cat((x[..., :r], x[..., sr : sr + self.head_dim - r]), dim=-1)

    def reference_format(self):
        from octave_reference import OctaveFormat

        return OctaveFormat.from_cache_dtype(
            self.cache_dtype, self.storage_head_dim, self.storage_rope_dim
        )

    def sign_bits(self):
        """CPU sign words for the wire format, independent of the test oracle."""
        import torch

        generator = torch.Generator(device="cpu").manual_seed(0x5E17)
        negative = torch.randint(
            0, 2, (self.storage_head_dim,), generator=generator, dtype=torch.int64
        ).view(-1, 32)
        words = (negative << torch.arange(32)).sum(-1)
        return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
