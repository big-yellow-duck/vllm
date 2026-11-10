# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized RDNA4 Octave attention with bounded, graph-stable scheduling."""

from pathlib import Path

from .hip import HipOctave, build_library, cache_root
from .splits import DEFAULT_WORKSPACE_BYTES, recommend_splits

ROOT = Path(__file__).resolve().parent


def build_tuned_library(build_dir=None, arch="gfx1201", *, head_dim=256, rope_dim=64):
    """Compile before graph capture; 256/64 retains the measured fast source."""
    from .geometry import OctaveGeometry

    geometry = OctaveGeometry(head_dim, rope_dim)
    build_dir = cache_root() / "tuned" if build_dir is None else Path(build_dir)
    if (geometry.head_dim, geometry.rope_dim) == (256, 64):
        return build_library(ROOT / "native/legacy", build_dir, arch)
    from .geometry_build import build_geometry_library

    return build_geometry_library(
        head_dim, rope_dim, Path(build_dir) / "geometry", arch
    )


class TunedHipOctave(HipOctave):
    """Graph-stable scheduling with a configurable split-workspace budget.

    Explicit integer split counts bypass the policy and its workspace limit.
    Automatic selection never reads a tensor's device values or free VRAM.
    """

    def __init__(
        self,
        library,
        *,
        workspace_limit_bytes=DEFAULT_WORKSPACE_BYTES,
        geometry_library=None,
    ):
        if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
            raise ValueError("workspace_limit_bytes must be a nonnegative integer")
        super().__init__(library)
        self.workspace_limit_bytes = workspace_limit_bytes
        self._geometry_backend = (
            HipOctave(geometry_library) if geometry_library is not None else None
        )
        if (
            self._geometry_backend is not None
            and self._geometry_backend.geometry != self.geometry
        ):
            raise ValueError("Companion library must use the same logical geometry")

    @classmethod
    def build(
        cls,
        head_dim=256,
        rope_dim=64,
        *,
        build_dir=None,
        arch="gfx1201",
        workspace_limit_bytes=DEFAULT_WORKSPACE_BYTES,
    ):
        """Build all dispatch paths eagerly, before capture or timing."""
        from .geometry_build import build_geometry_library

        build_dir = cache_root() / "tuned" if build_dir is None else Path(build_dir)
        library = build_tuned_library(
            build_dir, arch, head_dim=head_dim, rope_dim=rope_dim
        )
        companion = None
        if (head_dim, rope_dim) == (256, 64):
            companion = build_geometry_library(
                head_dim, rope_dim, Path(build_dir) / "geometry", arch
            )
        return cls(
            library,
            workspace_limit_bytes=workspace_limit_bytes,
            geometry_library=companion,
        )

    def _use_geometry(self, query, cache, table, prefill):
        if self.geometry_version:
            return True
        heads, kv_heads = query.shape[1], cache.shape[1]
        if self._geometry_backend is None or kv_heads <= 0 or heads % kv_heads:
            return False
        group = heads // kv_heads
        if prefill:
            tokens, requests = query.shape[0], table.shape[0]
            capacity = table.shape[1] * cache.shape[2]
            # Very small per-request MHA chunks underfill the query tile.
            if group == 1 and tokens < 32 and tokens // max(1, requests) < 8:
                return False
            # The existing packed FP32 paths win on short K3V3 contexts.
            # At 256/64 the three wire formats have distinct slot sizes.
            if (
                group <= 2
                and 512 <= capacity <= 2048
                and cache.shape[-1] != self.layout("octave_k3v4").slot_bytes
            ):
                return False
            if (
                group in (1, 2, 4, 8)
                and tokens >= 16
                and tokens > requests
                and tokens // max(1, requests) >= 4
                and (tokens < 256 or tokens // max(1, requests) < 32)
            ):
                return True
            return group not in (1, 2, 4, 6, 8)
        if group in (1, 2, 3, 6):
            return False
        capacity = table.shape[1] * cache.shape[2]
        tokens = query.shape[0]
        # Matrix setup needs enough reuse across heads or query rows. Keep
        # short-context and small-GQA latency on the original packed path.
        if group < 8:
            return tokens >= 32 and capacity >= 1024
        if group < 16:
            return tokens >= 256 or (tokens >= 8 and capacity >= 1024)
        return (
            capacity >= 65536
            or (capacity >= 512 and (tokens >= 4 or group >= 64))
            or tokens * group >= 256
        )

    def suggest_splits(
        self,
        query,
        cache,
        block_table,
        cache_dtype,
        *,
        prefill=False,
        max_context_tokens=None,
    ):
        if query.ndim != 3 or cache.ndim != 4 or block_table.ndim != 2:
            raise ValueError("Expected 3D query, 4D cache and 2D page table")
        return recommend_splits(
            tokens=query.shape[0],
            heads=query.shape[1],
            kv_heads=cache.shape[1],
            requests=block_table.shape[0],
            capacity=block_table.shape[1] * cache.shape[2],
            page_size=cache.shape[2],
            cache_dtype=cache_dtype,
            prefill=prefill,
            workspace_limit_bytes=self.workspace_limit_bytes,
            max_context_tokens=max_context_tokens,
            head_dim=self.head_dim,
            storage_head_dim=self.layout(cache_dtype).storage_head_dim,
            legacy_fast=not self._use_geometry(query, cache, block_table, prefill),
        )

    def decode(self, *args, splits="auto", **kwargs):
        return super().decode(*args, splits=splits, **kwargs)

    def window_store(self, *args, **kwargs):
        if self._geometry_backend is not None:
            return self._geometry_backend.window_store(*args, **kwargs)
        return super().window_store(*args, **kwargs)

    def decode_sparse(self, *args, **kwargs):
        if self._geometry_backend is not None:
            return self._geometry_backend.decode_sparse(*args, **kwargs)
        return super().decode_sparse(*args, **kwargs)

    def prefill(self, *args, splits="auto", **kwargs):
        return super().prefill(*args, splits=splits, **kwargs)

    def _attention(
        self,
        query,
        cache,
        table,
        mapping,
        lengths,
        signs,
        cache_dtype,
        scale,
        splits,
        out,
        scratch,
        *,
        key=None,
        value=None,
    ):
        if splits == "auto":
            splits = self.suggest_splits(
                query, cache, table, cache_dtype, prefill=key is not None
            )
        engine = (
            self._geometry_backend
            if self._geometry_backend is not None
            and self._use_geometry(query, cache, table, key is not None)
            else self
        )
        attention = super()._attention if engine is self else engine._attention
        return attention(
            query,
            cache,
            table,
            mapping,
            lengths,
            signs,
            cache_dtype,
            scale,
            splits,
            out,
            scratch,
            key=key,
            value=value,
        )
