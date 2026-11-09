# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload-aware, graph-stable split selection with a bounded workspace."""

import math

DEFAULT_WORKSPACE_BYTES = 2 << 30


def _ceil2(value):
    return 1 << max(0, math.ceil(math.log2(max(1, value))))


def _nearest2(value):
    return 1 << max(0, math.floor(math.log2(max(1, value)) + 0.5))


def _floor2(value):
    return 1 << max(0, int(value).bit_length() - 1)


def recommend_splits(
    *,
    tokens,
    heads,
    kv_heads,
    requests,
    capacity,
    page_size,
    cache_dtype,
    prefill=False,
    workspace_limit_bytes=DEFAULT_WORKSPACE_BYTES,
    max_context_tokens=None,
    head_dim=256,
    storage_head_dim=None,
    legacy_fast=True,
):
    """Return a power of two in 1..256 without reading device tensor values.

    The workspace budget bounds automatic FP32 split scratch; S=1 needs none.
    A host-known context bound can reduce an oversized allocation hint. Both
    are scheduling hints: device causal metadata remains authoritative.
    """
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    if max_context_tokens is not None:
        if not isinstance(max_context_tokens, int) or max_context_tokens < 0:
            raise ValueError("max_context_tokens must be a nonnegative integer")
        capacity = min(capacity, max_context_tokens)
    if (
        min(tokens, heads, kv_heads, requests, capacity, page_size) <= 0
        or heads % kv_heads
    ):
        return 1
    if not isinstance(head_dim, int) or head_dim < 1:
        raise ValueError("head_dim must be a positive integer")
    storage_head_dim = head_dim if storage_head_dim is None else storage_head_dim
    group = heads // kv_heads
    if not legacy_fast:
        rows = 16
        head_tile = min(group, rows)
        head_tiles = math.ceil(group / head_tile)
        tile_queries = (
            max(1, rows // head_tile)
            if prefill and tokens >= 16 and tokens > requests
            else 1
        )
        scalar = (
            head_dim < 32
            or (not prefill and group < 4)
            or (prefill and (tokens < 16 or tokens <= requests))
        )
        blocks = (
            tokens * heads
            if scalar
            else math.ceil(tokens / tile_queries) * kv_heads * head_tiles
        )
        # A small matrix grid benefits from shorter merge chains; scalar
        # batches need more independent context work to fill the device.
        target = (
            128
            if not prefill and group >= 4 and tokens <= 4 and capacity <= 32768
            else 384
        )
        # A tiny MHA query grid pays heavily for repeated setup/partial writes.
        # Keep enough matrix work per context partition to amortize that cost.
        if prefill and not scalar and group == 1 and tokens < 32:
            target = 128
        occupancy = _ceil2(target / blocks)
        segment = 512 if tokens < 256 else 2048 if tokens <= 1024 else 8192
        if not prefill and group < 4 and 8 <= tokens <= 128 and tokens * heads <= 256:
            segment = 128
        splits = max(occupancy, _nearest2(capacity / segment))
        splits = min(splits, _ceil2(capacity / 64), 8 if tokens >= 4096 else 256)
        maximum = min(
            256, _floor2(workspace_limit_bytes // (tokens * heads * (head_dim + 2) * 4))
        )
        return min(maximum, max(1, splits))
    per_request = max(1, math.ceil(tokens / requests))
    if prefill:
        prefix = max(0, capacity - per_request - (page_size - 1))
        if prefix < page_size:
            splits = 1
        else:
            native_reuse = (
                group == 6
                and cache_dtype == "octave_k3v4"
                and tokens >= 256
                and prefix >= 1000
            )
            tile_queries = (
                (6 if per_request >= 128 else 2)
                if native_reuse
                else (
                    2
                    if group == 6 and tokens >= 2
                    else max(1, 16 // group)
                    if tokens >= 256 and group in (1, 2, 4, 8)
                    else 1
                )
            )
            blocks = max(1, math.ceil(tokens / tile_queries) * kv_heads)
            occupancy = _ceil2(512 / blocks)
            if tokens < 256:
                useful = _nearest2(capacity / 256)
                splits = max(min(occupancy, useful), _nearest2(capacity / 2048))
            else:
                # Shorter partitions also reduce rare FP32 fallback tails;
                # wide query grids need fewer splits and less workspace.
                segment = 2048 if tokens <= 1024 else 8192
                splits = max(occupancy, _nearest2(capacity / segment))
            # Request-boundary FP32 tails need enough context partitions in
            # short batched chunks; wide grids otherwise avoid oversplitting.
            if (
                group == 6
                and 256 <= tokens <= 2048
                and 64 <= per_request < 256
                and requests > 1
            ):
                splits = max(splits, 8 if prefix >= 2048 else 4 if prefix >= 512 else 1)
            if tokens >= 4096:
                splits = min(splits, 8)
    elif tokens > requests and tokens <= 4 and group == 6:
        splits = _ceil2(capacity / 256)
    elif group == 6 and tokens >= 8 and (tokens >= 32 or capacity >= 512):
        # Batched decode shares a tile across one query's GQA heads. Avoid
        # multiplying an already wide grid into tiny context partitions.
        splits = max(_ceil2(128 / (tokens * kv_heads)), _nearest2(capacity / 1024))
    elif tokens >= 32:
        splits = max(1, min(_ceil2(512 / (tokens * heads)), _ceil2(capacity / 64)))
        if group < 4 and tokens <= 128 and capacity >= 1024 and tokens * heads <= 256:
            splits = max(splits, _nearest2(capacity / 256))
    else:
        if capacity <= 256:
            splits = min(16, _ceil2(capacity / 8))
        elif cache_dtype == "octave_k3v3_compact":
            splits = min(128, max(16, _nearest2(capacity / 128)))
        elif cache_dtype == "octave_k3v4":
            frontier = max(64, _nearest2(capacity / 256))
            splits = min(frontier, max(16, _ceil2(capacity / max(64, 8 * tokens))))
        elif capacity >= 16384:
            splits = max(128, _ceil2(capacity / 256))
        elif capacity >= 2048:
            splits = max(16, _nearest2(capacity / 128))
        else:
            splits = max(16, _ceil2(capacity / 64))
    maximum = min(
        256, _floor2(workspace_limit_bytes // (tokens * heads * (head_dim + 2) * 4))
    )
    return min(maximum, max(1, splits))
