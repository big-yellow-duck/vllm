# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager PyTorch Octave KV reference for the RDNA4 kernel bring-up."""

import importlib.util
import os
import sys
from pathlib import Path

import torch


def _load_octave_format():
    # Load the format's source directly so importing it needs no vLLM extension.
    relative_source = Path("vllm/v1/attention/ops/rocm_octave.py")
    default_root = next(
        (
            root
            for root in Path(__file__).resolve().parents
            if (root / relative_source).is_file()
        ),
        Path(__file__).resolve().parent,
    )
    root = Path(os.environ.get("OCTAVE_VLLM_ROOT", default_root))
    source = root / relative_source
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Set OCTAVE_VLLM_ROOT to a vLLM checkout."
        )
    spec = importlib.util.spec_from_file_location("_vllm_octave_format", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


format_ops = _load_octave_format()
OctaveFormat = format_ops.OctaveFormat
reference_quantize = format_ops.reference_quantize
reference_dequantize = format_ops.reference_dequantize
reference_decode = format_ops.reference_attention


def reference_cache_store(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    fmt: OctaveFormat,
) -> None:
    """Write K/V rows into cache shaped (blocks, KV heads, page size, bytes).

    Each slot number is a flattened physical page and token offset. Negative
    slots are padding. Live slot numbers must be unique within an update.
    """
    live = slot_mapping >= 0
    slots = slot_mapping[live]
    if slots.numel() == 0:
        return
    n = slot_mapping.numel()
    packed = reference_quantize(key[:n][live], value[:n][live], fmt)
    page_size = cache.shape[2]
    cache[slots // page_size, :, slots % page_size] = packed


def reference_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    fmt: OctaveFormat,
    scale: float,
) -> torch.Tensor:
    """FP32 causal attention over a packed prefix and a full-precision chunk.

    Q, K and V are the unrotated chunk tensors (tokens, heads, head size), with
    RoPE already applied to Q and K. Sequence lengths include the chunk.
    """
    page_size = cache.shape[2]
    group = query.shape[1] // cache.shape[1]
    out = torch.empty_like(query, dtype=torch.float32)
    starts = query_start_loc.tolist()
    for req, seq_len in enumerate(seq_lens.tolist()):
        start, stop = starts[req : req + 2]
        chunk_len = stop - start
        if chunk_len == 0:
            continue
        prefix_len = seq_len - chunk_len
        k, v = key[start:stop].float(), value[start:stop].float()
        if prefix_len:
            pos = torch.arange(prefix_len, device=cache.device)
            blocks = block_table[req, pos // page_size].long()
            k_prefix, v_prefix = reference_dequantize(
                cache[blocks, :, pos % page_size], fmt
            )
            k = torch.cat((k_prefix, k))
            v = torch.cat((v_prefix, v))
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,thd->hqt", query[start:stop].float(), k) * scale
        positions = torch.arange(seq_len, device=query.device)
        causal_mask = positions[None, :] > positions[prefix_len:, None]
        weights = scores.masked_fill(causal_mask, -torch.inf).softmax(dim=-1)
        out[start:stop] = torch.einsum("hqt,thd->qhd", weights, v)
    return out


def reference_quantize_geometry(key, value, layout):
    """Pack a logical head via its explicitly described physical wire layout."""
    return reference_quantize(
        layout.to_storage(key), layout.to_storage(value), layout.reference_format()
    )


def reference_dequantize_geometry(slots, layout):
    key, value = reference_dequantize(slots, layout.reference_format())
    return layout.from_storage(key), layout.from_storage(value)


def reference_decode_geometry(
    query, cache, table, mapping, lengths, layout, scale=None
):
    scale = layout.head_dim**-0.5 if scale is None else scale
    result = reference_decode(
        layout.to_storage(query),
        cache,
        table,
        mapping,
        lengths,
        layout.reference_format(),
        scale,
    )
    return layout.from_storage(result)


def reference_prefill_geometry(
    query, key, value, cache, table, starts, lengths, layout, scale=None
):
    scale = layout.head_dim**-0.5 if scale is None else scale
    result = reference_prefill(
        layout.to_storage(query),
        layout.to_storage(key),
        layout.to_storage(value),
        cache,
        table,
        starts,
        lengths,
        layout.reference_format(),
        scale,
    )
    return layout.from_storage(result)
