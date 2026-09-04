# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public router for the RDNA4 SplitKV paged-attention kernel family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .rdna4_splitkv_bf16 import compile_native_bf16_d256_stage
from .rdna4_splitkv_common import HEAD_DIM
from .rdna4_splitkv_direct import compile_direct_stage
from .rdna4_splitkv_general import compile_native_tail_stage
from .rdna4_splitkv_large_gqa import compile_large_gqa_stage
from .rdna4_splitkv_reduce import (
    _launch_reduce,
    run_grouped_stage_local_reduce,
)
from .runtime import run_compiled as _run_compiled


class SplitKVRoute(str, Enum):
    """Validated RDNA4 kernel specializations."""

    FP8_D128_GQA1_DIRECT = "fp8_d128_gqa1_direct"
    FP8_D256_GQA6_7_GROUPED = "fp8_d256_gqa6_7_grouped"
    FP8_D256_GQA9_16_GROUPED = "fp8_d256_gqa9_16_grouped"
    FP8_GENERAL_SPLIT = "fp8_general_split"
    FP16_D256_GQA4_SPLIT = "fp16_d256_gqa4_split"
    BF16_D256_SPLIT = "bf16_d256_split"


@dataclass(frozen=True)
class SplitKVConfig:
    """A promoted kernel route selected from the runtime tensor contract."""

    route: SplitKVRoute


def select_kernel_config(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> SplitKVConfig | None:
    """Select a specialization validated by the RDNA4 tuning sweep."""
    if query.ndim != 3 or key_cache.ndim != 5:
        return None
    num_kv_heads = int(key_cache.shape[1])
    if num_kv_heads <= 0 or int(query.shape[1]) % num_kv_heads:
        return None

    gqa = int(query.shape[1]) // num_kv_heads
    head_size = int(query.shape[2])
    page_size = int(key_cache.shape[3])
    batch_size = int(seq_lens.numel())
    page_aligned = page_size >= 8 and page_size % 8 == 0
    max_active_pages = (int(max_seq_len) + page_size - 1) // page_size

    if (
        query.dtype in (torch.bfloat16, torch.float16)
        and key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and head_size == 128
        and batch_size == 1
        and gqa == 1
        and page_aligned
        and num_kv_heads < 32
        and 512 < max_active_pages * page_size <= 1568
    ):
        return SplitKVConfig(SplitKVRoute.FP8_D128_GQA1_DIRECT)

    if (
        query.dtype == torch.bfloat16
        and key_cache.dtype == torch.float8_e4m3fn
        and head_size == 256
        and batch_size == 1
        and gqa in (6, 7)
        and page_aligned
    ):
        return SplitKVConfig(SplitKVRoute.FP8_D256_GQA6_7_GROUPED)

    if (
        query.dtype == torch.bfloat16
        and key_cache.dtype == torch.float8_e4m3fn
        and head_size == 256
        and 9 <= gqa <= 16
        and page_aligned
        and max_active_pages > 2 * batch_size
    ):
        return SplitKVConfig(SplitKVRoute.FP8_D256_GQA9_16_GROUPED)

    general_fp8 = (
        query.dtype == torch.float16
        and key_cache.dtype == torch.float8_e4m3fn
        and (head_size == 128 or (head_size == 256 and batch_size > 1))
        and 9 <= gqa <= 16
    ) or (
        query.dtype == torch.bfloat16
        and key_cache.dtype == torch.float8_e4m3fn
        and head_size == 256
        and batch_size > 1
        and 1 <= gqa <= 5
        and max_active_pages >= 8 * batch_size
    )
    if general_fp8 and page_aligned:
        return SplitKVConfig(SplitKVRoute.FP8_GENERAL_SPLIT)

    if (
        query.dtype == torch.float16
        and key_cache.dtype == torch.float16
        and head_size == 256
        and batch_size > 1
        and gqa == 4
        and page_aligned
    ):
        return SplitKVConfig(SplitKVRoute.FP16_D256_GQA4_SPLIT)

    if (
        query.dtype == torch.bfloat16
        and key_cache.dtype == torch.bfloat16
        and head_size == 256
        and ((gqa == 16 and batch_size == 1) or (gqa == 8 and batch_size > 1))
        and page_aligned
        and max_active_pages >= 16 * batch_size
    ):
        return SplitKVConfig(SplitKVRoute.BF16_D256_SPLIT)

    return None


def _run_direct_finalize(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_lse: torch.Tensor,
    scale: float,
) -> None:
    """Run one complete online-softmax scan and store caller output."""

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    batch = int(seq_lens.numel())
    num_query_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = num_query_heads // num_kv_heads
    grouped_stage = compile_direct_stage(
        query_dtype="fp16" if query.dtype == torch.float16 else "bf16",
        kv_dtype=(
            "fp8fnuz"
            if key_cache.dtype == torch.float8_e4m3fnuz
            else "fp8"
            if key_cache.element_size() == 1
            else "bf16"
        ),
        splits=1,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        head_dim=int(query.shape[2]),
        page_size=int(key_cache.shape[3]),
        softmax_scale=float(scale),
    )
    _run_compiled(
        grouped_stage,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
        mid_lse,
        batch,
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        int(output.stride(0)),
        int(output.stride(1)),
        int(output.shape[-1]),
        *map(int, mid_lse.stride()),
        torch.cuda.current_stream(query.device),
    )


def _run_large_gqa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Launch the local large-GQA stage and wave32 split reducer."""

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    batch = int(seq_lens.numel())
    num_query_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = num_query_heads // num_kv_heads
    page_size = int(key_cache.shape[3])
    stage1 = compile_large_gqa_stage(
        kv_dtype="fp8",
        splits=int(splits),
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=page_size,
        softmax_scale=float(scale),
    )
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage1,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        batch,
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        splits,
        64,
        stream,
    )
    return output


def _run_general_fp8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Launch the native D128 FP16 stage and wave32 split reducer."""

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    batch = int(seq_lens.numel())
    num_query_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = num_query_heads // num_kv_heads
    head_dim = int(query.shape[2])
    page_size = int(key_cache.shape[3])
    stage1 = compile_native_tail_stage(
        query_dtype="fp16" if query.dtype == torch.float16 else "bf16",
        kv_dtype=("fp8fnuz" if key_cache.dtype == torch.float8_e4m3fnuz else "fp8"),
        head_dim=head_dim,
        splits=int(splits),
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=page_size,
        softmax_scale=float(scale),
    )
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage1,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        batch,
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        splits,
        64,
        stream,
    )
    return output


def compile_ragged_fp16_stage(
    *,
    splits: int,
    num_kv_heads: int,
    query_group_size: int,
    page_size: int,
    softmax_scale: float,
):
    """Compile the D256 low-GQA FP16-query/FP16-cache semantic family."""

    if not 1 <= query_group_size <= 4:
        raise ValueError(f"low-GQA stage requires GQA 1..4, got {query_group_size}")
    return compile_native_tail_stage(
        query_dtype="fp16",
        kv_dtype="fp16",
        head_dim=256,
        splits=splits,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=page_size,
        softmax_scale=softmax_scale,
    )


def _run_fp16_d256_gqa4(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Launch the sequence-major FP16-cache grouped stage and local reducer."""

    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)
    batch = int(seq_lens.numel())
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = int(query.shape[1]) // num_kv_heads
    stage_splits = int(splits)
    stage1 = compile_ragged_fp16_stage(
        splits=stage_splits,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=int(key_cache.shape[3]),
        softmax_scale=float(scale),
    )
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage1,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        batch,
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        stage_splits,
        64,
        stream,
    )
    return output


def _run_bf16_d256(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
) -> torch.Tensor:
    """Run the isolated BF16 D256 GQA8/GQA16 specialization."""

    if query.dtype != torch.bfloat16:
        raise ValueError(f"query must be bfloat16, got {query.dtype}")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise ValueError("key_cache and value_cache must both be bfloat16")
    if int(query.shape[2]) != HEAD_DIM:
        raise ValueError(f"head dimension must be {HEAD_DIM}, got {query.shape[2]}")
    num_kv_heads = int(key_cache.shape[1])
    query_group_size = int(query.shape[1]) // num_kv_heads
    if query_group_size not in (8, 16):
        raise ValueError(f"query group size must be 8 or 16, got {query_group_size}")
    page_size = int(key_cache.shape[3])
    if k_scale.ndim == 0:
        k_scale = k_scale.reshape(1)
    if v_scale.ndim == 0:
        v_scale = v_scale.reshape(1)

    stage1 = compile_native_bf16_d256_stage(
        splits=int(splits),
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        page_size=page_size,
        softmax_scale=float(scale),
    )
    stream = torch.cuda.current_stream(query.device)
    _run_compiled(
        stage1,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        mid_out,
        mid_lse,
        int(seq_lens.numel()),
        int(block_tables.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        *map(int, key_cache.stride()),
        *map(int, value_cache.stride()),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
    _launch_reduce(
        query,
        seq_lens,
        query_start_loc,
        output,
        mid_out,
        mid_lse,
        int(splits),
        64,
        stream,
    )
    return output


def rdna4_splitkv_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    scale: float,
    max_seq_len: int,
    config: SplitKVConfig | None = None,
) -> SplitKVConfig:
    """Validate, select, and launch an RDNA4 SplitKV specialization."""
    if config is None:
        config = select_kernel_config(query, key_cache, seq_lens, max_seq_len)
    if config is None:
        raise ValueError("RDNA4 FlyDSL SplitKV does not support this configuration")

    args = (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
    )
    if config.route == SplitKVRoute.FP8_D128_GQA1_DIRECT:
        _run_direct_finalize(*args, mid_lse, scale)
    elif config.route == SplitKVRoute.FP8_D256_GQA6_7_GROUPED:
        run_grouped_stage_local_reduce(*args, mid_out, mid_lse, splits, scale)
    elif config.route == SplitKVRoute.FP8_D256_GQA9_16_GROUPED:
        _run_large_gqa(*args, mid_out, mid_lse, splits, scale)
    elif config.route == SplitKVRoute.FP8_GENERAL_SPLIT:
        _run_general_fp8(*args, mid_out, mid_lse, splits, scale)
    elif config.route == SplitKVRoute.FP16_D256_GQA4_SPLIT:
        _run_fp16_d256_gqa4(*args, mid_out, mid_lse, splits, scale)
    else:
        _run_bf16_d256(*args, mid_out, mid_lse, splits, scale)
    return config


__all__ = [
    "SplitKVConfig",
    "SplitKVRoute",
    "rdna4_splitkv_paged_attention",
    "select_kernel_config",
]
