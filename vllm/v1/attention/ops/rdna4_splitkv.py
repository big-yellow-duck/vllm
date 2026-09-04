# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RDNA4 backend selection for SplitKV paged attention."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _load_flydsl_splitkv() -> tuple[Any, Any]:
    """Load the in-tree kernels without making FlyDSL a vLLM dependency."""
    import flydsl.compiler  # noqa: F401
    import flydsl.expr  # noqa: F401

    from .flydsl_kernels.rdna4_splitkv import (
        rdna4_splitkv_paged_attention,
        select_kernel_config,
    )

    return select_kernel_config, rdna4_splitkv_paged_attention


@lru_cache(maxsize=1)
def is_rdna4_flydsl_splitkv_available() -> bool:
    """Return whether the FlyDSL compiler and in-tree kernels import."""
    try:
        _load_flydsl_splitkv()
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(
            "RDNA4 FlyDSL SplitKV is unavailable (%s); using another backend.",
            exc,
        )
        return False
    return True


def can_use_rdna4_hip_splitkv_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    scale: float,
    actual_max_splits: int,
    filter_by_query_len: bool,
) -> bool:
    """Return whether the tuned native HIP specialization applies."""
    from vllm.platforms.rocm import on_rdna4

    fp8_kv = key_cache.dtype == torch.float8_e4m3fn
    cache_groups = 16 if fp8_kv else 32
    cache_pack = 16 if fp8_kv else 8
    return (
        on_rdna4()
        and query.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and key_cache.dtype == value_cache.dtype
        and key_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and query.ndim == 3
        and query.shape[1:] == (12, 256)
        and output.shape == query.shape
        and key_cache.ndim == 5
        and key_cache.shape[1:] == (2, cache_groups, 1568, cache_pack)
        and value_cache.ndim == 4
        and value_cache.shape == (key_cache.shape[0], 2, 256, 1568)
        and block_tables.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and query_start_loc is not None
        and query_start_loc.dtype == torch.int32
        and filter_by_query_len
        and isinstance(k_scale, torch.Tensor)
        and isinstance(v_scale, torch.Tensor)
        and k_scale.dtype == torch.float32
        and v_scale.dtype == torch.float32
        and k_scale.numel() == 1
        and v_scale.numel() == 1
        and math.isclose(scale, 0.0625, rel_tol=0.0, abs_tol=1.0e-8)
        and actual_max_splits in (1, 2, 4, 8, 16)
    )


def get_rdna4_flydsl_splitkv_config(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    scale: float,
    actual_max_splits: int,
    max_seq_len: int,
    filter_by_query_len: bool,
) -> Any | None:
    """Return a validated FlyDSL configuration, or ``None`` for fallback."""
    from vllm.platforms.rocm import on_rdna4

    if (
        not on_rdna4()
        or not is_rdna4_flydsl_splitkv_available()
        or query.ndim != 3
        or key_cache.ndim != 5
        or value_cache.ndim != 4
    ):
        return None
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    page_size = key_cache.shape[3]
    cache_pack = 16 // key_cache.element_size()
    cache_groups = head_size // cache_pack
    supported_dtypes = (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    valid = (
        query.dtype in (torch.bfloat16, torch.float16)
        and output.dtype == query.dtype
        and key_cache.dtype == value_cache.dtype
        and key_cache.dtype in supported_dtypes
        and head_size in (128, 256)
        and output.shape == query.shape
        and num_kv_heads > 0
        and num_query_heads % num_kv_heads == 0
        and page_size >= 8
        and page_size % 8 == 0
        and key_cache.shape[2:] == (cache_groups, page_size, cache_pack)
        and value_cache.shape
        == (key_cache.shape[0], num_kv_heads, head_size, page_size)
        and key_cache.stride(4) == 1
        and value_cache.stride(3) == 1
        and block_tables.dtype == torch.int32
        and seq_lens.dtype == torch.int32
        and query_start_loc is not None
        and query_start_loc.dtype == torch.int32
        and filter_by_query_len
        and isinstance(k_scale, torch.Tensor)
        and isinstance(v_scale, torch.Tensor)
        and k_scale.dtype == torch.float32
        and v_scale.dtype == torch.float32
        and k_scale.numel() == 1
        and v_scale.numel() == 1
        and math.isfinite(scale)
        and actual_max_splits in (2, 4, 8, 16)
    )
    if not valid:
        return None
    select_kernel_config, _ = _load_flydsl_splitkv()
    return select_kernel_config(query, key_cache, seq_lens, max_seq_len)


def can_use_rdna4_flydsl_splitkv_paged_attention(**kwargs) -> bool:
    """Return whether a validated RDNA4 FlyDSL route applies."""
    return get_rdna4_flydsl_splitkv_config(**kwargs) is not None


def try_rdna4_splitkv_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    scale: float,
    actual_max_splits: int,
    max_seq_len: int,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    filter_by_query_len: bool,
) -> bool:
    """Run the selected RDNA4 backend and report whether one was used."""
    common_args = dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        k_scale=k_scale,
        v_scale=v_scale,
        scale=scale,
        actual_max_splits=actual_max_splits,
        filter_by_query_len=filter_by_query_len,
    )
    flydsl_config = None
    if envs.VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL:
        flydsl_config = get_rdna4_flydsl_splitkv_config(
            **common_args, max_seq_len=max_seq_len
        )
    use_hip = flydsl_config is None and can_use_rdna4_hip_splitkv_paged_attention(
        **common_args
    )
    if flydsl_config is None and not use_hip:
        return False

    assert query_start_loc is not None
    assert isinstance(k_scale, torch.Tensor)
    assert isinstance(v_scale, torch.Tensor)
    if flydsl_config is not None:
        _, launch = _load_flydsl_splitkv()
        launch(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            query_start_loc,
            k_scale,
            v_scale,
            output,
            mid_out,
            mid_lse,
            actual_max_splits,
            scale,
            max_seq_len,
            config=flydsl_config,
        )
        logger.info_once(
            "Using RDNA4 FlyDSL SplitKV route: %s", flydsl_config.route.value
        )
        return True

    token_halves = (
        seq_lens.numel() == 1
        and key_cache.dtype == torch.float8_e4m3fn
        and block_tables.shape[1] >= 6
        and actual_max_splits == 16
    )
    ops.rdna4_splitkv_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_loc,
        k_scale,
        v_scale,
        output,
        mid_out,
        mid_lse,
        actual_max_splits,
        token_halves,
    )
    return True


__all__ = [
    "can_use_rdna4_flydsl_splitkv_paged_attention",
    "can_use_rdna4_hip_splitkv_paged_attention",
    "get_rdna4_flydsl_splitkv_config",
    "is_rdna4_flydsl_splitkv_available",
    "try_rdna4_splitkv_paged_attention",
]
