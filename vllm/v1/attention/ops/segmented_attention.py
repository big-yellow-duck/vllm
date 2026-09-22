# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention dispatch for the opt-in ROCm token-major KV backend."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import get_kv_quant_mode

from .segmented_prefill import (
    MAX_QUERY_LEN,
    can_use_segmented_prefill,
    segmented_prefill_attention,
    select_segmented_config,
)
from .segmented_prefill_tuning import get_segmented_config
from .triton_unified_attention import unified_attention

logger = init_logger(__name__)


def segmented_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    kv_cache_dtype: str,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_query_len: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    *,
    sliding_window: int = 0,
    output_scale: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    causal: bool = True,
    softcap: float = 0.0,
) -> None:
    """Run segmented attention when eligible, otherwise use unified Triton."""
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        fp8_dtype = current_platform.fp8_dtype()
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

    from vllm.platforms.rocm import on_gfx1x, on_gfx12x

    segmented_pattern = (
        (on_gfx12x() if key_cache.element_size() == 1 else on_gfx1x())
        and causal
        and sliding_window <= 0
        and not softcap
        and sinks is None
        and output_scale is None
    )
    if (
        segmented_pattern
        and 0 < max_query_len <= MAX_QUERY_LEN
        and (
            can_use_segmented_prefill(
                query,
                key,
                value,
                output,
                key_cache,
                value_cache,
                block_table,
                query_start_loc,
                seq_lens,
                k_scale,
                v_scale,
            )
        )
    ):
        config = get_segmented_config(
            query.device,
            query.dtype,
            key_cache.dtype,
            query.shape[1],
            key_cache.shape[2],
            query.shape[2],
            key_cache.shape[1],
            sm_scale,
            len(seq_lens),
            max_query_len,
            max_seq_len,
        )
        if config is None:
            config = select_segmented_config(
                len(seq_lens),
                max_query_len,
                max_seq_len,
                query.shape[1],
                key_cache.shape[2],
                query.shape[2],
                key_cache.element_size() == 1,
            )
        segmented_prefill_attention(
            query,
            output,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            k_scale,
            v_scale,
            sm_scale,
            skip_decode=False,
            config=config,
        )
        return

    if sliding_window > 0:
        logger.info_once(
            "ROCM_SEGMENTED_ATTN is routing sliding-window attention to the "
            "unified Triton attention fallback."
        )
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=sm_scale,
        causal=causal,
        window_size=(-1, -1) if sliding_window <= 0 else (sliding_window, 0),
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        output_scale=output_scale,
        sinks=sinks,
        kv_quant_mode=get_kv_quant_mode(kv_cache_dtype),
    )
