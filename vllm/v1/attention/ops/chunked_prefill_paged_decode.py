# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import functools
import math

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .prefix_prefill import context_attention_fwd

logger = init_logger(__name__)

float8_info = torch.finfo(current_platform.fp8_dtype())


def has_native_kv_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Return whether KV cache blocks can use the native ROCm pairing.

    The native reshape_and_cache writer assumes packed blocks. If cache update
    needs reshape_and_cache_flash for a stride-padded hybrid layout, decode
    should use the matching Triton path too.
    """
    return (
        key_cache.stride(0) == key_cache.shape[1:].numel()
        and value_cache.stride(0) == value_cache.shape[1:].numel()
    )


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale_inv,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    num_queries_per_kv_padded: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    PHYSICAL_BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.int64,  # int
    stride_k_cache_4: tl.int64,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.int64,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
    USE_SINKS: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )

    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (num_queries_per_kv, HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
        L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_head_idx,
            mask=head_mask,
            other=float("-inf"),
        ).to(dtype=tl.float32)
        L = tl.where(float("-inf") < M, 1.0, 0.0)

    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_head_idx, mask=head_mask, other=0.0
        )

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    # iterate through tiles
    for j in range(0, num_blocks):
        start_n = j * BLOCK_SIZE
        # Calculate the logical location within a non-standard physical block,
        # such as 544 in Qwen/Qwen3-Next-80B-A3B-Thinking.
        # Supports non-contiguous mapping
        # from logical blocks to physical blocks
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        # Vectorized loading of physical block IDs
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        # 5D addressing logic of K
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # 4D addressing logic of V (Slot is innermost)
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        seq_mask = seq_offset[None, :] < boundary

        # First calculate the dot, then apply the mask.
        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        context_len = seq_len - 1

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len - seq_offset) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        # m_j : (num_queries_per_kv,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # P : (num_queries_per_kv, BLOCK_SIZE,)
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)

        # l_j : (num_queries_per_kv,)
        l_j = tl.sum(p, axis=1)

        # alpha : (num_queries_per_kv, )
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (num_queries_per_kv, BLOCK_SIZE,)
        acc += tl.dot(p.to(V.dtype), V)

    # epilogue
    acc = acc / (L[:, None] + 1e-10)
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset[:, None] + tl.arange(0, HEAD_SIZE_PADDED)[None, :],
        acc,
        mask=dim_mask[None, :] & head_mask[:, None],
    )


@triton.jit
def kernel_paged_attention_2d_splitkv(
    tmp_output_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    tmp_stride_0: tl.int64,
    tmp_stride_1: tl.int64,
    tmp_stride_2: tl.int64,
    exp_stride_0: tl.int64,
    exp_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    query_start_len_ptr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if cur_batch_query_len > 1:
        return

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    partition_start = partition_idx * PARTITION_SIZE
    partition_end = tl.minimum(partition_start + PARTITION_SIZE, seq_len)
    tokens_in_partition = tl.maximum(partition_end - partition_start, 0)
    num_blocks = cdiv_fn(tokens_in_partition, BLOCK_SIZE)
    block_table_offset = seq_idx * block_table_stride

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for j in range(0, num_blocks):
        abs_token_idx = partition_start + j * BLOCK_SIZE + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(
            block_tables_ptr + block_table_offset + l_block_idx,
            mask=abs_token_idx < partition_end,
            other=0,
        )
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE

        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )

        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )
        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_mask = abs_token_idx[None, :] < partition_end
        qk = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & seq_mask, qk, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    tmp = acc / (L[:, None] + 1e-10)
    tmp_offset = (
        seq_idx * tmp_stride_0
        + query_head_idx[:, None] * tmp_stride_1
        + partition_idx * tmp_stride_2
        + offs_d[None, :]
    )
    stat_offset = (
        seq_idx * exp_stride_0 + query_head_idx * exp_stride_1 + partition_idx
    )
    tl.store(
        tmp_output_ptr + tmp_offset,
        tmp,
        mask=dim_mask[None, :] & head_mask[:, None],
    )
    tl.store(exp_sums_ptr + stat_offset, L, mask=head_mask)
    tl.store(max_logits_ptr + stat_offset, M, mask=head_mask)


@triton.jit
def kernel_paged_attention_2d_splitkv_reduce(
    output_ptr,
    tmp_output_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    seq_lens_ptr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    tmp_stride_0: tl.int64,
    tmp_stride_1: tl.int64,
    tmp_stride_2: tl.int64,
    exp_stride_0: tl.int64,
    exp_stride_1: tl.int64,
    query_start_len_ptr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if cur_batch_query_len > 1:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    valid_partitions = cdiv_fn(seq_len, PARTITION_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    stat_base = seq_idx * exp_stride_0 + query_head_idx * exp_stride_1

    global_m = tl.full([], float("-inf"), dtype=tl.float32)
    for partition_idx in range(0, NUM_PARTITIONS):
        part_m = tl.load(
            max_logits_ptr + stat_base + partition_idx,
            mask=partition_idx < valid_partitions,
            other=float("-inf"),
        )
        global_m = tl.maximum(global_m, part_m)

    global_l = tl.full([], 0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)
    for partition_idx in range(0, NUM_PARTITIONS):
        part_l = tl.load(
            exp_sums_ptr + stat_base + partition_idx,
            mask=partition_idx < valid_partitions,
            other=0.0,
        )
        part_m = tl.load(
            max_logits_ptr + stat_base + partition_idx,
            mask=partition_idx < valid_partitions,
            other=float("-inf"),
        )
        weight = part_l * tl.exp(part_m - global_m)
        tmp_offset = (
            seq_idx * tmp_stride_0
            + query_head_idx * tmp_stride_1
            + partition_idx * tmp_stride_2
            + offs_d
        )
        tmp = tl.load(
            tmp_output_ptr + tmp_offset,
            mask=dim_mask & (partition_idx < valid_partitions),
            other=0.0,
        ).to(tl.float32)
        acc += weight * tmp
        global_l += weight

    acc = acc / (global_l + 1e-10)
    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )
    tl.store(output_ptr + output_offset + offs_d, acc, mask=dim_mask)


def _should_use_splitkv_paged_attention(
    head_size: int,
    block_size: int,
    max_seq_len: int,
    num_seqs: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None,
    sinks: torch.Tensor | None,
    output_scale: torch.Tensor | None,
    sliding_window: int,
) -> bool:
    return (
        current_platform.is_rocm()
        and current_platform.is_navi()
        and head_size == 256
        and block_size > 0
        and max_seq_len >= 4096
        and num_seqs <= 8
        and kv_cache_dtype == "auto"
        and alibi_slopes is None
        and sinks is None
        and output_scale is None
        and sliding_window == 0
    )


def _is_split_eligible(num_n_blocks: int, num_splits: int) -> bool:
    if num_splits == 1:
        return True
    return triton.cdiv(num_n_blocks, num_splits) != triton.cdiv(
        num_n_blocks, num_splits - 1
    )


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Choose SplitKV count using FlashAttention's occupancy heuristic."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1

    max_efficiency = 0.0
    efficiency = []
    for num_splits in range(1, max_splits + 1):
        if not _is_split_eligible(num_n_blocks, num_splits):
            efficiency.append(0.0)
            continue
        n_waves = batch_nheads_mblocks * num_splits / num_sms
        eff = n_waves / math.ceil(n_waves)
        max_efficiency = max(max_efficiency, eff)
        efficiency.append(eff)

    for num_splits in range(1, max_splits + 1):
        if not _is_split_eligible(num_n_blocks, num_splits):
            continue
        if efficiency[num_splits - 1] >= 0.85 * max_efficiency:
            return num_splits
    return 1


def _get_splitkv_num_partitions(
    max_seq_len: int,
    num_seqs: int,
    num_kv_heads: int,
    head_size: int,
) -> int:
    block_n = 256 if head_size <= 64 else 128 if head_size <= 128 else 64
    num_n_blocks = triton.cdiv(max_seq_len, block_n)
    # This fallback launches one stage-1 program per (seq, kv_head, split).
    batch_nheads_mblocks = num_seqs * num_kv_heads
    # FlashAttention uses 2x SMs for 128-thread SplitKV kernels. Use the same
    # occupancy target; on ROCm this maps to compute units.
    num_sms = _get_splitkv_num_sms()
    return _num_splits_heuristic(
        batch_nheads_mblocks=batch_nheads_mblocks,
        num_sms=num_sms,
        num_n_blocks=num_n_blocks,
        max_splits=128,
    )


@functools.cache
def _get_splitkv_num_sms() -> int:
    return current_platform.num_compute_units() * 2


def chunked_prefill_paged_decode(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    # Optional tensor for sinks
    sinks=None,
    is_block_table_ptr: bool = False,
    causal: bool = True,
    max_num_splits: int = 0,
):
    if sm_scale is None:
        sm_scale = 1.0 / (query.shape[2] ** 0.5)

    use_alibi_slopes = alibi_slopes is not None

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if max_query_len > 1:
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            kv_cache_dtype=kv_cache_dtype,
            k_cache=key_cache,
            v_cache=value_cache,
            b_loc=block_table,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_seq_len=max_seq_len,
            max_input_len=max_query_len,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=True,
            fp8_out_scale=output_scale,
            sinks=sinks,
            causal=causal,
        )

    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    # key may be None in cross-attention decode (already cached from encoder)
    num_kv_heads = key.shape[1] if key is not None else key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert key_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert value_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported FP8 kv_cache_dtype {kv_cache_dtype}: "
                f"should be one of 'fp8', 'fp8_e4m3', 'fp8_e5m2'."
            )

        key_cache = key_cache.view(target_dtype)
        value_cache = value_cache.view(target_dtype)

    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)

    from vllm.platforms.rocm import use_rocm_custom_paged_attention

    use_custom = use_rocm_custom_paged_attention(
        query.dtype,
        head_size,
        block_size,
        num_queries_per_kv,
        max_seq_len,
        sliding_window,
        kv_cache_dtype,
        alibi_slopes,
        sinks,
    )
    has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)
    # Force Triton for non-standard blocks like Qwen3's 544 and for
    # stride-padded hybrid layouts. The latter use reshape_and_cache_flash
    # during cache update, so keep decode on the matching stride-aware path.
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)
    if not is_pow2 or not has_native_layout:
        use_custom = False

    if use_custom:
        _PARTITION_SIZE_ROCM = 256
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
        assert _PARTITION_SIZE_ROCM % block_size == 0
        total_num_seq = block_table.shape[0]
        tmp_output = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions, head_size),
            dtype=query.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(total_num_seq, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_rocm(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale=sm_scale,
            block_tables=block_table,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_size=block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_out_scale=output_scale,
        )
    else:
        logger.warning_once(
            "Cannot use ROCm custom paged attention kernel,"
            " falling back to Triton implementation."
        )
        real_block_size = value_cache.shape[3]
        # The standard model directly uses the original block_size.
        # Non-standard 544 uses 32 to accommodate integer division logic.
        # Cap at 128 to avoid exceeding GPU shared memory limits
        # (e.g. hybrid Mamba models inflate block_size to 2048).
        # The kernel handles TRITON_BLOCK_SIZE != PHYSICAL_BLOCK_SIZE
        # via the l_block_idx/internal_offsets addressing logic.
        MAX_TRITON_BLOCK_SIZE = 128
        TRITON_BLOCK_SIZE = min(block_size, MAX_TRITON_BLOCK_SIZE) if is_pow2 else 32
        if is_block_table_ptr:
            # Using the physical base address of tensors
            kv_element_size = key_cache.element_size()
            block_byte_stride = key_cache.stride(0) * kv_element_size
            # Get the starting physical address of the KV Cache
            base_addr = key_cache.data_ptr()

            # Normalization: Directly calculate the block offset
            # of the pointer relative to the base address
            processed_block_table = ((block_table - base_addr) // block_byte_stride).to(
                torch.int32
            )
        else:
            processed_block_table = block_table.to(torch.int32)

        use_splitkv = _should_use_splitkv_paged_attention(
            head_size=head_size,
            block_size=real_block_size,
            max_seq_len=max_seq_len,
            num_seqs=num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            output_scale=output_scale,
            sliding_window=sliding_window,
        )
        num_partitions = 1
        if use_splitkv:
            if max_num_splits == 0:
                num_partitions = _get_splitkv_num_partitions(
                    max_seq_len=max_seq_len,
                    num_seqs=num_seqs,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                )
            else:
                num_partitions = min(max_num_splits, 128)
        use_splitkv = use_splitkv and num_partitions > 1
        if use_splitkv:
            PARTITION_SIZE = triton.cdiv(max_seq_len, num_partitions)
            total_num_seq = processed_block_table.shape[0]
            tmp_output = torch.empty(
                size=(total_num_seq, num_query_heads, num_partitions, head_size),
                dtype=query.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(total_num_seq, num_query_heads, num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)

            kernel_paged_attention_2d_splitkv[
                (
                    num_seqs,
                    num_kv_heads,
                    num_partitions,
                )
            ](
                tmp_output_ptr=tmp_output,
                exp_sums_ptr=exp_sums,
                max_logits_ptr=max_logits,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                tmp_stride_0=tmp_output.stride(0),
                tmp_stride_1=tmp_output.stride(1),
                tmp_stride_2=tmp_output.stride(2),
                exp_stride_0=exp_sums.stride(0),
                exp_stride_1=exp_sums.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                PARTITION_SIZE=PARTITION_SIZE,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                query_start_len_ptr=query_start_loc,
            )
            kernel_paged_attention_2d_splitkv_reduce[
                (
                    num_seqs,
                    num_query_heads,
                )
            ](
                output_ptr=output,
                tmp_output_ptr=tmp_output,
                exp_sums_ptr=exp_sums,
                max_logits_ptr=max_logits,
                seq_lens_ptr=seq_lens,
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                tmp_stride_0=tmp_output.stride(0),
                tmp_stride_1=tmp_output.stride(1),
                tmp_stride_2=tmp_output.stride(2),
                exp_stride_0=exp_sums.stride(0),
                exp_stride_1=exp_sums.stride(1),
                query_start_len_ptr=query_start_loc,
                PARTITION_SIZE=PARTITION_SIZE,
                NUM_PARTITIONS=num_partitions,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            )
        else:
            kernel_paged_attention_2d[
                (
                    num_seqs,
                    num_kv_heads,
                )
            ](
                output_ptr=output,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                sink_ptr=sinks,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                SLIDING_WINDOW=sliding_window,
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_SINKS=sinks is not None,
                USE_FP8=output_scale is not None,
            )
