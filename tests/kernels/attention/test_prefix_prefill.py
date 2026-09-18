# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import time
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

pytestmark = pytest.mark.skip_global_cleanup

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 64]
HEAD_SIZES = [24, 128]
DTYPES = [torch.float16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]

OPS = [chunked_prefill_paged_decode, context_attention_fwd]


def create_causal_attention_mask_for_sdpa(
    query_lens: list[int],
    seq_lens: list[int],
    sliding_window: int = 0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    total_queries = sum(query_lens)
    total_keys = sum(seq_lens)

    # Create a mask filled with -inf
    mask = torch.full(
        (total_queries, total_keys), float("-inf"), device=device, dtype=dtype
    )

    query_start = 0
    key_start = 0

    for query_len, seq_len in zip(query_lens, seq_lens):
        query_end = query_start + query_len
        key_end = key_start + seq_len
        q_indices = torch.arange(query_len, device=device)
        k_indices = torch.arange(seq_len, device=device)
        q_pos_in_seq = seq_len - query_len + q_indices

        valid_mask = k_indices[None, :] <= q_pos_in_seq[:, None]

        if sliding_window > 0:
            valid_mask &= k_indices[None, :] >= (
                q_pos_in_seq[:, None] - sliding_window + 1
            )

        mask[query_start:query_end, key_start:key_end][valid_mask] = 0.0

        query_start = query_end
        key_start = key_end

    return mask


def create_alibi_causal_mask(
    query_len: int,
    seq_len: int,
    alibi_slopes: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_pos = torch.arange(
        seq_len - query_len, seq_len, device=device, dtype=torch.float32
    )
    key_pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    rel_pos = key_pos[None, :] - query_pos[:, None]

    # Apply ALiBi slopes: [num_heads, query_len, seq_len]
    alibi_bias = alibi_slopes[:, None, None] * rel_pos[None, :, :]
    alibi_bias = alibi_bias.to(dtype)

    # Apply causal mask: prevent attending to future positions
    # causal_mask[i, j] = True if key_pos[j] <= query_pos[i]
    causal_mask = key_pos[None, :] <= query_pos[:, None]
    alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float("-inf"))

    # Add batch dimension: [1, num_heads, query_len, seq_len]
    # SDPA expects batch dimension even for single sequences
    return alibi_bias.unsqueeze(0)


def _make_unified_paged_case(
    query_lens: list[int],
    context_lens: list[int],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    fp8: bool,
    causal: bool = True,
):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(419)
    total_queries = sum(query_lens)
    qkv = (
        torch.randn(
            total_queries,
            num_heads + 2 * num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25
    )
    query, key, value = qkv.split((num_heads, num_kv_heads, num_kv_heads), dim=1)
    cache_dtype = current_platform.fp8_dtype() if fp8 else torch.bfloat16
    k_scale = torch.tensor(0.125 if fp8 else 1.0, device=device)
    v_scale = torch.tensor(0.25 if fp8 else 1.0, device=device)
    max_seq_len = max(q + c for q, c in zip(query_lens, context_lens))
    blocks_per_seq = triton.cdiv(max_seq_len, block_size)
    num_blocks = len(query_lens) * blocks_per_seq
    key_cache = (
        torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
        / k_scale
    ).to(cache_dtype)
    value_cache = (
        torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.25
        / v_scale
    ).to(cache_dtype)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        len(query_lens), blocks_per_seq
    )
    starts = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.tensor(
        [q + c for q, c in zip(query_lens, context_lens)],
        device=device,
        dtype=torch.int32,
    )

    first = 0
    for seq, (query_len, context_len) in enumerate(zip(query_lens, context_lens)):
        for query_pos in range(query_len):
            cache_pos = context_len + query_pos
            block = block_table[seq, cache_pos // block_size]
            offset = cache_pos % block_size
            key_cache[block, offset] = (key[first + query_pos].float() / k_scale).to(
                cache_dtype
            )
            value_cache[block, offset] = (
                value[first + query_pos].float() / v_scale
            ).to(cache_dtype)
        first += query_len

    references = []
    first = 0
    for seq, (query_len, context_len) in enumerate(zip(query_lens, context_lens)):
        physical = block_table[seq].long()
        seq_len = context_len + query_len
        full_key = (
            key_cache[physical].flatten(0, 1)[:seq_len].float() * k_scale
        ).repeat_interleave(num_heads // num_kv_heads, dim=1)
        full_value = (
            value_cache[physical].flatten(0, 1)[:seq_len].float() * v_scale
        ).repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", query[first : first + query_len].float(), full_key
        ) / math.sqrt(head_size)
        causal_mask = (
            torch.arange(seq_len, device=device)[None, :]
            > context_len + torch.arange(query_len, device=device)[:, None]
        )
        if causal:
            scores.masked_fill_(causal_mask[None], -float("inf"))
        references.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_value))
        first += query_len

    return {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "starts": starts,
        "seq_lens": seq_lens,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "reference": torch.cat(references),
        "max_seq_len": max_seq_len,
    }


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1

    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    # Reshape for SDPA: (seq_len, num_heads, head_size) ->
    # (1, num_heads, seq_len, head_size)
    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )

    # Expand key and value for GQA/MQA to match query heads
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, sliding_window, device=device, dtype=dtype
    )

    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")

    # Reshape output back to (num_tokens, num_heads, head_size)
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    block_size: int = 32,
) -> None:
    # Exercises the KV_FROM_CACHE path of context_attention_fwd: the current
    # chunk K/V are not passed as dense tensors (k=v=None); they are read back
    # from the paged KV cache, as done by layers that re-attend an already
    # cached sequence with the query only (e.g. IQuest LoopCoder's
    # attn(q, None, None)). The whole sequence therefore lives in the cache and
    # the result must still match a dense causal SDPA reference.
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    # Unlike the dense test, write the WHOLE sequence (context + current chunk)
    # into the paged cache, since the current chunk is read back from the cache.
    for i in range(BS):
        cur = 0
        block_id = 0
        while cur < seq_lens[i]:
            start_loc = b_seq_start_loc[i] + cur
            if cur + block_size > seq_lens[i]:
                end_loc = b_seq_start_loc[i] + seq_lens[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur += block_size
            block_id += 1
    # transpose to the paged cache layouts the kernel expects:
    #   K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    #   V_cache[num_blocks, num_kv_heads, head_size, block_size]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=0,
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))

    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )
    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, 0, device=device, dtype=dtype
    )
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_block_table_boundary(device: str) -> None:
    # Boundary guard for the KV_FROM_CACHE block-table load. With an
    # exact-sized block table (row length == number of blocks for the
    # sequence) and a query that ends the sequence, the last K/V tile has
    # padded lanes whose absolute positions step past the sequence end. Those
    # lanes must not read a block-table entry past this batch's row.
    #
    # seq_len is a whole number of blocks, so bn_logical for a padded lane
    # lands exactly on num_blocks (one past the last valid row entry), and
    # query_len is not tile-aligned so the overshoot is actually exercised.
    # The result must still match a dense causal SDPA reference.
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    dtype = torch.float16
    kv_cache_dtype = "auto"
    num_heads = 4
    num_queries_per_kv = 1
    num_kv_heads = num_heads // num_queries_per_kv
    head_size = 32
    block_size = 32
    num_blocks = 4
    seq_len = num_blocks * block_size  # 128 == exact multiple of block_size
    query_len = 99  # < seq_len and not a multiple of the kernel tile

    query = torch.empty(query_len, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(query_len, num_heads, head_size, dtype=dtype)

    kv = torch.empty(seq_len, 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    # Exact-sized block table: row length == num_blocks (identity mapping).
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)

    # Write the whole sequence (context + current chunk) into the paged cache.
    for cur in range(0, seq_len, block_size):
        block_id = cur // block_size
        end = min(cur + block_size, seq_len)
        start_slot = block_table[0, block_id] * block_size
        end_slot = start_slot + (end - cur)
        k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            key[cur:end]
        )
        v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            value[cur:end]
        )

    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        seq_len,
        query_len,
        k_scale,
        v_scale,
        sliding_window=0,
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))
    query_sdpa = query.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, query_len, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size)
    value_sdpa = value[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, seq_len, head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        [query_len], [seq_len], 0, device=device, dtype=dtype
    )
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, query_len, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_alibi_unsupported(device: str) -> None:
    # The cached-K/V (k=None) path is not supported together with ALiBi; the
    # entry point must reject it up-front with a clear NotImplementedError
    # rather than launching the kernel with an unsupported combination.
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    num_heads = 4
    num_kv_heads = 4
    head_size = 16
    x = 8
    block_size = 16
    num_blocks = 4
    query_len = 8

    query = torch.empty(query_len, num_heads, head_size, dtype=torch.float16)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.float16
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([query_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    alibi_slopes = torch.ones(num_heads, dtype=torch.float32, device=device)

    with pytest.raises(NotImplementedError):
        context_attention_fwd(
            query,
            None,
            None,
            output,
            "auto",
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            query_len,
            query_len,
            k_scale,
            v_scale,
            alibi_slopes=alibi_slopes,
        )


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(
                closest_power_of_2, total_num_heads - closest_power_of_2
            )
            extra_powers = torch.arange(
                start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
            )
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")
    scale = float(1.0 / (head_size**0.5))

    # Prepare query, key, value for SDPA
    # Expand key and value for GQA/MQA to match query heads
    key_expanded = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    value_expanded = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )

    output_ref = torch.empty_like(output)

    torch.accelerator.synchronize()
    start_time = time.time()

    query_start = 0
    key_start = 0
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        query_end = query_start + query_len
        key_end = key_start + seq_len

        # Get query, key, value for this sequence
        q = query[query_start:query_end]  # [query_len, num_heads, head_size]
        k = key_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]
        v = value_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]

        # Reshape for SDPA: (batch=1, num_heads, seq_len, head_size)
        q_sdpa = q.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
        q_sdpa = (
            q_sdpa.permute(1, 2, 0, 3)
            .reshape(1, num_heads, query_len, head_size)
            .contiguous()
        )

        k_sdpa = (
            k.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )
        v_sdpa = (
            v.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )

        # Create ALiBi causal mask for this sequence using utility function
        alibi_mask = create_alibi_causal_mask(
            query_len, seq_len, alibi_slopes, device, dtype
        )

        # Compute attention
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=alibi_mask,
            dropout_p=0.0,
            scale=scale,
        )

        # Reshape output back to [query_len, num_heads, head_size]
        out = out.view(num_heads, query_len, head_size).permute(1, 0, 2)
        output_ref[query_start:query_end].copy_(out)

        query_start = query_end
        key_start = key_end

    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


# These tests are optional to only run when explicitly invoked
#
# pytest -v -s --optional \
# tests/kernels/test_prefix_prefill.py::test_contexted_kv_attention_f32
#
# These tests are useful to test model dtype float32 on Turing devices.
# We skip them to not increase the time when running tests on CI
@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention(
        num_heads,
        num_queries_per_kv,
        head_size,
        sliding_window,
        dtype,
        kv_cache_dtype,
        device,
        op,
    )


@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention_alibi(
        num_heads, num_queries_per_kv, head_size, dtype, kv_cache_dtype, device, op
    )


# Hybrid mamba + full-attention models get a non-power-of-2 attention page.
NONSTANDARD_BLOCK_SIZE_SHAPES = [
    (64, 1, 128, 544),
    (8, 4, 256, 1040),
    (8, 4, 256, 1056),
]


@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size,block_size", NONSTANDARD_BLOCK_SIZE_SHAPES
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_qwen3_nonstandard_block_size(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
    op: Callable,
) -> None:
    """Non-power-of-2 pages must match, even when a tile straddles a page."""
    if not current_platform.is_rocm():
        pytest.skip("Non-power-of-2 block sizes are only exercised on ROCm CI.")

    test_contexted_kv_attention(
        num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        block_size=block_size,
        sliding_window=0,
        dtype=dtype,
        kv_cache_dtype="auto",
        device=device,
        op=op,
    )


@pytest.mark.parametrize(
    "dim,page,query_len,context_len",
    [
        (64, 32, 2, 35),
        (128, 16, 3, 65),
        (256, 784, 9, 1601),
        (64, 32, 33, 35),
        (128, 16, 129, 65),
        (256, 1568, 65, 1601),
    ],
)
@torch.inference_mode()
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_rocm_context_tuning_candidates_match_sdpa(
    dim, page, query_len, context_len, kv_dtype
):
    """Launch tuning must preserve attention at ragged tile/page boundaries."""
    if not current_platform.is_rocm():
        pytest.skip("ROCm context launch configurations")
    from vllm.v1.attention.ops.prefix_prefill_tuning import _CONFIGS, _make_inputs

    device = torch.device("cuda:0")
    q, k, v, kc, vc, table, starts, lengths, one = _make_inputs(
        device, 4, 2, dim, page, 2, query_len, query_len + context_len, kv_dtype
    )
    output = torch.empty_like(q)
    references = []
    v_scale = one * 2 if kv_dtype == torch.float8_e4m3fn else one
    for b in range(2):
        context_k = (
            kc.float()[table[b].long()]
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, 2, dim)[:context_len]
            * one
        )
        context_v = (
            vc.float()[table[b].long()]
            .permute(0, 3, 1, 2)
            .reshape(-1, 2, dim)[:context_len]
            * v_scale
        )
        full_k = torch.cat(
            (context_k, k[b * query_len : (b + 1) * query_len])
        ).repeat_interleave(2, dim=1)
        full_v = torch.cat(
            (context_v, v[b * query_len : (b + 1) * query_len])
        ).repeat_interleave(2, dim=1)
        mask = torch.arange(query_len + context_len, device=device)[None, :] <= (
            torch.arange(query_len, device=device)[:, None] + context_len
        )
        references.append(
            F.scaled_dot_product_attention(
                q[b * query_len : (b + 1) * query_len].float().transpose(0, 1),
                full_k.transpose(0, 1),
                full_v.transpose(0, 1),
                attn_mask=mask,
            ).transpose(0, 1)
        )
    reference = torch.cat(references)
    for config in _CONFIGS:
        try:
            context_attention_fwd(
                q,
                k,
                v,
                output,
                "auto" if kv_dtype == torch.bfloat16 else "fp8",
                kc,
                vc,
                table,
                starts,
                lengths,
                query_len + context_len,
                query_len,
                one,
                v_scale,
                skip_decode=True,
                _launch_config=config,
            )
        except triton.OutOfResources:
            continue
        torch.testing.assert_close(output.float(), reference, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_rocm_context_tuning_persists_without_retuning(tmp_path, monkeypatch, kv_dtype):
    """A fresh table must reuse disk winners without invoking the benchmark."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops import prefix_prefill_tuning as tuning

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", gcnArchName="gfx1201", multi_processor_count=64
        ),
    )
    monkeypatch.setattr(tuning, "_TABLES", {})
    monkeypatch.setattr(tuning, "_memory_budget", lambda _: 2**50)

    def tune(*args):
        return {"workload": list(args[-1]), "best": tuning._DEFAULT, "results": []}

    monkeypatch.setattr(tuning, "_tune_workload", tune)
    args = (torch.device("cuda:0"), torch.bfloat16, 4, 2, 64, 32, 0.125, 32, 64, 1)
    tuning.warmup_context_attention(*args, kv_dtype=kv_dtype)
    cache = next(tmp_path.rglob("*.json"))
    saved = cache.read_bytes(), cache.stat().st_mtime_ns
    tuning._TABLES.clear()

    def forbidden(*args):
        pytest.fail("Cached engine startup attempted to tune")

    monkeypatch.setattr(tuning, "_tune_workload", forbidden)
    tuning.warmup_context_attention(*args, kv_dtype=kv_dtype)
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == saved
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"),
            4,
            2,
            64,
            32,
            1,
            31,
            63,
            0.125,
            kv_dtype=kv_dtype,
        )
        == tuning._DEFAULT
    )
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"),
            4,
            2,
            64,
            32,
            1,
            128,
            128,
            0.125,
            kv_dtype=kv_dtype,
        )
        is None
    )
    other_dtype = torch.float8_e4m3fn if kv_dtype == torch.bfloat16 else torch.bfloat16
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"),
            4,
            2,
            64,
            32,
            1,
            31,
            63,
            0.125,
            kv_dtype=other_dtype,
        )
        is None
    )


def test_rocm_context_tuning_recovers_corrupt_cache(tmp_path, monkeypatch):
    """A truncated cache must be replaced rather than accepted as a winner."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops import prefix_prefill_tuning as tuning

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", gcnArchName="gfx1201", multi_processor_count=64
        ),
    )
    monkeypatch.setattr(tuning, "_TABLES", {})
    monkeypatch.setattr(tuning, "_memory_budget", lambda _: 2**50)
    calls = []

    def tune(*args):
        calls.append(args[-1])
        return {"workload": list(args[-1]), "best": tuning._DEFAULT, "results": []}

    monkeypatch.setattr(tuning, "_tune_workload", tune)
    args = (torch.device("cuda:0"), torch.bfloat16, 4, 2, 64, 32, 0.125, 32, 64, 1)
    tuning.warmup_context_attention(*args)
    original_workloads = calls.copy()
    cache = next(tmp_path.rglob("*.json"))
    cache.write_text('{"identity":')
    tuning._TABLES.clear()
    calls.clear()
    tuning.warmup_context_attention(*args)
    assert calls == original_workloads
    import json

    assert [r["workload"] for r in json.loads(cache.read_text())["records"]] == [
        list(workload) for workload in original_workloads
    ]


def test_rocm_context_tuning_extends_buckets_and_reuses_saved_limits(
    tmp_path, monkeypatch
):
    """Reuse saved ceiling buckets and tune only missing shapes after limit changes."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops import prefix_prefill_tuning as tuning

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", gcnArchName="gfx1201", multi_processor_count=64
        ),
    )
    monkeypatch.setattr(tuning, "_TABLES", {})
    monkeypatch.setattr(tuning, "_memory_budget", lambda _: 2**50)
    calls = []

    def tune(*args):
        workload = args[-1]
        calls.append(workload)
        best = tuning._CONFIGS[0] if workload[1] == 8192 else tuning._DEFAULT
        if workload[1] == 4:
            best = tuning._CONFIGS[1]
        return {"workload": list(workload), "best": best, "results": []}

    monkeypatch.setattr(tuning, "_tune_workload", tune)
    args = (
        torch.device("cuda:0"),
        torch.bfloat16,
        4,
        2,
        64,
        32,
        0.125,
        65536,
        65536,
        1,
    )
    tuning.warmup_context_attention(*args)
    assert len(calls) == 120
    assert {q for _, q, _ in calls} == {2**exponent for exponent in range(1, 17)}
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"), 4, 2, 64, 32, 1, 3, 4099, 0.125
        )
        == tuning._CONFIGS[1]
    )
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"), 4, 2, 64, 32, 1, 5155, 5155, 0.125
        )
        == tuning._CONFIGS[0]
    )
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"), 4, 2, 64, 32, 1, 65536, 65536, 0.125
        )
        is not None
    )
    assert (
        tuning.get_context_attention_config(
            torch.device("cuda:0"), 4, 2, 64, 32, 1, 65537, 65537, 0.125
        )
        is None
    )
    calls.clear()
    short = (*args[:-3], 2048, 2048, 1)
    tuning.warmup_context_attention(*short)
    assert set(calls) == {(1, q, 2048) for q in (2, 4, 8, 16, 32, 64, 128, 256, 512)}
    cache = next(tmp_path.rglob("*.json"))
    saved = cache.read_bytes(), cache.stat().st_mtime_ns
    tuning._TABLES.clear()

    def forbidden(*args):
        pytest.fail("Saved long/short engine startup attempted to tune")

    monkeypatch.setattr(tuning, "_tune_workload", forbidden)
    tuning.warmup_context_attention(*args)
    tuning.warmup_context_attention(*short)
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == saved


def test_rocm_context_tuning_prunes_tokens_and_paged_kv_memory():
    """Long-prefix coverage must not allocate impossible independent KV batches."""
    from vllm.v1.attention.ops import prefix_prefill_tuning as tuning

    limits = (8192, 524288, 64)
    raw = set(tuning._workloads(*limits))
    assert {b for b, _, _ in raw} == {1, 2, 4, 8, 16, 32}
    assert (1, 2, 262146) in raw
    assert all(
        sum(tuning._query_lengths(b, q, 8192)) <= 8192 and s <= 524288
        for b, q, s in raw
    )
    assert (4, 8192, 9216) in raw
    assert tuning._query_lengths(4, 8192, 8192) == [8189, 1, 1, 1]
    for token_limit in (17, 32, 8192):
        bounded = set(tuning._workloads(token_limit, 9216, 32))
        query_buckets = sorted({q for _, q, _ in bounded})
        for b, q, _ in bounded:
            actual = max(tuning._query_lengths(b, q, token_limit))
            assert next(bucket for bucket in query_buckets if bucket >= actual) == q

    scratch = set(tuning._workloads(*limits, memory_budget_bytes=4 * 2**30))
    assert (1, 2, 262146) in scratch
    assert (32, 2, 262146) not in scratch
    assert all(
        tuning._scratch_bytes(12, 2, 256, 784, *w, max_tokens=8192) <= 4 * 2**30
        for w in scratch
    )

    layouts = ((784, 784 * 2048 * 12), (0, 8 * 2**20))
    model = set(
        tuning._workloads(
            *limits,
            memory_budget_bytes=4 * 2**30,
            cache_layouts=layouts,
            cache_budget_bytes=2**30,
        )
    )
    assert model < scratch
    assert (1, 2, 262146) not in model
    assert (32, 2, 2) in model
    assert all(
        b
        * sum(
            ((s + page - 1) // page) * size if page else size for page, size in layouts
        )
        <= 2**30
        for b, _, s in model
    )


@pytest.mark.parametrize(
    "dim,page,hq,hk,dtype,kv_dtype,byte_cache,force_splits",
    [
        (128, 16, 4, 4, torch.bfloat16, torch.bfloat16, False, None),
        (128, 32, 8, 2, torch.float16, torch.float16, False, None),
        (256, 784, 12, 2, torch.bfloat16, torch.bfloat16, False, None),
        (256, 1568, 12, 2, torch.bfloat16, torch.float8_e4m3fn, False, None),
        (128, 32, 16, 1, torch.bfloat16, torch.float8_e4m3fn, True, None),
        (256, 32, 4, 2, torch.float16, torch.float8_e4m3fn, False, None),
        (128, 32, 8, 2, torch.bfloat16, torch.bfloat16, False, 32),
    ],
)
@torch.inference_mode()
def test_segmented_prefill_ragged_fresh_kv_graph_ownership(
    monkeypatch, dim, page, hq, hk, dtype, kv_dtype, byte_cache, force_splits
):
    """Short/long/decode ownership follows changed metadata on graph replay."""
    from tests.kernels.attention.test_splitkv_paged_decode import _pack_cache
    from vllm import envs
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops.prefix_prefill_tuning import _make_inputs

    if not current_platform.is_rocm() or not (
        on_gfx12x() if kv_dtype.itemsize == 1 else on_gfx1x()
    ):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    monkeypatch.setattr(envs, "VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE", False)
    if force_splits is not None:
        from vllm.v1.attention.ops import segmented_prefill as segmented

        original = segmented.select_segmented_config
        monkeypatch.setattr(
            segmented,
            "select_segmented_config",
            lambda *args: dict(original(*args), splits=force_splits),
        )
    query_lengths = [0, 1, 2, 7, 33, 129]
    contexts = [0, 67, 0, page - 1, page + 1, 33]
    device = torch.device("cuda:0")
    q, k, v, kc, vc, table, starts, lengths, _ = _make_inputs(
        device,
        hq,
        hk,
        dim,
        page,
        len(contexts),
        max(query_lengths),
        max(query_lengths) + max(contexts),
        kv_dtype,
    )
    total = sum(query_lengths)
    tensors = []
    for source, heads in ((q, hq), (k, hk), (v, hk)):
        target = torch.empty((total, heads + 1, dim), device=device, dtype=dtype)
        target = target[:, :heads]
        target.copy_(source[:total])
        tensors.append(target)
    q, k, v = tensors
    # Recover NHD only to reuse the established interleaved-cache packing helper.
    dense_k = kc.permute(0, 3, 1, 2, 4).reshape(kc.shape[0], page, hk, dim)
    dense_v = vc.permute(0, 3, 1, 2)
    if kv_dtype in (torch.float16, torch.bfloat16):
        dense_k, dense_v = dense_k.to(dtype), dense_v.to(dtype)
    kc, vc = _pack_cache(dense_k, dense_v, padded_stride=True)
    table.copy_(torch.randperm(table.numel(), device=device).reshape_as(table))
    ks = torch.tensor(0.13 if kv_dtype.itemsize == 1 else 1.0, device=device)
    vs = torch.tensor(0.27 if kv_dtype.itemsize == 1 else 1.0, device=device)
    output = torch.full((total, hq + 1, dim), 17.0, device=device, dtype=dtype)[:, :hq]

    def metadata(qlens):
        starts.copy_(
            torch.tensor(
                [0, *torch.tensor(qlens).cumsum(0).tolist()],
                device=device,
                dtype=torch.int32,
            )
        )
        lengths.copy_(
            torch.tensor(
                [c + n for c, n in zip(contexts, qlens)],
                device=device,
                dtype=torch.int32,
            )
        )

    def run():
        context_attention_fwd(
            q,
            k,
            v,
            output,
            "fp8" if kv_dtype.itemsize == 1 else "auto",
            kc.view(torch.uint8) if byte_cache else kc,
            vc.view(torch.uint8) if byte_cache else vc,
            table,
            starts,
            lengths,
            max(contexts) + 129,
            129,
            ks,
            vs,
            skip_decode=True,
        )

    def check(qlens):
        first = 0
        for seq, (nq, context) in enumerate(zip(qlens, contexts)):
            if nq <= 1:
                assert torch.all(output[first : first + nq] == 17.0)
                first += nq
                continue
            physical = table[seq].long()
            prefix_k = (
                kc[physical]
                .permute(0, 3, 1, 2, 4)
                .reshape(-1, hk, dim)[:context]
                .float()
                * ks
            )
            prefix_v = (
                vc[physical].permute(0, 3, 1, 2).reshape(-1, hk, dim)[:context].float()
                * vs
            )
            full_k = torch.cat(
                (prefix_k, k[first : first + nq].float())
            ).repeat_interleave(hq // hk, 1)
            full_v = torch.cat(
                (prefix_v, v[first : first + nq].float())
            ).repeat_interleave(hq // hk, 1)
            logits = torch.einsum(
                "qhd,khd->hqk", q[first : first + nq].float(), full_k
            ) / math.sqrt(dim)
            mask = (
                torch.arange(context + nq, device=device)[None, :]
                > context + torch.arange(nq, device=device)[:, None]
            )
            logits.masked_fill_(mask[None], -float("inf"))
            ref = torch.einsum("hqk,khd->qhd", logits.softmax(-1), full_v)
            actual = output[first : first + nq].float()
            assert torch.isfinite(actual).all()
            relative = (actual - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)
            assert relative.max().item() < 0.01
            first += nq

    metadata(query_lengths)
    run()
    torch.cuda.synchronize()
    check(query_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replay_lengths = [1, 0, 7, 2, 34, 128]
    metadata(replay_lengths)
    q.mul_(0.75)
    k.add_(0.25)
    v.add_(0.5)
    output.fill_(17.0)
    graph.replay()
    torch.cuda.synchronize()
    check(replay_lengths)


@pytest.mark.parametrize(
    "fp8,force_splits",
    [(False, None), (True, None), (False, 4), (True, 4)],
)
@torch.inference_mode()
def test_chunked_prefill_routes_unified_cache_to_segmented(
    monkeypatch, fp8, force_splits
):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops import segmented_prefill as segmented

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    case = _make_unified_paged_case(
        [1, 2, 7, 129],
        [31, 64, 97, 128],
        num_heads=12,
        num_kv_heads=2,
        head_size=256,
        block_size=32,
        fp8=fp8,
    )
    output = torch.empty_like(case["query"])
    routed = []
    original = segmented.segmented_prefill_attention
    if force_splits is not None:
        original_selector = segmented.select_segmented_unified_config
        monkeypatch.setattr(
            segmented,
            "select_segmented_unified_config",
            lambda *args: dict(original_selector(*args), splits=force_splits),
        )

    def segmented_spy(*args, **kwargs):
        routed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(segmented, "segmented_prefill_attention", segmented_spy)
    chunked_prefill_paged_decode(
        query=case["query"],
        key=case["key"],
        value=case["value"],
        output=output,
        kv_cache_dtype="fp8" if fp8 else "auto",
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        block_table=case["block_table"],
        query_start_loc=case["starts"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        max_query_len=129,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
    )

    assert routed == [True]
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_rocm_attn_segmented_layout_cache_update(monkeypatch, fp8):
    from types import SimpleNamespace

    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    kv_cache_dtype = "fp8" if fp8 else "auto"
    impl = RocmAttentionImpl(
        12,
        256,
        256**-0.5,
        2,
        None,
        None,
        kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )
    assert impl._use_unified_kv_layout
    cache = torch.zeros(3, 2, 32, 512, device="cuda", dtype=cache_dtype)
    key = torch.randn(5, 2, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 7, 32, 65, 95], device="cuda", dtype=torch.int64)
    k_scale = torch.tensor(0.125 if fp8 else 1.0, device="cuda")
    v_scale = torch.tensor(0.25 if fp8 else 1.0, device="cuda")
    layer = SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)

    impl.do_kv_cache_update(layer, key, value, cache, slots)
    key_cache, value_cache = impl._split_kv_cache(cache)
    if fp8:
        key_cache = key_cache.view(impl.fp8_dtype)
        value_cache = value_cache.view(impl.fp8_dtype)
    cached_key = torch.stack(
        [key_cache[int(slot) // 32, int(slot) % 32] for slot in slots]
    )
    cached_value = torch.stack(
        [value_cache[int(slot) // 32, int(slot) % 32] for slot in slots]
    )
    expected_key = (key.float() / k_scale).to(key_cache.dtype)
    expected_value = (value.float() / v_scale).to(value_cache.dtype)
    torch.testing.assert_close(cached_key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(cached_value, expected_value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kv_cache_dtype,is_gfx1x,is_gfx12x,expected",
    [
        ("auto", True, False, True),
        ("fp8", True, False, False),
        ("auto", False, False, False),
        ("fp8", True, True, True),
    ],
)
def test_rocm_attn_segmented_layout_arch_gate(
    monkeypatch, kv_cache_dtype, is_gfx1x, is_gfx12x, expected
):
    from vllm.platforms import rocm
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl

    monkeypatch.setattr(rocm, "on_gfx1x", lambda: is_gfx1x)
    monkeypatch.setattr(rocm, "on_gfx12x", lambda: is_gfx12x)
    impl = RocmAttentionImpl(
        12,
        256,
        256**-0.5,
        2,
        None,
        None,
        kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )
    assert impl._use_unified_kv_layout is expected


@pytest.mark.parametrize(
    "query_lens,context_lens,max_query_len,causal",
    [([129, 1], [31, 64], 4097, True), ([33, 1], [31, 64], 33, False)],
)
@torch.inference_mode()
def test_chunked_prefill_unified_cache_routes_unsupported_segmented_patterns(
    monkeypatch, query_lens, context_lens, max_query_len, causal
):
    import importlib

    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops import segmented_prefill as segmented
    from vllm.v1.attention.ops import triton_unified_attention as unified

    chunked = importlib.import_module(
        "vllm.v1.attention.ops.chunked_prefill_paged_decode"
    )

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x unified prefill fallback")
    case = _make_unified_paged_case(
        query_lens,
        context_lens,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        block_size=32,
        fp8=False,
        causal=causal,
    )
    output = torch.empty_like(case["query"])
    routed = []
    original_unified = unified.unified_attention
    original_context = chunked.context_attention_fwd

    def unified_spy(*args, **kwargs):
        routed.append("unified")
        return original_unified(*args, **kwargs)

    def context_spy(*args, **kwargs):
        routed.append("context")
        return original_context(*args, **kwargs)

    monkeypatch.setattr(unified, "unified_attention", unified_spy)
    monkeypatch.setattr(chunked, "context_attention_fwd", context_spy)
    monkeypatch.setattr(
        segmented,
        "segmented_prefill_attention",
        lambda *args, **kwargs: pytest.fail(
            "unsupported pattern routed to segmented prefill"
        ),
    )
    chunked_prefill_paged_decode(
        query=case["query"],
        key=case["key"],
        value=case["value"],
        output=output,
        kv_cache_dtype="auto",
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        block_table=case["block_table"],
        query_start_loc=case["starts"],
        seq_lens=case["seq_lens"],
        max_seq_len=case["max_seq_len"],
        max_query_len=max_query_len,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        causal=causal,
    )

    assert routed == ["context" if causal else "unified"]
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@torch.inference_mode()
def test_segmented_prefill_cache_offsets_cross_int32_boundary():
    """A small logical prefix can live beyond 2**31 elements in a strided cache."""
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops.segmented_prefill import segmented_prefill_attention

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x segmented prefill")
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 10 * 2**30:
        pytest.skip("The address-boundary fixture needs 10 GiB free VRAM")
    device = torch.device("cuda:0")
    stride = 2**30 + 4096
    kc = torch.empty_strided(
        (3, 1, 16, 32, 8),
        (stride, 4096, 256, 8, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_strided(
        (3, 1, 128, 32), (stride, 4096, 32, 1), device=device, dtype=torch.bfloat16
    )
    dense_k = torch.randn(2, 32, 1, 128, device=device, dtype=torch.bfloat16)
    dense_v = torch.randn_like(dense_k)
    for logical, physical in enumerate((2, 1)):
        kc[physical].copy_(dense_k[logical].reshape(32, 1, 16, 8).permute(1, 2, 0, 3))
        vc[physical].copy_(dense_v[logical].permute(1, 2, 0))
    q = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 1, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    table = torch.tensor([[2, 1]], device=device, dtype=torch.int32)
    starts = torch.tensor([0, 2], device=device, dtype=torch.int32)
    lengths = torch.tensor([35], device=device, dtype=torch.int32)
    one = torch.ones((), device=device)
    segmented_prefill_attention(
        q, k, v, out, kc, vc, table, starts, lengths, 2, 35, one, one, 128**-0.5
    )
    full_k = torch.cat((dense_k.flatten(0, 1)[:33], k)).float().repeat_interleave(4, 1)
    full_v = torch.cat((dense_v.flatten(0, 1)[:33], v)).float().repeat_interleave(4, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), full_k) / math.sqrt(128)
    scores[:, 0, -1] = -float("inf")
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_v)
    error = ((out.float() - expected).norm(dim=-1) / expected.norm(dim=-1)).max()
    assert torch.isfinite(out).all() and error < 0.01
