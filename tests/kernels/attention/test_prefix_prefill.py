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
    sliding_window: int = 0,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
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
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        query_positions = context_len + torch.arange(query_len, device=device)
        key_positions = torch.arange(seq_len, device=device)
        causal_mask = key_positions[None, :] > query_positions[:, None]
        if causal:
            scores.masked_fill_(causal_mask[None], -float("inf"))
        if sliding_window > 0:
            window_mask = key_positions[None, :] < (
                query_positions[:, None] - sliding_window + 1
            )
            scores.masked_fill_(window_mask[None], -float("inf"))
        if sinks is not None:
            sink_scores = sinks.float()[:, None, None].expand(-1, query_len, 1)
            probabilities = torch.cat((scores, sink_scores), dim=-1).softmax(-1)
            probabilities = probabilities[..., :-1]
        else:
            probabilities = scores.softmax(-1)
        references.append(torch.einsum("hqk,khd->qhd", probabilities, full_value))
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


def test_segmented_tuning_candidates_preserve_workspace_bound():
    """Startup candidates may reduce but never enlarge reserved split scratch."""
    from vllm.v1.attention.ops import segmented_prefill as segmented
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    for batch, query_len, seq_len, heads, kv_heads, dim, fp8 in (
        (1, 1, 128, 8, 8, 128, True),
        (32, 1, 262144, 16, 1, 128, True),
        (4, 1024, 8192, 16, 4, 256, False),
        (1, 1024, 131072, 6, 1, 256, True),
    ):
        default = segmented.select_segmented_config(
            batch, query_len, seq_len, heads, kv_heads, dim, fp8
        )
        candidates = tuning._candidate_configs(default, batch, query_len, heads, dim)
        assert 1 <= len(candidates) <= tuning._MAX_CANDIDATES
        assert (
            tuning._normalized_config(default, batch, query_len, heads, dim)
            in candidates
        )
        assert all(config["splits"] <= default["splits"] for config in candidates)
        assert all(dim % config["bk"] == 0 for config in candidates)


def test_segmented_fp8_d256_long_extend_config():
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning
    from vllm.v1.attention.ops.segmented_prefill import select_segmented_config

    for query_len, seq_len, splits in (
        (256, 8192, 4),
        (1024, 131072, 8),
        (4096, 262144, 2),
    ):
        config = select_segmented_config(1, query_len, seq_len, 6, 1, 256, True)
        assert config == {
            "bm": 64,
            "bn": 128,
            "bk": 64,
            "splits": splits,
            "warps": 8,
            "stages": 1,
            "waves_per_eu": 6,
        }
    candidates = tuning._candidate_configs(config, 1, query_len, 6, 256)
    assert any(candidate["waves_per_eu"] == 2 for candidate in candidates)

    assert "waves_per_eu" not in select_segmented_config(
        1, 128, 131072, 6, 1, 256, True
    )
    assert "waves_per_eu" not in select_segmented_config(1, 256, 4096, 6, 1, 256, True)


def test_segmented_fp8_d128_long_extend_config():
    from vllm.v1.attention.ops.segmented_prefill import select_segmented_config

    for query_len, seq_len, heads, splits in (
        (256, 8192, 6, 4),
        (512, 32768, 1, 16),
        (1024, 131072, 6, 8),
        (4096, 262144, 16, 1),
        (256, 131072, 1, 32),
    ):
        config = select_segmented_config(1, query_len, seq_len, heads, 1, 128, True)
        assert config == {
            "bm": 64,
            "bn": 128,
            "bk": 64,
            "splits": splits,
            "warps": 8,
            "stages": 3,
            "waves_per_eu": 6,
            "prefix_fast": True,
            **({"qk_pipeline": 3} if query_len >= 512 or seq_len >= 32768 else {}),
        }

    for query_len, seq_len, heads in (
        (256, 8192, 1),
        (256, 32768, 1),
        (512, 8192, 1),
        (1024, 8192, 2),
        (256, 4096, 6),
    ):
        config = select_segmented_config(1, query_len, seq_len, heads, 1, 128, True)
        assert "waves_per_eu" not in config
        assert "prefix_fast" not in config
        assert "qk_pipeline" not in config


@pytest.mark.parametrize("dim", (128, 256))
def test_segmented_fp8_long_extend_split_workspace_bound(dim):
    from vllm.v1.attention.ops.segmented_prefill import (
        MAX_LONG_EXTEND_WORKSPACE_BYTES,
        segmented_query_capacity,
        select_segmented_config,
    )

    for batch, query_len, seq_len, heads, expected_splits in (
        (1, 256, 8192, 6, 4),
        (1, 256, 32768, 6, 16),
        (1, 256, 131072, 6, 32),
        (1, 1024, 32768, 1, 16),
        (1, 1024, 131072, 1, 32),
        (1, 1024, 131072, 6, 8),
        (1, 4096, 131072, 1, 8),
        (4, 1024, 131072, 6, 2),
        (5, 257, 131072, 6, 4),
        (21, 256, 131072, 6, 2),
    ):
        config = select_segmented_config(batch, query_len, seq_len, heads, 1, dim, True)
        assert config["splits"] == expected_splits
        if seq_len >= 32768:
            scratch = (
                batch
                * segmented_query_capacity(query_len)
                * heads
                * config["splits"]
                * (dim + 1)
                * 4
            )
            assert scratch <= MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128

    for batch in (1, 4, 16):
        for query_len in (257, 1025, 4095):
            for ratio in (1, 6, 16):
                for kv_heads in (1, 4):
                    for seq_len in (32768, 131072):
                        heads = ratio * kv_heads
                        config = select_segmented_config(
                            batch, query_len, seq_len, heads, kv_heads, dim, True
                        )
                        if (
                            config.get("waves_per_eu") != 6
                            or config["bm"] != 64
                            or config["splits"] == 1
                        ):
                            continue
                        scratch = (
                            batch
                            * segmented_query_capacity(query_len)
                            * heads
                            * config["splits"]
                            * (dim + 1)
                            * 4
                        )
                        assert scratch <= MAX_LONG_EXTEND_WORKSPACE_BYTES * dim // 128


def test_segmented_bf16_d256_long_extend_config():
    from vllm.v1.attention.ops.segmented_prefill import select_segmented_config

    for query_len, seq_len in ((1024, 8192), (4096, 262144)):
        config = select_segmented_config(1, query_len, seq_len, 6, 1, 256, False)
        assert config == {
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "splits": 1,
            "warps": 8,
            "stages": 3,
            "waves_per_eu": 6,
            "pv_split": True,
            "qk_pipeline": 3,
        }

    assert "waves_per_eu" not in select_segmented_config(
        1, 256, 262144, 6, 1, 256, False
    )
    assert "waves_per_eu" not in select_segmented_config(
        1, 1024, 4096, 6, 1, 256, False
    )


@torch.inference_mode()
def test_segmented_bf16_d256_long_extend_matches_dense_reference():
    from vllm.v1.attention.ops.segmented_prefill import segmented_prefill_attention

    case = _make_unified_paged_case(
        [1024],
        [7168],
        num_heads=6,
        num_kv_heads=1,
        head_size=256,
        block_size=1568,
        fp8=False,
    )
    output = torch.empty_like(case["query"])
    segmented_prefill_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        1024,
        case["max_seq_len"],
        case["k_scale"],
        case["v_scale"],
        256**-0.5,
        skip_decode=False,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize(
    "query_len,heads,qk_amplitude,dim",
    [
        (256, 1, 1, 256),
        (256, 6, 1, 256),
        (512, 6, 1, 256),
        (256, 6, 4, 256),
        (256, 6, 16, 256),
        (256, 6, 1, 128),
        (512, 6, 4, 128),
        (256, 6, 16, 128),
    ],
)
@torch.inference_mode()
def test_segmented_fp8_long_extend_matches_dense_reference(
    query_len, heads, qk_amplitude, dim
):
    from vllm.platforms.rocm import on_gfx12x
    from vllm.v1.attention.ops.segmented_prefill import segmented_prefill_attention

    if not current_platform.is_rocm() or not on_gfx12x():
        pytest.skip("FP8 KV requires gfx12")
    case = _make_unified_paged_case(
        [query_len],
        [8192 - query_len],
        num_heads=heads,
        num_kv_heads=1,
        head_size=dim,
        block_size=1568,
        fp8=True,
    )
    if qk_amplitude != 1:
        # Check peaked logits too: one-term FP8 Q quantization passed only the
        # nearly uniform fixture and deviated by 4-70% at these amplitudes.
        case["query"].mul_(qk_amplitude)
        case["k_scale"].mul_(qk_amplitude)
        full_key = (
            case["key_cache"][case["block_table"][0].long()]
            .flatten(0, 1)[:8192]
            .float()
            * case["k_scale"]
        ).repeat_interleave(heads, dim=1)
        full_value = (
            case["value_cache"][case["block_table"][0].long()]
            .flatten(0, 1)[:8192]
            .float()
            * case["v_scale"]
        ).repeat_interleave(heads, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", case["query"].float(), full_key
        ) / math.sqrt(dim)
        future = torch.arange(8192, device=case["query"].device)[None, :] > (
            8192
            - query_len
            + torch.arange(query_len, device=case["query"].device)[:, None]
        )
        scores.masked_fill_(future[None], -float("inf"))
        case["reference"] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_value)
    output = torch.empty_like(case["query"])
    segmented_prefill_attention(
        case["query"],
        output,
        case["key_cache"],
        case["value_cache"],
        case["block_table"],
        case["starts"],
        case["seq_lens"],
        query_len,
        case["max_seq_len"],
        case["k_scale"],
        case["v_scale"],
        dim**-0.5,
        skip_decode=False,
    )
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


def test_segmented_tuning_protects_static_incumbent():
    """Noise-sized gains must not replace the static configuration."""
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    incumbent = {"bm": 16}
    challenger = {"bm": 32}
    samples = {
        tuning._config_key(incumbent): [100.0, 101.0, 99.0, 100.0, 100.0],
        tuning._config_key(challenger): [99.0, 100.0, 98.0, 99.0, 99.0],
    }
    winner, comparisons = tuning._select_tuned_winner(
        incumbent, [incumbent, challenger], samples
    )
    assert winner["config"] == incumbent
    assert comparisons[1]["paired_speedup_vs_default"] < 1.02


def test_segmented_tuning_promotes_verified_challenger():
    """A finalist with a stable material gain should replace the incumbent."""
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    incumbent = {"bm": 16}
    challenger = {"bm": 32}
    samples = {
        tuning._config_key(incumbent): [100.0, 102.0, 98.0, 101.0, 99.0],
        tuning._config_key(challenger): [94.0, 96.0, 92.0, 95.0, 93.0],
    }
    winner, _ = tuning._select_tuned_winner(incumbent, [incumbent, challenger], samples)
    assert winner["config"] == challenger
    assert winner["paired_speedup_vs_default"] > 1.02


def test_segmented_tuning_balances_tp_workloads_without_overlap():
    """Every missing bucket belongs to exactly one reasonably balanced rank."""
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    workloads = list(tuning._workloads(8192, 262144, 32))
    shards = tuning._shard_workloads(
        workloads,
        4,
        8192,
        12,
        2,
        256,
        True,
    )
    flattened = [workload for shard in shards for workload in shard]
    assert sorted(flattened) == sorted(workloads)
    assert len(flattened) == len(set(flattened))
    owners = {
        workload[:2]: rank for rank, shard in enumerate(shards) for workload in shard
    }
    assert all(
        owners[workload[:2]] == rank
        for rank, shard in enumerate(shards)
        for workload in shard
    )

    weights = [
        sum(
            tuning._workload_weight(workload, 8192, 12, 2, 256, True)
            for workload in shard
        )
        for shard in shards
    ]
    largest_group = max(
        sum(
            tuning._workload_weight(workload, 8192, 12, 2, 256, True)
            for workload in workloads
            if workload[:2] == group
        )
        for group in {workload[:2] for workload in workloads}
    )
    assert max(weights) - min(weights) <= largest_group


def test_segmented_tuning_prunes_scheduler_and_kv_limits():
    """Generated buckets must be reachable under scheduler and cache limits."""
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    limits = (8192, 262144, 32)
    raw = set(
        tuning._workloads(
            *limits,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            heads=16,
            kv_heads=1,
            dim=128,
            page=16,
        )
    )
    assert {batch for batch, _, _ in raw} == {1, 2, 4, 8, 16, 32}
    assert {query for _, query, _ in raw} == set(tuning._QUERY_BUCKETS)
    assert (32, 1, 262144) in raw
    assert all(
        sum(tuning._query_lengths(batch, query, limits[0])) <= limits[0]
        and batch <= limits[2]
        and query <= min(limits[0], tuning.MAX_QUERY_LEN)
        and query <= seq_len <= limits[1]
        for batch, query, seq_len in raw
    )

    layouts = ((16, 16 * 128 * 2),)
    bounded = set(
        tuning._workloads(
            *limits,
            memory_budget_bytes=2**50,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            heads=16,
            kv_heads=1,
            dim=128,
            page=16,
            cache_layouts=layouts,
            cache_budget_bytes=2**20,
        )
    )
    assert bounded < raw
    assert all(
        batch * sum(((seq_len + block - 1) // block) * size for block, size in layouts)
        <= 2**20
        for batch, _, seq_len in bounded
    )


def test_segmented_tuning_persists_without_retuning(tmp_path, monkeypatch):
    """A fresh process table must reuse persistent segmented winners."""
    from types import SimpleNamespace

    from vllm.v1.attention.ops import segmented_prefill as segmented
    from vllm.v1.attention.ops import segmented_prefill_tuning as tuning

    monkeypatch.setenv("VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test", gcnArchName="gfx1201", multi_processor_count=32
        ),
    )
    monkeypatch.setattr(tuning, "_TABLES", {})
    monkeypatch.setattr(tuning, "_memory_budget", lambda _: 2**50)
    calls = []

    def tune(*args):
        heads, kv_heads, dim = args[3:6]
        batch, query_len, seq_len = args[-1]
        default = segmented.select_segmented_config(
            batch, query_len, seq_len, heads, kv_heads, dim, False
        )
        best = tuning._candidate_configs(default, batch, query_len, heads, dim)[0]
        calls.append(args[-1])
        return {
            "workload": list(args[-1]),
            "query_lengths": tuning._query_lengths(batch, query_len, args[-2]),
            "default": default,
            "best": best,
            "results": [],
        }

    monkeypatch.setattr(tuning, "_tune_workload", tune)
    args = (
        torch.device("cuda:0"),
        torch.bfloat16,
        4,
        2,
        128,
        16,
        128**-0.5,
        4,
        8,
        2,
    )
    tuning.warmup_segmented_attention(*args)
    assert calls
    first_call_count = len(calls)
    cache = next(tmp_path.rglob("*.json"))
    saved = cache.read_bytes(), cache.stat().st_mtime_ns
    expected = tuning.get_segmented_config(
        torch.device("cuda:0"),
        torch.bfloat16,
        torch.bfloat16,
        4,
        2,
        128,
        16,
        128**-0.5,
        1,
        1,
        8,
    )
    assert expected is not None

    # The maximum token count changes the ragged query mix for a bucket, so it
    # must select a distinct persistent cache instead of reusing this one.
    tuning.warmup_segmented_attention(*args[:7], 3, *args[8:])
    assert len(calls) > first_call_count
    assert len(list(tmp_path.rglob("*.json"))) == 2

    tuning._TABLES.clear()

    def forbidden(*args):
        pytest.fail("Persistent segmented startup cache attempted to retune")

    monkeypatch.setattr(tuning, "_tune_workload", forbidden)
    tuning.warmup_segmented_attention(*args)
    assert (cache.read_bytes(), cache.stat().st_mtime_ns) == saved
    assert (
        tuning.get_segmented_config(
            torch.device("cuda:0"),
            torch.bfloat16,
            torch.bfloat16,
            4,
            2,
            128,
            16,
            128**-0.5,
            1,
            1,
            8,
        )
        == expected
    )
    monkeypatch.setenv("VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE", "0")
    assert (
        tuning.get_segmented_config(
            torch.device("cuda:0"),
            torch.bfloat16,
            torch.bfloat16,
            4,
            2,
            128,
            16,
            128**-0.5,
            1,
            1,
            8,
        )
        is None
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
def test_segmented_prefill_ragged_unified_graph_replay(
    monkeypatch, dim, page, hq, hk, dtype, kv_dtype, byte_cache, force_splits
):
    """Unified-cache segmented prefill follows metadata on graph replay."""
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

    if not current_platform.is_rocm() or not (
        on_gfx12x() if kv_dtype.itemsize == 1 else on_gfx1x()
    ):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    if force_splits is not None:
        original = dispatcher.select_segmented_config
        monkeypatch.setattr(
            dispatcher,
            "select_segmented_config",
            lambda *args: dict(original(*args), splits=force_splits),
        )
    query_lengths = [0, 1, 2, 7, 33, 129]
    contexts = [0, 67, 0, page - 1, page + 1, 33]
    device = torch.device("cuda:0")
    total = sum(query_lengths)
    q = torch.randn(total, hq + 1, dim, device=device, dtype=dtype)[:, :hq]
    k = torch.randn(total, hk + 1, dim, device=device, dtype=dtype)[:, :hk]
    v = torch.randn_like(k)
    blocks_per_seq = triton.cdiv(max(map(sum, zip(query_lengths, contexts))), page)
    num_blocks = len(contexts) * blocks_per_seq
    page_elements = page * hk * dim
    backing = torch.empty(num_blocks * 2 * page_elements, device=device, dtype=kv_dtype)
    cache_shape = (num_blocks, page, hk, dim)
    strides = (2 * page_elements, hk * dim, dim, 1)
    kc = torch.as_strided(backing, cache_shape, strides)
    vc = torch.as_strided(backing, cache_shape, strides, page_elements)
    ks = torch.tensor(0.13 if kv_dtype.itemsize == 1 else 1.0, device=device)
    vs = torch.tensor(0.27 if kv_dtype.itemsize == 1 else 1.0, device=device)
    kc.copy_((torch.randn(cache_shape, device=device).float() * 0.25 / ks).to(kv_dtype))
    vc.copy_((torch.randn(cache_shape, device=device).float() * 0.25 / vs).to(kv_dtype))
    table = torch.randperm(num_blocks, device=device, dtype=torch.int32).view(
        len(contexts), blocks_per_seq
    )
    host_table = table.cpu().tolist()
    starts = torch.empty(len(contexts) + 1, device=device, dtype=torch.int32)
    lengths = torch.empty(len(contexts), device=device, dtype=torch.int32)
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
        first = 0
        for seq, (nq, context) in enumerate(zip(qlens, contexts)):
            for local in range(nq):
                position = context + local
                physical = host_table[seq][position // page]
                offset = position % page
                kc[physical, offset].copy_((k[first + local].float() / ks).to(kv_dtype))
                vc[physical, offset].copy_((v[first + local].float() / vs).to(kv_dtype))
            first += nq

    def run():
        dispatcher.segmented_attention(
            query=q,
            key=k,
            value=v,
            output=output,
            kv_cache_dtype="fp8" if kv_dtype.itemsize == 1 else "auto",
            key_cache=kc.view(torch.uint8) if byte_cache else kc,
            value_cache=vc.view(torch.uint8) if byte_cache else vc,
            block_table=table,
            query_start_loc=starts,
            seq_lens=lengths,
            max_seq_len=max(contexts) + 129,
            max_query_len=129,
            k_scale=ks,
            v_scale=vs,
            sm_scale=dim**-0.5,
        )

    def check(qlens):
        first = 0
        for seq, (nq, context) in enumerate(zip(qlens, contexts)):
            if nq == 0:
                first += nq
                continue
            physical = table[seq].long()
            full_k = (
                kc[physical].reshape(-1, hk, dim)[: context + nq].float() * ks
            ).repeat_interleave(hq // hk, 1)
            full_v = (
                vc[physical].reshape(-1, hk, dim)[: context + nq].float() * vs
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
    torch.accelerator.synchronize()
    check(query_lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    replay_lengths = [1, 0, 7, 2, 34, 128]
    q.mul_(0.75)
    k.add_(0.25)
    v.add_(0.5)
    metadata(replay_lengths)
    output.fill_(17.0)
    graph.replay()
    torch.accelerator.synchronize()
    check(replay_lengths)


@pytest.mark.parametrize(
    "fp8,force_splits",
    [(False, None), (True, None), (False, 4), (True, 4)],
)
@torch.inference_mode()
def test_segmented_backend_routes_unified_cache_to_segmented(
    monkeypatch, fp8, force_splits
):
    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.ops import segmented_attention as dispatcher
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

        def tuned_config(*args):
            config = segmented.select_segmented_config(
                args[8],
                args[9],
                args[10],
                args[3],
                args[4],
                args[5],
                args[2].itemsize == 1,
            )
            return dict(config, splits=force_splits)

        monkeypatch.setattr(dispatcher, "get_segmented_config", tuned_config)

    def segmented_spy(*args, **kwargs):
        routed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "segmented_prefill_attention", segmented_spy)
    dispatcher.segmented_attention(
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
        sm_scale=256**-0.5,
    )

    assert routed == [True]
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize("fp8", [False, True])
@torch.inference_mode()
def test_rocm_segmented_attn_layout_cache_update(fp8):
    from types import SimpleNamespace

    from vllm.platforms.rocm import on_gfx1x, on_gfx12x
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_segmented_attn import (
        RocmSegmentedAttentionImpl,
    )

    if not current_platform.is_rocm() or not (on_gfx12x() if fp8 else on_gfx1x()):
        pytest.skip("gfx1x segmented prefill (FP8 requires gfx12)")
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    kv_cache_dtype = "fp8" if fp8 else "auto"
    impl = RocmSegmentedAttentionImpl(
        12,
        256,
        256**-0.5,
        2,
        None,
        None,
        kv_cache_dtype,
        attn_type=AttentionType.DECODER,
    )
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
def test_rocm_segmented_attn_arch_gate(
    monkeypatch, kv_cache_dtype, is_gfx1x, is_gfx12x, expected
):
    from vllm.platforms import rocm
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.rocm_segmented_attn import (
        RocmSegmentedAttentionImpl,
    )

    monkeypatch.setattr(rocm, "on_gfx1x", lambda: is_gfx1x)
    monkeypatch.setattr(rocm, "on_gfx12x", lambda: is_gfx12x)
    args = (
        12,
        256,
        256**-0.5,
        2,
        None,
        None,
        kv_cache_dtype,
    )
    if expected:
        assert isinstance(
            RocmSegmentedAttentionImpl(*args, attn_type=AttentionType.DECODER),
            RocmSegmentedAttentionImpl,
        )
    else:
        with pytest.raises(ValueError, match="ROCM_SEGMENTED_ATTN requires"):
            RocmSegmentedAttentionImpl(*args, attn_type=AttentionType.DECODER)


@pytest.mark.parametrize(
    "query_lens,context_lens,max_query_len,causal",
    [([129, 1], [31, 64], 4097, True), ([33, 1], [31, 64], 33, False)],
)
@torch.inference_mode()
def test_segmented_backend_routes_unsupported_patterns_to_unified(
    monkeypatch, query_lens, context_lens, max_query_len, causal
):
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

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
    original_unified = dispatcher.unified_attention

    def unified_spy(*args, **kwargs):
        routed.append("unified")
        return original_unified(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "unified_attention", unified_spy)
    monkeypatch.setattr(
        dispatcher,
        "segmented_prefill_attention",
        lambda *args, **kwargs: pytest.fail(
            "unsupported pattern routed to segmented prefill"
        ),
    )
    dispatcher.segmented_attention(
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
        sm_scale=128**-0.5,
        causal=causal,
    )

    assert routed == ["unified"]
    relative = (output.float() - case["reference"]).norm(dim=-1) / case[
        "reference"
    ].norm(dim=-1).clamp_min(1e-6)
    assert torch.isfinite(output).all() and relative.max().item() < 0.01


@pytest.mark.parametrize(
    "feature",
    [
        "sliding_window",
        "sinks",
        "softcap",
        "sliding_softcap",
        "sliding_sinks",
        "output_scale",
    ],
)
@torch.inference_mode()
def test_segmented_backend_unified_feature_fallback_accuracy(monkeypatch, feature):
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops import segmented_attention as dispatcher

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x unified attention feature fallback")

    num_heads = 8
    configured_sliding_window = 16 if feature.startswith("sliding") else 0
    sliding_window = configured_sliding_window - 1 if configured_sliding_window else 0
    softcap = 5.0 if "softcap" in feature else 0.0
    sinks = None
    if "sinks" in feature:
        sinks = torch.linspace(-0.5, 0.5, num_heads, device="cuda:0")
    case = _make_unified_paged_case(
        [7, 2],
        [40, 35],
        num_heads=num_heads,
        num_kv_heads=2,
        head_size=128,
        block_size=32,
        fp8=False,
        sliding_window=configured_sliding_window,
        softcap=softcap,
        sinks=sinks,
    )
    output_scale = None
    if feature == "output_scale":
        output_scale = torch.tensor(0.5, device="cuda:0")
        output = torch.empty_like(case["query"], dtype=current_platform.fp8_dtype())
    else:
        output = torch.empty_like(case["query"])

    routed = []
    log_messages = []
    original_unified = dispatcher.unified_attention

    def unified_spy(*args, **kwargs):
        routed.append("unified")
        return original_unified(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "unified_attention", unified_spy)
    monkeypatch.setattr(
        dispatcher,
        "segmented_prefill_attention",
        lambda *args, **kwargs: pytest.fail(
            "feature fallback routed to segmented prefill"
        ),
    )
    monkeypatch.setattr(
        dispatcher.logger,
        "info_once",
        lambda message, *args: log_messages.append(message),
    )

    dispatcher.segmented_attention(
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
        max_query_len=7,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        sm_scale=128**-0.5,
        sliding_window=sliding_window,
        softcap=softcap,
        output_scale=output_scale,
        sinks=sinks,
    )

    assert routed == ["unified"]
    if configured_sliding_window:
        expected_log = (
            "ROCM_SEGMENTED_ATTN is routing sliding-window attention to the "
            "unified Triton attention fallback."
        )
        assert log_messages == [expected_log]
    else:
        assert not log_messages

    actual = output.float()
    if output_scale is not None:
        actual *= output_scale
        torch.testing.assert_close(actual, case["reference"], atol=0.2, rtol=0.2)
    else:
        relative = (actual - case["reference"]).norm(dim=-1) / case["reference"].norm(
            dim=-1
        ).clamp_min(1e-6)
        assert torch.isfinite(output).all() and relative.max().item() < 0.01


@torch.inference_mode()
def test_segmented_prefill_cache_offsets_cross_int32_boundary():
    """A small logical prefix can live beyond 2**31 elements in a strided cache."""
    from vllm.platforms.rocm import on_gfx1x
    from vllm.v1.attention.ops.segmented_prefill import segmented_prefill_attention

    if not current_platform.is_rocm() or not on_gfx1x():
        pytest.skip("gfx1x segmented prefill")
    torch.accelerator.empty_cache()
    free, _ = torch.accelerator.get_memory_info()
    if free < 10 * 2**30:
        pytest.skip("The address-boundary fixture needs 10 GiB free VRAM")
    device = torch.device("cuda:0")
    stride = 2**30 + 4096
    kc = torch.empty_strided(
        (3, 32, 1, 128),
        (stride, 128, 128, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    vc = torch.empty_strided(
        (3, 32, 1, 128),
        (stride, 128, 128, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_k = torch.randn(2, 32, 1, 128, device=device, dtype=torch.bfloat16)
    dense_v = torch.randn_like(dense_k)
    for logical, physical in enumerate((2, 1)):
        kc[physical].copy_(dense_k[logical])
        vc[physical].copy_(dense_v[logical])
    q = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    table = torch.tensor([[2, 1]], device=device, dtype=torch.int32)
    starts = torch.tensor([0, 2], device=device, dtype=torch.int32)
    lengths = torch.tensor([35], device=device, dtype=torch.int32)
    one = torch.ones((), device=device)
    segmented_prefill_attention(
        q, out, kc, vc, table, starts, lengths, 2, 35, one, one, 128**-0.5
    )
    full_k = dense_k.flatten(0, 1)[:35].float().repeat_interleave(4, 1)
    full_v = dense_v.flatten(0, 1)[:35].float().repeat_interleave(4, 1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), full_k) / math.sqrt(128)
    scores[:, 0, -1] = -float("inf")
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), full_v)
    error = ((out.float() - expected).norm(dim=-1) / expected.norm(dim=-1)).max()
    assert torch.isfinite(out).all() and error < 0.01
