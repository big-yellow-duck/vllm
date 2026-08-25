# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1x, on_gfx12x
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import chunked_prefill_paged_decode as paged_decode_ops
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _choose_fallback_block_size,
    _paged_attention_2d_splitkv_decode,
    kernel_paged_attention_2d,
    reserve_splitkv_workspace,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)

DEVICE = current_platform.device_type


@dataclass(frozen=True)
class SplitKVCase:
    query_dtype: torch.dtype
    kv_dtype: torch.dtype
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    page_size: int
    seq_lens: tuple[int, ...]
    splits: int | None
    k_scale: float = 1.0
    v_scale: float = 1.0
    check_torch_reference: bool = False
    padded_stride: bool = False


CASES = [
    pytest.param(
        SplitKVCase(torch.bfloat16, torch.bfloat16, 4, 4, 128, 16, (257,), 4),
        id="native-bf16-mha",
    ),
    pytest.param(
        SplitKVCase(
            torch.float16,
            torch.float16,
            16,
            4,
            128,
            32,
            (257, 513, 1025),
            4,
        ),
        id="native-fp16-gqa4",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.bfloat16,
            12,
            2,
            256,
            1568,
            (1567, 1568, 1569),
            7,
            check_torch_reference=True,
            padded_stride=True,
        ),
        id="native-qwen-page-boundary",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (4014,),
            1,
            0.73,
            1.27,
            True,
            True,
        ),
        id="fp8-qwen-one-split",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (8192,),
            14,
            0.73,
            1.27,
            padded_stride=True,
        ),
        id="fp8-qwen-long",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            12,
            2,
            256,
            1568,
            (32768, 16384, 8192),
            14,
            0.73,
            1.27,
            padded_stride=True,
        ),
        id="fp8-qwen-ragged",
    ),
    pytest.param(
        SplitKVCase(
            torch.float16,
            torch.float8_e4m3fn,
            8,
            1,
            128,
            528,
            (527, 528, 529),
            7,
            0.5,
            1.5,
        ),
        id="fp8-fp16-mqa-odd-page",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            16,
            4,
            128,
            784,
            (783, 784, 785),
            16,
            0.75,
            1.25,
        ),
        id="fp8-empty-splits",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            5,
            1,
            256,
            1056,
            (1055, 1056, 1057),
            4,
            0.75,
            1.25,
        ),
        id="fp8-odd-head-group",
    ),
    pytest.param(
        SplitKVCase(
            torch.bfloat16,
            torch.float8_e4m3fn,
            4,
            4,
            128,
            32,
            (2049, 4097, 8193),
            None,
            0.75,
            1.25,
        ),
        id="fp8-production-heuristic",
    ),
]


def _pack_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    padded_stride: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, page_size, num_kv_heads, head_size = key_cache.shape
    x = 16 // key_cache.element_size()
    packed_key = (
        key_cache.view(num_blocks, page_size, num_kv_heads, head_size // x, x)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    packed_value = value_cache.permute(0, 2, 3, 1).contiguous()
    if not padded_stride:
        return packed_key, packed_value

    page_elements = packed_key[0].numel()
    backing = torch.empty(
        num_blocks * 2 * page_elements,
        dtype=packed_key.dtype,
        device=packed_key.device,
    )
    key_cache = torch.as_strided(
        backing,
        size=packed_key.shape,
        stride=(2 * page_elements, *packed_key.stride()[1:]),
    )
    value_cache = torch.as_strided(
        backing,
        size=packed_value.shape,
        stride=(2 * page_elements, *packed_value.stride()[1:]),
        storage_offset=page_elements,
    )
    key_cache.copy_(packed_key)
    value_cache.copy_(packed_value)
    return key_cache, value_cache


def _make_inputs(case: SplitKVCase):
    batch_size = len(case.seq_lens)
    blocks_per_seq = [
        (seq_len + case.page_size - 1) // case.page_size for seq_len in case.seq_lens
    ]
    num_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    query = torch.randn(
        batch_size,
        case.num_query_heads,
        case.head_size,
        dtype=case.query_dtype,
        device=DEVICE,
    )
    dense_key = torch.randn(
        num_blocks,
        case.page_size,
        case.num_kv_heads,
        case.head_size,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(case.kv_dtype)
    dense_value = torch.randn(
        num_blocks,
        case.page_size,
        case.num_kv_heads,
        case.head_size,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(case.kv_dtype)

    permutation = torch.randperm(num_blocks, device=DEVICE, dtype=torch.int64)
    block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=DEVICE)
    cursor = 0
    for seq_idx, (seq_len, num_seq_blocks) in enumerate(
        zip(case.seq_lens, blocks_per_seq)
    ):
        physical_blocks = permutation[cursor : cursor + num_seq_blocks]
        block_tables[seq_idx, :num_seq_blocks] = physical_blocks.to(torch.int32)
        cursor += num_seq_blocks
        tail = seq_len % case.page_size
        if tail:
            last_block = physical_blocks[-1]
            dense_key[last_block, tail:] = float("nan")
            dense_value[last_block, tail:] = float("nan")

    key_cache, value_cache = _pack_cache(
        dense_key, dense_value, padded_stride=case.padded_stride
    )
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=DEVICE)
    k_scale = torch.tensor(case.k_scale, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(case.v_scale, dtype=torch.float32, device=DEVICE)
    return (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    )


def _run_non_split(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
) -> torch.Tensor:
    if output is None:
        output = torch.empty_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    page_size = key_cache.shape[3]
    block_size = _choose_fallback_block_size(page_size)
    num_queries_per_kv = num_query_heads // num_kv_heads

    kernel_paged_attention_2d[(seq_lens.shape[0], num_kv_heads)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out_scale_inv=1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(num_queries_per_kv), 16),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=page_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
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
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        USE_SINKS=False,
        USE_FP8=False,
    )
    return output


def _torch_reference(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    outputs = []
    page_size = dense_key.shape[1]
    num_kv_heads = dense_key.shape[2]
    repeat = query.shape[1] // num_kv_heads
    for seq_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor.item())
        num_blocks = (seq_len + page_size - 1) // page_size
        block_ids = block_tables[seq_idx, :num_blocks].long()
        key = dense_key[block_ids].reshape(-1, num_kv_heads, query.shape[2])
        value = dense_value[block_ids].reshape(-1, num_kv_heads, query.shape[2])
        key = key[:seq_len].float() * k_scale
        value = value[:seq_len].float() * v_scale
        key = torch.repeat_interleave(key, repeat, dim=1)
        value = torch.repeat_interleave(value, repeat, dim=1)
        scores = torch.einsum("hd,shd->hs", query[seq_idx].float(), key) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hs,shd->hd", probabilities, value))
    return torch.stack(outputs).to(query.dtype)


@pytest.mark.skipif(not on_gfx1x(), reason="SplitKV decode requires gfx1x")
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="SplitKV decode requires a GPU"
)
@pytest.mark.parametrize("case", CASES)
@torch.inference_mode()
def test_paged_attention_2d_splitkv_decode(case: SplitKVCase) -> None:
    if case.kv_dtype.itemsize == 1 and not on_gfx12x():
        pytest.skip("FP8 SplitKV decode requires gfx12x")
    set_random_seed(0)
    (
        query,
        dense_key,
        dense_value,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    scale = case.head_size**-0.5

    output = _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        actual_max_splits=case.splits,
        max_seq_len=max(case.seq_lens),
    )
    reference = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
    )

    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    atol = get_default_atol(output)
    rtol = get_default_rtol(output)
    if case.kv_dtype.itemsize == 1:
        atol = max(atol, 0.01)
        rtol = max(rtol, 0.01)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)

    if case.check_torch_reference:
        torch_reference = _torch_reference(
            query,
            dense_key,
            dense_value,
            block_tables,
            seq_lens,
            scale,
            case.k_scale if case.kv_dtype.itemsize == 1 else 1.0,
            case.v_scale if case.kv_dtype.itemsize == 1 else 1.0,
        )
        torch.testing.assert_close(output, torch_reference, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ((1, 3, 0, 1), (4014, 8192, 1568, 32768)),
        ((2, 4), (1568, 4014)),
    ],
)
@torch.inference_mode()
def test_fp8_splitkv_preserves_non_decode_rows(
    query_lens: tuple[int, ...], seq_lens: tuple[int, ...]
) -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        seq_lens,
        14,
        0.73,
        1.27,
    )
    (
        _,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query = torch.randn(sum(query_lens), 12, 256, dtype=torch.bfloat16, device=DEVICE)
    query_start_loc = torch.tensor(
        [0] + [sum(query_lens[: index + 1]) for index in range(len(query_lens))],
        dtype=torch.int32,
        device=DEVICE,
    )
    output = torch.full_like(query, 7.0)
    reference = output.clone()
    scale = 256**-0.5

    _paged_attention_2d_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        scale,
        k_scale,
        v_scale,
        output=output,
        actual_max_splits=14,
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        scale,
        k_scale,
        v_scale,
        output=reference,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    decode_rows = [
        int(query_start_loc[index].item())
        for index, query_len in enumerate(query_lens)
        if query_len == 1
    ]
    if decode_rows:
        torch.testing.assert_close(
            output[decode_rows], reference[decode_rows], atol=0.01, rtol=0.01
        )
    non_decode = torch.ones(sum(query_lens), dtype=torch.bool, device=DEVICE)
    non_decode[decode_rows] = False
    torch.testing.assert_close(
        output[non_decode], torch.full_like(output[non_decode], 7)
    )


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@torch.inference_mode()
def test_chunked_decode_routes_padded_fp8_cache_and_forwards_scales(
    monkeypatch,
) -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (1374,),
        None,
        0.73,
        1.27,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    expected = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        256**-0.5,
        k_scale,
        v_scale,
    )
    key_storage = key_cache.view(torch.uint8)
    value_storage = value_cache.view(torch.uint8)
    original = paged_decode_ops._paged_attention_2d_splitkv_decode
    forwarded = {}

    def route_spy(*args, **kwargs):
        forwarded["k_scale"] = kwargs["k_scale"]
        forwarded["v_scale"] = kwargs["v_scale"]
        forwarded["cache_dtype"] = kwargs["key_cache"].dtype
        return original(*args, **kwargs)

    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", route_spy
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="fp8",
        key_cache=key_storage,
        value_cache=value_storage,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=1374,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert forwarded == {
        "k_scale": k_scale,
        "v_scale": v_scale,
        "cache_dtype": current_platform.fp8_dtype(),
    }
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx1x(), reason="BF16 SplitKV decode requires gfx1x")
@torch.inference_mode()
def test_chunked_decode_routes_padded_bf16_cache(monkeypatch) -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.bfloat16,
        12,
        2,
        256,
        1568,
        (8192,),
        None,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    expected = _run_non_split(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        256**-0.5,
        k_scale,
        v_scale,
    )
    original = paged_decode_ops._paged_attention_2d_splitkv_decode
    routed = []

    def route_spy(*args, **kwargs):
        routed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", route_spy
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=8192,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert routed == [True]
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not on_gfx1x(), reason="Native ROCm decode requires gfx1x")
@torch.inference_mode()
def test_native_paged_attention_precedes_splitkv_for_standard_layout(
    monkeypatch,
) -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.bfloat16,
        16,
        4,
        128,
        32,
        (257,),
        None,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    output = torch.empty_like(query)
    native_called = []

    def native_spy(output, *args, **kwargs) -> None:
        native_called.append(True)
        output.fill_(3)

    def unexpected_splitkv(*args, **kwargs):
        raise AssertionError("SplitKV must not precede the native ROCm kernel")

    monkeypatch.setattr(
        "vllm.platforms.rocm.use_rocm_custom_paged_attention",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(paged_decode_ops.ops, "paged_attention_rocm", native_spy)
    monkeypatch.setattr(
        paged_decode_ops, "_paged_attention_2d_splitkv_decode", unexpected_splitkv
    )
    paged_decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=257,
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert native_called == [True]
    torch.testing.assert_close(output, torch.full_like(output, 3))


@pytest.mark.skipif(not on_gfx12x(), reason="FP8 SplitKV decode requires gfx12x")
@torch.inference_mode()
def test_fp8_splitkv_locked_workspace_cudagraph_replays_dynamic_sequence() -> None:
    set_random_seed(0)
    case = SplitKVCase(
        torch.bfloat16,
        torch.float8_e4m3fn,
        12,
        2,
        256,
        1568,
        (8192,),
        16,
        0.73,
        1.27,
        padded_stride=True,
    )
    (
        query,
        _,
        _,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
    ) = _make_inputs(case)
    output = torch.empty_like(query)

    reset_workspace_manager()
    init_workspace_manager(query.device)
    reserve_splitkv_workspace(
        max_batch_size=1,
        num_query_heads=12,
        num_kv_heads=2,
        head_size=256,
        physical_block_size=1568,
        max_seq_len=8192,
        allow_short_context=True,
    )
    lock_workspace()
    seq_lens.fill_(1)

    def run_splitkv() -> None:
        _paged_attention_2d_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            256**-0.5,
            k_scale,
            v_scale,
            output=output,
            actual_max_splits=16,
            max_seq_len=8192,
        )

    graph = None
    try:
        run_splitkv()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_splitkv()

        query.copy_(torch.randn_like(query))
        seq_lens.fill_(8192)
        graph.replay()
        torch.accelerator.synchronize()
        expected = _run_non_split(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            256**-0.5,
            k_scale,
            v_scale,
        )
        torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
    finally:
        del graph
        reset_workspace_manager()
