# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm Triton SplitKV decode fallback.

The standard page-16/page-32 rows are direct Triton diagnostics. Production
dispatch still gives the native ROCm paged-attention kernel first refusal.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _choose_fallback_block_size,
    _get_num_splits,
    _paged_attention_2d_splitkv_decode,
    kernel_paged_attention_2d,
)


@dataclass(frozen=True)
class Case:
    name: str
    query_dtype: torch.dtype
    kv_dtype: torch.dtype
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    page_size: int
    seq_lens: tuple[int, ...]
    padded_stride: bool = False


def _cases() -> tuple[list[Case], list[Case]]:
    fp8 = current_platform.fp8_dtype()
    general = [
        Case("bf16-mha-d128", torch.bfloat16, torch.bfloat16, 8, 8, 128, 16, (4096,)),
        Case(
            "bf16-gqa4-d128",
            torch.bfloat16,
            torch.bfloat16,
            16,
            4,
            128,
            32,
            (8192,),
        ),
        Case(
            "bf16-mqa-d128-padded",
            torch.bfloat16,
            torch.bfloat16,
            8,
            1,
            128,
            544,
            (8192, 4097, 2049, 513),
            True,
        ),
        Case(
            "bf16-gqa6-d256-padded",
            torch.bfloat16,
            torch.bfloat16,
            12,
            2,
            256,
            1568,
            (8192,),
            True,
        ),
        Case(
            "bf16-gqa4-crossover",
            torch.bfloat16,
            torch.bfloat16,
            16,
            4,
            128,
            32,
            (4096,) * 16,
        ),
        Case("fp8-mha-d128", torch.bfloat16, fp8, 8, 8, 128, 16, (8192,)),
        Case("fp8-gqa4-d128", torch.bfloat16, fp8, 16, 4, 128, 32, (8192,)),
        Case(
            "fp8-mqa-d128-padded",
            torch.float16,
            fp8,
            8,
            1,
            128,
            528,
            (8192, 4097, 2049, 513),
            True,
        ),
        Case(
            "fp8-gqa6-d256-padded",
            torch.bfloat16,
            fp8,
            12,
            2,
            256,
            1568,
            (8192,),
            True,
        ),
        Case(
            "fp8-gqa16-d256-padded",
            torch.bfloat16,
            fp8,
            16,
            1,
            256,
            1056,
            (32768,),
            True,
        ),
        Case(
            "fp8-gqa4-crossover",
            torch.bfloat16,
            fp8,
            16,
            4,
            128,
            32,
            (4096,) * 16,
        ),
    ]
    qwen_lengths = (180, 1374, 4014, 8192, 32768, 131072)
    qwen = [
        Case(
            f"qwen-b1-s{seq_len}",
            torch.bfloat16,
            fp8,
            12,
            2,
            256,
            1568,
            (seq_len,),
            True,
        )
        for seq_len in qwen_lengths
    ]
    qwen.extend(
        [
            Case(
                "qwen-ragged-b3",
                torch.bfloat16,
                fp8,
                12,
                2,
                256,
                1568,
                (32768, 16384, 8192),
                True,
            ),
            Case(
                "qwen-ragged-b13",
                torch.bfloat16,
                fp8,
                12,
                2,
                256,
                1568,
                tuple(16384 - 1024 * index for index in range(13)),
                True,
            ),
        ]
    )
    return general, qwen


def _padded_cache_views(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = key_cache.shape[0]
    page_elements = key_cache[0].numel()
    backing = torch.empty(
        num_blocks * 2 * page_elements,
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    padded_key = torch.as_strided(
        backing,
        key_cache.shape,
        (2 * page_elements, *key_cache.stride()[1:]),
    )
    padded_value = torch.as_strided(
        backing,
        value_cache.shape,
        (2 * page_elements, *value_cache.stride()[1:]),
        page_elements,
    )
    padded_key.copy_(key_cache)
    padded_value.copy_(value_cache)
    return padded_key, padded_value


def _make_inputs(case: Case):
    device = current_platform.device_type
    blocks_per_seq = [math.ceil(length / case.page_size) for length in case.seq_lens]
    num_blocks = sum(blocks_per_seq)
    x = 16 // case.kv_dtype.itemsize
    query = torch.randn(
        len(case.seq_lens),
        case.num_query_heads,
        case.head_size,
        dtype=case.query_dtype,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        case.num_kv_heads,
        case.head_size // x,
        case.page_size,
        x,
        dtype=torch.bfloat16,
        device=device,
    ).to(case.kv_dtype)
    value_cache = torch.randn(
        num_blocks,
        case.num_kv_heads,
        case.head_size,
        case.page_size,
        dtype=torch.bfloat16,
        device=device,
    ).to(case.kv_dtype)
    if case.padded_stride:
        key_cache, value_cache = _padded_cache_views(key_cache, value_cache)

    block_tables = torch.zeros(
        len(case.seq_lens), max(blocks_per_seq), dtype=torch.int32, device=device
    )
    permutation = torch.randperm(num_blocks, dtype=torch.int32, device=device)
    cursor = 0
    for row, num_seq_blocks in enumerate(blocks_per_seq):
        block_tables[row, :num_seq_blocks] = permutation[
            cursor : cursor + num_seq_blocks
        ]
        cursor += num_seq_blocks
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=device)
    k_scale = torch.tensor(0.73, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.27, dtype=torch.float32, device=device)
    return query, key_cache, value_cache, block_tables, seq_lens, k_scale, v_scale


def _run_non_split(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    k_scale,
    v_scale,
    output,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    page_size = key_cache.shape[3]
    queries_per_kv = num_query_heads // num_kv_heads
    kernel_paged_attention_2d[(seq_lens.shape[0], num_kv_heads)](
        output,
        query,
        key_cache,
        value_cache,
        None,
        block_tables,
        seq_lens,
        None,
        head_size**-0.5,
        k_scale,
        v_scale,
        1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(queries_per_kv), 16),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=_choose_fallback_block_size(page_size),
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
        filter_by_query_len=False,
        query_start_len_ptr=None,
        USE_SINKS=False,
        USE_FP8=False,
    )


def _capture(fn):
    fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


@torch.inference_mode()
def benchmark_case(case: Case, warmup: int, rep: int, graph: bool) -> None:
    query, key, value, tables, lens, k_scale, v_scale = _make_inputs(case)
    max_seq_len = max(case.seq_lens)
    splits = _get_num_splits(
        len(case.seq_lens),
        case.num_kv_heads,
        case.head_size,
        case.page_size,
        max_seq_len,
        allow_short_context=case.kv_dtype.itemsize == 1,
    )
    baseline_output = torch.empty_like(query)
    split_output = torch.empty_like(query)
    mid_out = torch.empty(
        len(case.seq_lens),
        case.num_query_heads,
        splits,
        case.head_size,
        dtype=torch.float32,
        device=query.device,
    )
    mid_lse = torch.empty(
        len(case.seq_lens),
        case.num_query_heads,
        splits,
        dtype=torch.float32,
        device=query.device,
    )

    def baseline() -> None:
        _run_non_split(
            query,
            key,
            value,
            tables,
            lens,
            k_scale,
            v_scale,
            baseline_output,
        )

    def splitkv() -> None:
        _paged_attention_2d_splitkv_decode(
            query,
            key,
            value,
            tables,
            lens,
            case.head_size**-0.5,
            k_scale,
            v_scale,
            output=split_output,
            actual_max_splits=splits,
            max_seq_len=max_seq_len,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    baseline()
    splitkv()
    torch.accelerator.synchronize()
    torch.testing.assert_close(split_output, baseline_output, atol=0.02, rtol=0.02)
    baseline_fn = _capture(baseline) if graph else baseline
    splitkv_fn = _capture(splitkv) if graph else splitkv
    baseline_ms = triton.testing.do_bench(
        baseline_fn, warmup=warmup, rep=rep, return_mode="median"
    )
    splitkv_ms = triton.testing.do_bench(
        splitkv_fn, warmup=warmup, rep=rep, return_mode="median"
    )
    scratch_bytes = mid_out.nbytes + mid_lse.nbytes
    print(
        f"| {case.name} | {splits} | {scratch_bytes / 2**20:.2f} | "
        f"{baseline_ms * 1000:.2f} | {splitkv_ms * 1000:.2f} | "
        f"{baseline_ms / splitkv_ms:.2f}x |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("general", "qwen", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup time in ms")
    parser.add_argument("--rep", type=int, default=100, help="Measurement time in ms")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not current_platform.is_rocm():
        raise RuntimeError("This benchmark requires ROCm.")

    general, qwen = _cases()
    cases = general if args.suite == "general" else qwen
    if args.suite == "all":
        cases = general + qwen
    print(f"GPU: {torch.cuda.get_device_name()} | graph={args.graph}")
    print("| case | splits | scratch MiB | 2D us | SplitKV us | speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for case in cases:
        benchmark_case(case, args.warmup, args.rep, args.graph)


if __name__ == "__main__":
    main()
