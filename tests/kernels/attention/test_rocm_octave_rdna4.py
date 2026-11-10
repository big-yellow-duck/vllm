# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDNA4 Octave HIP kernels against the existing PyTorch format reference."""

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.attention.ops.rocm_octave_rdna4 import TunedHipOctave


def _rdna4():
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_rdna4

    return on_rdna4()


pytestmark = pytest.mark.skipif(not _rdna4(), reason="Octave kernels need RDNA4")

BLOCK, KV_HEADS = 16, 2
LENGTHS = [1, 37, 257]
CACHE_DTYPES = ["octave_k3v3", "octave_k3v4", "octave_k3v3_compact"]


@pytest.fixture(
    scope="module",
    params=[
        (64, 0, 8),
        (80, 32, 3),
        (128, 128, 1),
        (256, 64, 6),
        (512, 64, 16),
        (1024, 128, 2),
    ],
)
def engine(request):
    head, rope, group = request.param
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return TunedHipOctave.build(head, rope, arch=arch), group


def _inputs(hip, cache_dtype, dtype, token_major):
    torch.manual_seed(1171)
    layout = hip.layout(cache_dtype)
    pages = sum(math.ceil(n / BLOCK) for n in LENGTHS)
    shape = (pages + 2, KV_HEADS, BLOCK, layout.slot_bytes)
    if token_major:
        shape = (pages + 2, BLOCK, KV_HEADS, layout.slot_bytes)
    cache = torch.full(shape, 239, dtype=torch.uint8, device="cuda")
    if token_major:
        cache = cache.transpose(1, 2)
    table = torch.full(
        (len(LENGTHS), math.ceil(max(LENGTHS) / BLOCK) + 1),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    physical = torch.randperm(pages, device="cuda") + 1
    used, slots = 0, []
    for req, length in enumerate(LENGTHS):
        count = math.ceil(length / BLOCK)
        table[req, :count] = physical[used : used + count]
        used += count
        pos = torch.arange(length, device="cuda")
        slots.append(table[req, pos // BLOCK].long() * BLOCK + pos % BLOCK)
    key = torch.randn(sum(LENGTHS), KV_HEADS, hip.head_dim, device="cuda").to(dtype)
    value = torch.randn_like(key)
    return cache, table, torch.cat(slots), key, value, layout.sign_bits().to("cuda")


def _check_close(actual, expected, tolerance):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error.item() < tolerance


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
def test_store_matches_reference_and_skips_padding(
    engine, cache_dtype, dtype, token_major
):
    hip, _ = engine
    layout = hip.layout(cache_dtype)
    cache, _, slots, key, value, signs = _inputs(hip, cache_dtype, dtype, token_major)
    before = cache.clone()
    slots[3] = -1
    hip.store(key, value, cache, slots, signs, cache_dtype)
    live = slots >= 0
    packed = cache[slots[live] // BLOCK, :, slots[live] % BLOCK]
    fmt = layout.reference_format()
    actual = sq.reference_dequantize(packed, fmt)
    expected = sq.reference_dequantize(
        sq.reference_quantize(
            layout.to_storage(key[live]), layout.to_storage(value[live]), fmt
        ),
        fmt,
    )
    for a, b in zip(actual, expected):
        _check_close(a, b, 1e-2)
    written = torch.zeros(cache.shape[:3], dtype=torch.bool, device="cuda")
    written[slots[live] // BLOCK, :, slots[live] % BLOCK] = True
    assert torch.equal(cache[~written], before[~written])


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("splits", [1, 7, 64])
def test_decode_matches_reference_and_replays(
    engine, cache_dtype, dtype, token_major, splits
):
    hip, group = engine
    layout = hip.layout(cache_dtype)
    cache, table, slots, key, value, signs = _inputs(
        hip, cache_dtype, dtype, token_major
    )
    fmt = layout.reference_format()
    cache[slots // BLOCK, :, slots % BLOCK] = sq.reference_quantize(
        layout.to_storage(key), layout.to_storage(value), fmt
    )
    mapping = torch.tensor([0, 1, 1, 2, 2, 2, 999], device="cuda", dtype=torch.int32)
    lengths = torch.tensor(
        [1, 1, 37, 253, 255, 257, 0], device="cuda", dtype=torch.int32
    )
    query = (torch.randn(7, KV_HEADS * group, hip.head_dim, device="cuda") * 2).to(
        dtype
    )
    output = torch.empty_like(query, dtype=torch.float32)
    scratch = torch.empty(
        *query.shape[:2], splits, hip.head_dim + 2, device="cuda", dtype=torch.float32
    )

    def call():
        return hip.decode(
            query,
            cache,
            table,
            mapping,
            lengths,
            signs,
            cache_dtype,
            splits=splits,
            out=output,
            scratch=scratch,
        )

    def reference():
        return layout.from_storage(
            sq.reference_attention(
                layout.to_storage(query),
                cache,
                table,
                mapping,
                lengths,
                fmt,
                hip.head_dim**-0.5,
            )
        )

    call()
    _check_close(output, reference(), 2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    query.neg_()
    output.fill_(float("nan"))
    graph.replay()
    _check_close(output, reference(), 2e-2)
    assert torch.count_nonzero(output[-1]).item() == 0


@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("splits", [1, 8])
def test_prefill_matches_reference(engine, cache_dtype, dtype, token_major, splits):
    hip, group = engine
    layout = hip.layout(cache_dtype)
    cache, table, slots, key, value, signs = _inputs(
        hip, cache_dtype, dtype, token_major
    )
    fmt = layout.reference_format()
    chunks, starts, offset = [0, 17, 129], [0], 0
    keys, values = [], []
    for length, chunk in zip(LENGTHS, chunks):
        end, prefix = offset + length, length - chunk
        prefix_slots = slots[offset : offset + prefix]
        cache[prefix_slots // BLOCK, :, prefix_slots % BLOCK] = sq.reference_quantize(
            layout.to_storage(key[offset : offset + prefix]),
            layout.to_storage(value[offset : offset + prefix]),
            fmt,
        )
        keys.append(key[end - chunk : end])
        values.append(value[end - chunk : end])
        starts.append(starts[-1] + chunk)
        offset = end
    key, value = torch.cat(keys), torch.cat(values)
    query = torch.randn(starts[-1], KV_HEADS * group, hip.head_dim, device="cuda").to(
        dtype
    )
    output = hip.prefill(
        query,
        key,
        value,
        cache,
        table,
        torch.tensor(starts, device="cuda", dtype=torch.int32),
        torch.tensor(LENGTHS, device="cuda", dtype=torch.int32),
        signs,
        cache_dtype,
        splits=splits,
    )
    expected = torch.empty_like(output)
    for req, (length, chunk) in enumerate(zip(LENGTHS, chunks)):
        if chunk == 0:
            continue
        lo, hi = starts[req : req + 2]
        prefix = length - chunk
        pos = torch.arange(prefix, device="cuda")
        k, v = sq.reference_dequantize(
            cache[table[req, pos // BLOCK].long(), :, pos % BLOCK], fmt
        )
        k = torch.cat((layout.from_storage(k), key[lo:hi].float()))
        v = torch.cat((layout.from_storage(v), value[lo:hi].float()))
        k, v = (x.repeat_interleave(group, dim=1) for x in (k, v))
        scores = torch.einsum("qhd,khd->hqk", query[lo:hi].float(), k)
        scores *= hip.head_dim**-0.5
        causal = (
            torch.arange(length, device="cuda")[None, :]
            > (prefix + torch.arange(chunk, device="cuda"))[:, None]
        )
        weights = scores.masked_fill(causal[None], -torch.inf).softmax(-1)
        expected[lo:hi] = torch.einsum("hqk,khd->qhd", weights, v)
    _check_close(output, expected, 2e-2)
