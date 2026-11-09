# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-only format and attention checks for future native Octave kernels."""

import struct

import pytest
import torch
import torch.nn.functional as F
from octave_reference import (
    OctaveFormat,
    format_ops,
    reference_cache_store,
    reference_decode,
    reference_dequantize,
    reference_prefill,
    reference_quantize,
)

HEAD, ROPE, KV_HEADS, GROUP = 256, 64, 2, 6
SCALE = HEAD**-0.5
FORMATS = [
    pytest.param("octave_k3v3_compact", 196, id="k3v3-compact"),
    pytest.param("octave_k3v3", 216, id="k3v3"),
    pytest.param("octave_k3v4", 248, id="k3v4"),
]


@pytest.fixture(params=["cpu", "rocm"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("ROCm device required")
    return torch.device("cuda", 0)


def _cache(num_pages, page_size, fmt, device, token_major=False):
    if token_major:
        return torch.full(
            (num_pages, page_size, KV_HEADS, fmt.slot_bytes),
            255,
            device=device,
            dtype=torch.uint8,
        ).transpose(1, 2)
    return torch.full(
        (num_pages, KV_HEADS, page_size, fmt.slot_bytes),
        255,
        device=device,
        dtype=torch.uint8,
    )


def _relative_error(actual, expected):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    return ((actual - expected).norm() / expected.norm()).item()


def _dense_oracle(query, key, value, lengths):
    """Independent FP64 CPU SDPA over logical, unpacked sequences."""
    query = query.cpu().double().transpose(0, 1)
    key = key.cpu().double().repeat_interleave(GROUP, dim=1).transpose(0, 1)
    value = value.cpu().double().repeat_interleave(GROUP, dim=1).transpose(0, 1)
    mask = torch.arange(key.shape[1]) < torch.tensor(lengths)[:, None]
    return F.scaled_dot_product_attention(
        query, key, value, mask, scale=SCALE
    ).transpose(0, 1)


@pytest.mark.parametrize("cache_dtype,slot_bytes", FORMATS)
def test_slot_matches_golden_bytes(device, cache_dtype, slot_bytes):
    fmt = OctaveFormat.from_cache_dtype(cache_dtype, HEAD, ROPE)
    assert fmt.slot_bytes == slot_bytes

    # With normalized magnitudes of one, the codes are 2/5 (3 bit) or
    # 4/11 (4 bit). Alternating groups exercise both planes and both nibbles.
    pattern = torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1], device=device).float()
    k_rot = pattern.repeat(HEAD // 8).view(1, 1, HEAD)
    v_rot = k_rot * 8
    signs = format_ops.signs(HEAD, str(device))

    def codes3(n):
        return bytes([0x66, 0x99] * (n // 8) + [0xAA, 0x55] * (n // 16))

    def codes4(n):
        return bytes([0xB4, 0x4B] * (n // 4))

    if fmt.compact:
        k_bytes, k_scales = codes3(256), [64 / 45]
    else:
        amplitudes = torch.tensor([1, 2, 4, 8], device=device).repeat_interleave(64)
        k_rot = k_rot * amplitudes
        k_bytes = codes4(64) + codes3(192)
        k_scales = [64 / 43, 2 * 64 / 45, 4 * 64 / 45, 8 * 64 / 45]
    v_bytes, v_scale = (
        (codes4(256), 8 * 64 / 43)
        if fmt.v_bits == 4
        else (
            codes3(256),
            8 * 64 / 45,
        )
    )
    expected = (
        k_bytes
        + v_bytes
        + struct.pack("<" + "e" * (len(k_scales) + 1), *k_scales, v_scale)
    )
    expected = expected.ljust(slot_bytes, b"\0")

    key = format_ops.unrotate(k_rot, signs, fmt.k_rotation_block)
    value = format_ops.unrotate(v_rot, signs, HEAD)
    packed = reference_quantize(key, value, fmt)
    assert bytes(packed.flatten().tolist()) == expected
    k_got, v_got = reference_dequantize(packed, fmt, rotated=True)
    torch.testing.assert_close(k_got, k_rot, atol=0, rtol=5e-4)
    torch.testing.assert_close(v_got, v_rot, atol=0, rtol=5e-4)

    zero = torch.zeros_like(key)
    k_zero, v_zero = reference_dequantize(reference_quantize(zero, zero, fmt), fmt)
    assert torch.count_nonzero(k_zero) == torch.count_nonzero(v_zero) == 0


@pytest.mark.parametrize(
    "token_major", [False, True], ids=["head-major", "token-major"]
)
def test_store_writes_only_mapped_slots(device, token_major):
    fmt = OctaveFormat.from_cache_dtype("octave_k3v3", HEAD, ROPE)
    cache = _cache(3, 16, fmt, device, token_major)
    key = torch.arange(5 * KV_HEADS * HEAD, device=device).reshape(5, KV_HEADS, HEAD)
    key = (key % 17 - 8).half()
    value = key.roll(3, dims=-1)
    expected = cache.clone()
    expected[2, :, 5] = reference_quantize(key[0:1], value[0:1], fmt)[0]
    expected[0, :, 2] = reference_quantize(key[2:3], value[2:3], fmt)[0]
    slots = torch.tensor([37, -1, 2, -1], dtype=torch.int64, device=device)

    reference_cache_store(key, value, cache, slots, fmt)
    assert torch.equal(cache, expected)
    reference_cache_store(key, value, cache, slots.new_full(slots.shape, -1), fmt)
    assert torch.equal(cache, expected)


@pytest.mark.parametrize("cache_dtype,slot_bytes", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_paged_decode_and_chunked_prefill_accuracy(
    device, cache_dtype, slot_bytes, dtype
):
    fmt = OctaveFormat.from_cache_dtype(cache_dtype, HEAD, ROPE)
    generator = torch.Generator().manual_seed(13)

    def randn(tokens, heads):
        return torch.randn(tokens, heads, HEAD, generator=generator)

    seq_lens, chunk_lens = [1, 33, 513], [1, 4, 129]
    page_size = 16
    page_counts = [(n + page_size - 1) // page_size for n in seq_lens]
    num_pages = sum(page_counts) + 2
    pages = iter(
        torch.randperm(num_pages, generator=generator).split(page_counts + [2])
    )
    cache = _cache(num_pages, page_size, fmt, device, token_major=True)
    block_table = torch.full(
        (len(seq_lens), max(page_counts)), -1, device=device, dtype=torch.int32
    )
    q_chunks, k_chunks, v_chunks = [], [], []
    q_decode, q_to_req, q_to_klen = [], [], []
    decode_expected, prefill_expected = [], []
    decode_dense, prefill_dense = [], []
    keys, values, keys_quantized, values_quantized = [], [], [], []
    query_start = [0]

    for req, (seq_len, chunk_len) in enumerate(zip(seq_lens, chunk_lens)):
        page_ids = next(pages).to(device)
        block_table[req, : page_ids.numel()] = page_ids
        positions = torch.arange(seq_len, device=device)
        slots = page_ids[positions // page_size] * page_size + positions % page_size
        key, value = randn(seq_len, KV_HEADS), randn(seq_len, KV_HEADS)
        key[..., 100] *= 8
        key, value = key.to(device, dtype), value.to(device, dtype)
        reference_cache_store(key, value, cache, slots, fmt)
        packed = reference_quantize(key, value, fmt)
        k_quant, v_quant = reference_dequantize(packed, fmt)
        keys.append(key)
        values.append(value)
        keys_quantized.append(k_quant)
        values_quantized.append(v_quant)

        query = randn(chunk_len, KV_HEADS * GROUP).to(device, dtype)
        q_chunks.append(query)
        prefix_len = seq_len - chunk_len
        k_chunks.append(key[prefix_len:])
        v_chunks.append(value[prefix_len:])
        query_start.append(query_start[-1] + chunk_len)
        lengths = list(range(prefix_len + 1, seq_len + 1))
        prefill_expected.append(
            _dense_oracle(
                query,
                torch.cat((k_quant[:prefix_len], key[prefix_len:])),
                torch.cat((v_quant[:prefix_len], value[prefix_len:])),
                lengths,
            )
        )
        prefill_dense.append(_dense_oracle(query, key, value, lengths))

        n_decode = min(4, chunk_len)
        q_decode.append(query[-n_decode:])
        q_to_req.extend([req] * n_decode)
        q_to_klen.extend(lengths[-n_decode:])
        decode_expected.append(
            _dense_oracle(q_decode[-1], k_quant, v_quant, lengths[-n_decode:])
        )
        decode_dense.append(prefill_dense[-1][-n_decode:])

    def metadata(values):
        return torch.tensor(values, device=device, dtype=torch.int32)

    q_decode.append(randn(1, KV_HEADS * GROUP).to(device, dtype))
    decode = reference_decode(
        torch.cat(q_decode),
        cache,
        block_table,
        metadata(q_to_req + [len(seq_lens)]),
        metadata(q_to_klen + [0]),
        fmt,
        SCALE,
    )
    assert torch.count_nonzero(decode[-1]) == 0
    decode = decode[:-1].cpu().double()
    prefill = (
        reference_prefill(
            torch.cat(q_chunks),
            torch.cat(k_chunks),
            torch.cat(v_chunks),
            cache,
            block_table,
            metadata(query_start),
            metadata(seq_lens),
            fmt,
            SCALE,
        )
        .cpu()
        .double()
    )
    # FP32 QK and PV reductions can differ from the CPU FP64 oracle near zero.
    torch.testing.assert_close(decode, torch.cat(decode_expected), atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(
        prefill, torch.cat(prefill_expected), atol=5e-6, rtol=2e-5
    )

    k_err = _relative_error(torch.cat(keys_quantized), torch.cat(keys))
    v_err = _relative_error(torch.cat(values_quantized), torch.cat(values))
    decode_err = _relative_error(decode, torch.cat(decode_dense))
    prefill_err = _relative_error(prefill, torch.cat(prefill_dense))
    print(
        f"{device} {cache_dtype} {slot_bytes} B {dtype} relative L2 "
        f"K {k_err:.2%} V {v_err:.2%} decode {decode_err:.2%} prefill {prefill_err:.2%}"
    )
    assert k_err < 0.20
    assert v_err < (0.12 if fmt.v_bits == 4 else 0.22)
    attention_error_limit = 0.20 if fmt.v_bits == 4 else 0.25
    assert decode_err < attention_error_limit
    assert prefill_err < attention_error_limit
