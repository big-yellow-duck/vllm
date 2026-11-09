# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the selected RDNA4 attention paths against the Torch oracle."""

import math

import pytest
import torch
from octave_hip import HEAD, HipOctave, build_library
from octave_reference import (
    OctaveFormat,
    reference_cache_store,
    reference_decode,
    reference_prefill,
)
from task_runner import check_output, graph_for, int32, paged_inputs


@pytest.fixture(scope="module")
def hip():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("A ROCm GPU is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1200", "gfx1201"):
        pytest.skip("The native attention kernels require gfx1200 or gfx1201")
    return HipOctave(build_library(arch=arch))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "cache_dtype,length,chunk,splits",
    [
        ("octave_k3v3", 128, 0, 1),
        ("octave_k3v3_compact", 8192, 0, 32),
        ("octave_k3v3", 32768, 0, 128),
        ("octave_k3v4", 4352, 256, 8),
    ],
    ids=["short-cooperative", "compact-blockwise", "long-dot4", "prefill-compact-lds"],
)
def test_selected_paths_replay_with_changed_queries(
    hip, dtype, cache_dtype, length, chunk, splits
):
    fmt = OctaveFormat.from_cache_dtype(cache_dtype, HEAD, 64)
    inp = paged_inputs([length], 16, 2, fmt, dtype, True, 421)
    reference_cache_store(inp.key, inp.value, inp.cache, inp.slots, fmt)
    count = chunk or 1
    query = inp.randn(count, 12).mul_(2)
    output_storage = torch.full((query.numel() + 16,), 12345.0, device="cuda")
    output = output_storage[8:-8].view(query.shape)
    scratch_shape = (count, 12, splits, HEAD + 2)
    scratch_storage = torch.full(
        (math.prod(scratch_shape) + 16,), 12345.0, device="cuda"
    )
    scratch = scratch_storage[8:-8].view(scratch_shape)
    lengths = int32([length])
    requests = int32([0])
    starts = int32([0, count])
    if chunk:
        key, value = inp.key[-chunk:], inp.value[-chunk:]

        def call():
            return hip.prefill(
                query,
                key,
                value,
                inp.cache,
                inp.table,
                starts,
                lengths,
                inp.signs,
                cache_dtype,
                splits=splits,
                out=output,
                scratch=scratch,
            )

        def reference():
            return reference_prefill(
                query, key, value, inp.cache, inp.table, starts, lengths, fmt, 1 / 16
            )
    else:

        def call():
            return hip.decode(
                query,
                inp.cache,
                inp.table,
                requests,
                lengths,
                inp.signs,
                cache_dtype,
                splits=splits,
                out=output,
                scratch=scratch,
            )

        def reference():
            return reference_decode(
                query, inp.cache, inp.table, requests, lengths, fmt, 1 / 16
            )

    call()
    check_output(output, reference(), "selected path", 0.02)
    eager = output.clone()
    graph = graph_for(call, 1)
    output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(output, eager)
    query.neg_()
    graph.replay()
    check_output(output, reference(), "selected path query mutation", 0.02)
    if not chunk:
        lengths.zero_()
        requests.fill_(999999)
        graph.replay()
        assert torch.count_nonzero(output).item() == 0
    for storage in (output_storage, scratch_storage):
        assert torch.all(storage[:8] == 12345.0).item()
        assert torch.all(storage[-8:] == 12345.0).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compact_prefill_handles_empty_requests_and_request_boundaries(hip, dtype):
    fmt = OctaveFormat.from_cache_dtype("octave_k3v4", HEAD, 64)
    lengths, chunks = [1024, 4223, 4226], [0, 127, 130]
    inp = paged_inputs(lengths, 16, 2, fmt, dtype, True, 637)
    reference_cache_store(inp.key, inp.value, inp.cache, inp.slots, fmt)
    key_parts, value_parts, offset = [], [], 0
    for length, chunk in zip(lengths, chunks):
        end = offset + length
        key_parts.append(inp.key[end - chunk : end])
        value_parts.append(inp.value[end - chunk : end])
        offset = end
    query = inp.randn(sum(chunks), 12).mul_(2)
    key, value = torch.cat(key_parts), torch.cat(value_parts)
    starts, seq_lens = int32([0, 0, 127, 257]), int32(lengths)
    expected = reference_prefill(
        query, key, value, inp.cache, inp.table, starts, seq_lens, fmt, 1 / 16
    )
    output = hip.prefill(
        query,
        key,
        value,
        inp.cache,
        inp.table,
        starts,
        seq_lens,
        inp.signs,
        "octave_k3v4",
        splits=8,
    )
    check_output(output, expected, "compact prefill request boundaries", 0.02)
