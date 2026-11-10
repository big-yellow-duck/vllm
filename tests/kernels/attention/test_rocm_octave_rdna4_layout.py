# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for RDNA4 Octave layouts and bounded split scheduling."""

import pytest
import torch

from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.attention.ops.rocm_octave_rdna4.geometry import OctaveGeometry
from vllm.v1.attention.ops.rocm_octave_rdna4.splits import recommend_splits


@pytest.mark.parametrize(
    "dim,rope",
    [
        (32, 32),
        (40, 32),
        (64, 0),
        (80, 64),
        (96, 24),
        (128, 128),
        (128, 32),
        (160, 64),
        (192, 64),
        (256, 64),
        (256, 128),
        (384, 64),
        (512, 64),
        (768, 128),
        (1024, 128),
    ],
)
@pytest.mark.parametrize("name", ["octave_k3v3", "octave_k3v4", "octave_k3v3_compact"])
def test_layout_preserves_logical_attention(dim, rope, name):
    layout = OctaveGeometry(dim, rope).layout(name)
    generator = torch.Generator().manual_seed(291)
    x = torch.randn(3, dim, generator=generator, dtype=torch.float64)
    y = torch.randn(3, dim, generator=generator, dtype=torch.float64)
    assert torch.equal(layout.from_storage(layout.to_storage(x)), x)
    torch.testing.assert_close(
        (layout.to_storage(x) * layout.to_storage(y)).sum(-1),
        (x * y).sum(-1),
        rtol=1e-14,
        atol=1e-14,
    )
    fmt = layout.reference_format()
    assert (layout.slot_bytes, layout.off_scales, layout.num_k_scales) == (
        fmt.slot_bytes,
        fmt.off_scales,
        fmt.num_k_scales,
    )
    assert layout.sign_bits().shape == (layout.storage_head_dim // 32,)


@pytest.mark.parametrize(
    "dim,rope", [(0, 0), (1025, 64), (128, -1), (128, 129), (1024, 32), (128.5, 32)]
)
def test_invalid_geometry_rejected(dim, rope):
    with pytest.raises(ValueError):
        OctaveGeometry(dim, rope)


@pytest.mark.parametrize("dim", range(64, 1025, 64))
def test_sign_words_match_wire_format(dim):
    layout = OctaveGeometry(dim, 0).layout("octave_k3v4")
    assert torch.equal(layout.sign_bits(), sq.sign_bits(dim))


@pytest.mark.parametrize("dim", [32, 80, 128, 192, 256, 384, 512, 1024])
@pytest.mark.parametrize("group", [1, 3, 6, 16, 64])
@pytest.mark.parametrize("prefill", [False, True])
def test_general_geometry_scratch_budget(dim, group, prefill):
    for budget in (0, 1000, 3 << 20, 1 << 30):
        splits = recommend_splits(
            tokens=8192,
            heads=group * 2,
            kv_heads=2,
            requests=1 if prefill else 8192,
            capacity=270336,
            page_size=16,
            cache_dtype="octave_k3v4",
            prefill=prefill,
            head_dim=dim,
            storage_head_dim=OctaveGeometry(dim, min(dim, 64)).block_head_dim,
            legacy_fast=False,
            workspace_limit_bytes=budget,
        )
        assert 1 <= splits <= 256 and splits & (splits - 1) == 0
        assert (
            8192 * group * 2 * splits * (dim + 2) * 4 if splits > 1 else 0
        ) <= budget


@pytest.mark.parametrize(
    "tokens,heads,kv,requests,prefill",
    [
        (1, 8, 2, 1, False),
        (16, 12, 2, 16, False),
        (8192, 12, 2, 8192, False),
        (257, 8, 8, 7, True),
        (8192, 12, 2, 1, True),
        (8192, 48, 8, 128, True),
    ],
)
@pytest.mark.parametrize("budget", [0, 1000, 3 << 20, 1 << 30])
def test_automatic_workspace_never_exceeds_budget(
    tokens, heads, kv, requests, prefill, budget
):
    splits = recommend_splits(
        tokens=tokens,
        heads=heads,
        kv_heads=kv,
        requests=requests,
        capacity=270336,
        page_size=16,
        cache_dtype="octave_k3v4",
        prefill=prefill,
        workspace_limit_bytes=budget,
    )
    assert 1 <= splits <= 256 and splits & (splits - 1) == 0
    allocated = tokens * heads * splits * 258 * 4 if splits > 1 else 0
    assert allocated <= budget


@pytest.mark.parametrize("heads,kv", [(8, 8), (8, 2), (16, 2)])
def test_long_context_other_gqa_ratios_are_split(heads, kv):
    splits = recommend_splits(
        tokens=1,
        heads=heads,
        kv_heads=kv,
        requests=1,
        capacity=262144,
        page_size=16,
        cache_dtype="octave_k3v4",
    )
    assert splits >= 128


@pytest.mark.parametrize("bad", [-1, 1.5, None])
def test_invalid_workspace_budget_is_rejected(bad):
    with pytest.raises(ValueError):
        recommend_splits(
            tokens=1,
            heads=12,
            kv_heads=2,
            requests=1,
            capacity=1024,
            page_size=16,
            cache_dtype="octave_k3v4",
            workspace_limit_bytes=bad,
        )
