# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public split-selection contracts: bounded workspace and geometry coverage."""

import pytest
from octave_splits import recommend_splits


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
