# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded AITER configuration probe on the same Q2/C65536 shape."""

import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch
from aiter.ops.triton.attention import unified_attention as aiter
from eviction import ReadEviction
from fixtures import inputs
from tune import error, measure, reference

d = inputs([2], [65536])
d["context"] = 65536
ref = reference(d)
out = torch.empty_like(d["q"])
evict = ReadEviction()
records = []
original = aiter.select_3d_config
for segments, tile in itertools.product((16, 32, 64, 128), (16, 32)):
    r = dict(segments=segments, tile=tile)

    def choose(*args, segments=segments, tile=tile, **kw):
        cfg = original(*args, **kw)
        return tuple(
            dict(c, NUM_SEGMENTS_PER_SEQ=segments, TILE_SIZE=tile) for c in cfg
        )

    try:
        with patch.object(aiter, "select_3d_config", choose):

            def run():
                aiter.unified_attention(
                    d["q"],
                    d["kn"],
                    d["vn"],
                    out,
                    d["starts"],
                    2,
                    d["lens"],
                    65538,
                    0.0625,
                    True,
                    (-1, -1),
                    d["table"],
                    0,
                    None,
                    d["ks"],
                    d["vs"],
                )

            for _ in range(6):
                run()
            torch.cuda.synchronize()
            r["error"] = error(out, ref)
            assert r["error"] < 0.01 and torch.isfinite(out).all()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
        r["read_eviction_us"] = measure(g, evict, True, samples=30)
        r["reuse_us"] = measure(g, evict, False, samples=30)
        g.replay()
        torch.cuda.synchronize()
        assert error(out, ref) < 0.01
    except Exception as e:
        r["failure"] = str(e)
    records.append(r)
    print(json.dumps(r), flush=True)
    Path("aiter-search.json").write_text(json.dumps(records, indent=2) + "\n")
