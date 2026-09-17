# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Isolated launch search for the existing context kernel on Q2/C65536."""

import itertools
import json
from pathlib import Path

import torch
from eviction import ReadEviction
from fixtures import inputs
from tune import error, measure, reference

from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

d = inputs([2], [65536])
d["context"] = 65536
ref = reference(d)
out = torch.empty_like(d["q"])
evict = ReadEviction()
records = []
for m, n, u, w in itertools.product((16, 32), (16, 32, 64), (1, 4), (4,)):
    config = dict(
        BLOCK_M=m,
        BLOCK_N=n,
        num_unroll_cache=u,
        num_unroll_request=1,
        num_warps=w,
        num_stages=1,
    )
    r = dict(config=config)
    try:

        def run(config=config):
            context_attention_fwd(
                d["q"],
                d["kd"],
                d["vd"],
                out,
                "auto",
                d["kc"],
                d["vc"],
                d["table"],
                d["starts"],
                d["lens"],
                65538,
                2,
                d["ks"],
                d["vs"],
                sm_scale=0.0625,
                skip_decode=True,
                _launch_config=config,
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        r["error"] = error(out, ref)
        assert r["error"] < 0.01
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        r["read_eviction_us"] = measure(g, evict, True, samples=12)
        r["reuse_us"] = measure(g, evict, False, samples=12)
    except Exception as e:
        r["failure"] = str(e)
    records.append(r)
    print(json.dumps(r), flush=True)
    Path("context-search.json").write_text(json.dumps(records, indent=2) + "\n")
