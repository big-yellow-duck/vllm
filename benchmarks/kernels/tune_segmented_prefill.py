# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded offline search for short-prefill launch ranges."""

import argparse
import itertools
import json
from pathlib import Path

import torch
from benchmark_segmented_prefill import run_case

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cases", type=Path, required=True)
p.add_argument("--backend", default="segmented")
p.add_argument("--samples", type=int, default=7)
a = p.parse_args()
if a.backend == "segmented":
    configs = [
        dict(bm=m, bn=n, bk=k, stages=st, warps=4, splits=s)
        for (m, n, k, st), s in itertools.product(
            [
                (16, 32, 256, 1),
                (16, 64, 256, 1),
                (32, 32, 64, 1),
                (32, 64, 64, 1),
                (32, 64, 64, 2),
                (32, 64, 128, 1),
                (64, 32, 64, 1),
                (64, 64, 64, 1),
                (64, 64, 64, 2),
            ],
            [1, 2, 4, 8, 16, 32],
        )
    ] + [
        dict(bm=m, bn=64, bk=64, stages=st, warps=8, splits=s)
        for m, st, s in itertools.product([32, 64], [1, 2], [1, 2, 4, 8, 16])
    ]
else:
    configs = [None] + [
        dict(splits=s, TILE_SIZE=n, force3d=True, num_stages=1)
        for s, n in itertools.product([4, 8, 16, 32, 64], [16, 32])
    ]
records = json.loads(a.output.read_text()) if a.output.exists() else []
with torch.inference_mode():
    for case in json.loads(a.cases.read_text()):
        done = {
            json.dumps(row["config"], sort_keys=True)
            for row in records
            if row["case"] == case and row["backend"] == a.backend
        }
        pending = [
            cfg for cfg in configs if json.dumps(cfg, sort_keys=True) not in done
        ]
        if a.backend == "segmented":
            qcap = max(case["queries"])
            batch = len(case["queries"])
            heads = case.get("heads", 12)
            hk = case.get("kv_heads", 2)
            pending = [
                cfg
                for cfg in pending
                if cfg["bm"] <= max(32, 1 << (qcap * (heads // hk) - 1).bit_length())
                and 16
                <= batch
                * hk
                * ((qcap * (heads // hk) + cfg["bm"] - 1) // cfg["bm"])
                * cfg["splits"]
                <= 768
            ]
        if pending:
            records.extend(run_case(case, [a.backend], a.samples, pending))
        a.output.write_text(json.dumps(records, indent=2) + "\n")
