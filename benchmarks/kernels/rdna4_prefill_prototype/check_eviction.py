# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import torch
import triton
from calibrate import read_sum
from eviction import ReadEviction
from tune import measure

x = torch.randn(8 * 1024**2, device="cuda", dtype=torch.bfloat16)
y = torch.empty(triton.cdiv(x.numel(), 4096), device="cuda")
read = ReadEviction()
write = torch.zeros(256 * 1024**2, device="cuda", dtype=torch.int8)
for _ in range(20):
    read_sum[(y.numel(),)](x, y, x.numel(), 4096, num_warps=4)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    read_sum[(y.numel(),)](x, y, x.numel(), 4096, num_warps=4)
r = []
for round in range(5):
    for name, evict in [("write", write), ("read", read)]:
        t = measure(g, evict, True, samples=30)
        row = dict(round=round, eviction=name, us=t, GBs=x.numel() * 2 / t / 1000)
        print(row, flush=True)
        r.append(row)
Path("eviction-comparison.json").write_text(json.dumps(r, indent=2))
