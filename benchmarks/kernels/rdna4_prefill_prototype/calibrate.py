# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from tune import measure


@tr.jit
def read_sum(X, OUT, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, mask=i < N, other=0).to(tl.float32)
    tl.store(OUT + tl.program_id(0), tl.sum(x, 0))


def main():
    records = []
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.int8)
    for size in (16 * 1024**2, 64 * 1024**2, 256 * 1024**2):
        x = torch.randn(size // 2, device="cuda", dtype=torch.bfloat16)
        for b in (1024, 2048, 4096, 8192, 16384):
            for w in (4, 8):
                out = torch.empty(tr.cdiv(x.numel(), b), device="cuda")

                def run(x=x, out=out, b=b, w=w):
                    read_sum[(out.numel(),)](x, out, x.numel(), b, num_warps=w)

                for _ in range(10):
                    run()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run()
                for _ in range(10):
                    g.replay()
                cold = measure(g, flush, True, samples=30)
                warm = measure(g, flush, False, samples=30)
                r = dict(
                    bytes=size,
                    block=b,
                    warps=w,
                    cold_us=cold,
                    reuse_us=warm,
                    cold_GBs=size / cold / 1e3,
                )
                records.append(r)
                print(json.dumps(r), flush=True)
    Path("calibration.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
