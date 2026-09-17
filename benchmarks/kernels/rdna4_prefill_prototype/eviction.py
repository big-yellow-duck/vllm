# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only cache eviction to avoid a large dirty-cache writeback workload."""

import torch
import triton
import triton.language as tl


@triton.jit
def evict_kernel(X, OUT, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, mask=i < N, other=0).to(tl.int32)
    tl.store(OUT + tl.program_id(0), tl.sum(x, 0))


class ReadEviction:
    def __init__(self, size=256 * 1024**2):
        self.buffer = torch.zeros(size, device="cuda", dtype=torch.int8)
        self.out = torch.empty(
            triton.cdiv(size, 16384), device="cuda", dtype=torch.int32
        )
        self.zero_()

    def zero_(self):
        evict_kernel[(self.out.numel(),)](
            self.buffer, self.out, self.buffer.numel(), 16384, num_warps=4
        )
