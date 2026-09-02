# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated FlyDSL versus HIP benchmark for RDNA4 block-scaled FP8 MM."""

import argparse
import statistics

import flydsl  # noqa: F401 -- load compiler libraries before PyTorch
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.flydsl_kernels import (
    rdna4_fp8_blockscale as rdna4_flydsl,
)

rdna4_fp8_block_scaled_mm = rdna4_flydsl.rdna4_fp8_block_scaled_mm

DEFAULT_SHAPES = [
    (1, 5120, 3072),
    (2, 5120, 8704),
    (4, 5120, 3072),
    (16, 8192, 5120),
    (17, 5120, 3072),
    (33, 17408, 5120),
    (39, 16384, 8192),
    (48, 17408, 5120),
    (64, 5120, 3072),
    (65, 128, 256),
    (256, 8192, 5120),
    (523, 5120, 8704),
    (784, 7168, 5120),
    (1024, 8192, 5120),
]


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,N,K") from exc
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape


def _inputs(m: int, n: int, k: int):
    generator = torch.Generator(device="cuda").manual_seed(m * 1_000_003 + n * 101 + k)
    a = (torch.randn((m, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    weight = (torch.randn((n, k), device="cuda", generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.rand(
        (m, k // 128), device="cuda", generator=generator, dtype=torch.float32
    )
    weight_scale = torch.rand(
        (n // 128, k // 128),
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    )
    return a, weight, a_scale, weight_scale


def _hip_mm(a, weight, a_scale, weight_scale):
    op = (
        ops.rdna4_fp8_block_scaled_mm_decode
        if a.shape[0] <= 64
        else ops.rdna4_fp8_block_scaled_mm_prefill
    )
    return op(a, weight, a_scale, weight_scale)


def _graph_kernel_ms(fn, calls: int, replays: int) -> float:
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end) / calls)
    return statistics.median(elapsed)


def _tflops(m: int, n: int, k: int, elapsed_ms: float) -> float:
    return 2.0 * m * n * k / elapsed_ms / 1.0e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--replays", type=int, default=9)
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise RuntimeError(f"benchmark requires gfx1200 or gfx1201, got {arch!r}")
    if not hasattr(torch.ops._rocm_C, "rdna4_fp8_block_scaled_mm_decode"):
        raise RuntimeError("vLLM was built without the RDNA4 HIP reference")

    print(f"device={properties.name},arch={arch}")
    print(
        "M,N,K,max_abs,mean_abs,flydsl_ms,hip_ms,hip_over_flydsl,"
        "flydsl_tflops,hip_tflops"
    )
    for m, n, k in args.shape or DEFAULT_SHAPES:
        inputs = _inputs(m, n, k)
        fly_out_buffer = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        rdna4_fp8_block_scaled_mm(*inputs, out=fly_out_buffer)
        fly_out = rdna4_fp8_block_scaled_mm(*inputs)
        hip_out = _hip_mm(*inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(fly_out, hip_out, atol=0.0625, rtol=0.02)
        error = (fly_out.float() - hip_out.float()).abs()

        fly_ms = _graph_kernel_ms(
            lambda inputs=inputs, out=fly_out_buffer: rdna4_fp8_block_scaled_mm(
                *inputs, out=out
            ),
            args.calls,
            args.replays,
        )
        hip_ms = _graph_kernel_ms(
            lambda inputs=inputs: _hip_mm(*inputs), args.calls, args.replays
        )
        print(
            f"{m},{n},{k},{error.max().item():.6f},"
            f"{error.mean().item():.6f},{fly_ms:.6f},{hip_ms:.6f},"
            f"{hip_ms / fly_ms:.6f},{_tflops(m, n, k, fly_ms):.3f},"
            f"{_tflops(m, n, k, hip_ms):.3f}"
        )


if __name__ == "__main__":
    main()
