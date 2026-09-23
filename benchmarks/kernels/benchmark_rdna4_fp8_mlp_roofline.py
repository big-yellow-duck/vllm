# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure cache-hot and rotating-weight RDNA4 FP8 MLP GEMMs.

The rotating variant uses distinct weight allocations to mimic streaming
through model layers. Its weight-bytes/time rate is logical, not a DRAM counter.

Examples:
    python benchmarks/kernels/benchmark_rdna4_fp8_mlp_roofline.py
    python benchmarks/kernels/benchmark_rdna4_fp8_mlp_roofline.py --shape 8,8704,5120

"""

import argparse
import statistics

import flydsl  # noqa: F401 -- load compiler libraries before PyTorch
import torch

from vllm.model_executor.kernels.linear.scaled_mm.flydsl_kernels import (
    rdna4_fp8_blockscale as rdna4_flydsl,
)

DEFAULT_SHAPES = ((8, 8704, 5120), (8, 5120, 4352))
rdna4_fp8_block_scaled_mm = rdna4_flydsl.rdna4_fp8_block_scaled_mm


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,N,K") from exc
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive sizes")
    return shape


def _time_graph(graph: torch.cuda.CUDAGraph, calls: int, replays: int) -> float:
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(samples)


def _run_shape(m: int, n: int, k: int, weights_count: int, replays: int) -> None:
    a = torch.randint(0, 2, (m, k), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    weight = torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    weights = [weight.clone() for _ in range(weights_count)]
    a_scale = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
    weight_scale = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32)

    # Compile before capture so both graphs measure only GPU GEMMs.
    rdna4_fp8_block_scaled_mm(a, weights[0], a_scale, weight_scale)
    torch.cuda.synchronize()
    for mode in ("hot", "rotated"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index in range(weights_count):
                chosen = weights[0] if mode == "hot" else weights[index]
                rdna4_fp8_block_scaled_mm(a, chosen, a_scale, weight_scale)
        median_us = _time_graph(graph, weights_count, replays)
        weight_bytes = n * k
        print(
            f"{m},{n},{k},{mode},{weights_count},{median_us:.3f},"
            f"{weight_bytes / median_us / 1000:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--weights", type=int, default=16)
    parser.add_argument("--replays", type=int, default=21)
    args = parser.parse_args()
    if args.weights <= 0 or args.replays <= 0:
        parser.error("--weights and --replays must be positive")
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise RuntimeError(f"benchmark requires gfx1200 or gfx1201, got {arch!r}")
    print(f"device={properties.name},arch={arch}")
    print("M,N,K,mode,distinct_weight_allocations,median_us,logical_weight_GBs")
    for shape in args.shape or DEFAULT_SHAPES:
        _run_shape(*shape, args.weights, args.replays)


if __name__ == "__main__":
    main()
