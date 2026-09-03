# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public router for the complete RDNA4 block-scaled FP8 GEMM family."""

import torch

from .rdna4_fp8_blockscale_common import SCALE_K
from .rdna4_fp8_blockscale_prefill import _run_prefill
from .rdna4_fp8_blockscale_small_m import _run_small_m


def validate_tensors(a, weight, a_scale, weight_scale):
    """Validate the tensor contract shared by every RDNA4 implementation."""
    import torch

    tensors = (a, weight, a_scale, weight_scale)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be on the GPU")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must be rank two")
    if a.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError("RDNA4 block-FP8 operands must use torch.float8_e4m3fn")
    if a_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        raise TypeError("RDNA4 block-FP8 scales must use torch.float32")
    if any(t.device != a.device for t in tensors):
        raise ValueError("RDNA4 block-FP8 inputs must share a device")
    arch = getattr(torch.cuda.get_device_properties(a.device), "gcnArchName", "")
    if not (arch.startswith("gfx1200") or arch.startswith("gfx1201")):
        raise ValueError(
            f"RDNA4 block-FP8 route requires gfx1200 or gfx1201, got {arch!r}"
        )

    m, k = a.shape
    n, weight_k = weight.shape
    if m <= 0:
        raise ValueError(f"RDNA4 block-FP8 requires positive M, got {m}")
    if n <= 0 or n % SCALE_K:
        raise ValueError(
            f"RDNA4 block-FP8 requires positive N divisible by 128, got {n}"
        )
    if k <= 0 or k % SCALE_K:
        raise ValueError(
            f"RDNA4 block-FP8 requires positive K divisible by 128, got {k}"
        )
    if weight_k != k:
        raise ValueError(f"weight K mismatch: A has K={k}, weight has K={weight_k}")
    if (
        not a.is_contiguous()
        or not a_scale.is_contiguous()
        or not weight_scale.is_contiguous()
    ):
        raise ValueError("A and both scale tensors must be contiguous")
    if weight.stride(1) != 1:
        raise ValueError("weight must have contiguous K rows (weight.stride(1) == 1)")
    if tuple(a_scale.shape) != (m, k // SCALE_K):
        raise ValueError(
            f"activation scale shape must be {(m, k // SCALE_K)}, "
            f"got {tuple(a_scale.shape)}"
        )
    if tuple(weight_scale.shape) != (n // SCALE_K, k // SCALE_K):
        raise ValueError(
            f"weight scale shape must be {(n // SCALE_K, k // SCALE_K)}, "
            f"got {tuple(weight_scale.shape)}"
        )
    return m, n, k


def rdna4_fp8_block_scaled_mm(a, weight, a_scale, weight_scale):
    """Run the complete RDNA4 block-scaled FP8 stack for any positive M."""
    m, n, k = validate_tensors(a, weight, a_scale, weight_scale)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    stream = torch.cuda.current_stream(a.device)
    if m <= 64:
        return _run_small_m(a, weight, a_scale, weight_scale, out, stream, m, n, k)

    return _run_prefill(a, weight, a_scale, weight_scale, out, stream, m, n, k)


__all__ = [
    "rdna4_fp8_block_scaled_mm",
]
