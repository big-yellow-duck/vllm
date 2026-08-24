# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_VALIDATED_BM32_SHAPES = {
    (523, 5120, 3072),
    (523, 5120, 8704),
    (523, 7168, 5120),
    (523, 8192, 5120),
    (784, 5120, 3072),
    (784, 5120, 8704),
    (784, 8192, 5120),
}


def should_use_rdna4_bm32(m: int, n: int, k: int) -> bool:
    """Select BM32 only in measured tile bands and workload shapes."""
    return 64 <= m <= 128 or 225 <= m <= 256 or (m, n, k) in _VALIDATED_BM32_SHAPES


@triton.jit
def _rdna4_fp8_block_scaled_mm_bm32(
    A,
    B,
    C,
    As,
    Bs,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    stride_am,
    stride_bm,
    stride_asm,
    stride_ask,
    stride_bsm,
    stride_bsk,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, 32)
    num_pid_n = tl.cdiv(N, 128)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * 32 + tl.arange(0, 32)) % M
    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * stride_bm + offs_k[:, None]
    as_ptrs = As + offs_m * stride_asm
    bs_ptrs = Bs + pid_n * stride_bsm

    accumulator = tl.zeros((32, 128), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, 128)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(as_ptrs + k_block * stride_ask)
        b_s = tl.load(bs_ptrs + k_block * stride_bsk)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 128
        b_ptrs += 128

    out_m = pid_m * 32 + tl.arange(0, 32)
    out_n = pid_n * 128 + tl.arange(0, 128)
    tl.store(
        C + out_m[:, None] * N + out_n[None, :],
        accumulator.to(tl.bfloat16),
        mask=out_m[:, None] < M,
    )


def _run_rdna4_bm32(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 32) * triton.cdiv(n, 128),)
    _rdna4_fp8_block_scaled_mm_bm32[grid](
        a,
        weight,
        out,
        a_scale,
        weight_scale,
        m,
        n,
        k,
        a.stride(0),
        weight.stride(0),
        a_scale.stride(0),
        a_scale.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        GROUP_M=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def _supports_specialized_layout(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    m, k = a.shape
    n = weight.shape[0]
    return (
        a.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and a_scale.dtype == torch.float32
        and weight_scale.dtype == torch.float32
        and a.is_contiguous()
        and weight.stride(1) == 1
        and a_scale.is_contiguous()
        and weight_scale.is_contiguous()
        and n % 128 == 0
        and k % 128 == 0
        and a_scale.shape == (m, k // 128)
        and weight_scale.shape == (n // 128, k // 128)
    )


def _rdna4_fp8_block_scaled_mm_impl(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = weight.shape[0]
    specialized_layout = _supports_specialized_layout(a, weight, a_scale, weight_scale)
    if specialized_layout and m in (1, 2) and k % 128 == 0:
        return ops.rdna4_fp8_block_scaled_mm_decode(a, weight, a_scale, weight_scale)
    if specialized_layout and should_use_rdna4_bm32(m, n, k):
        return _run_rdna4_bm32(a, weight, a_scale, weight_scale)
    return w8a8_triton_block_scaled_mm(
        a, weight, a_scale, weight_scale, [128, 128], torch.bfloat16
    )


def _rdna4_fp8_block_scaled_mm_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del a_scale, weight_scale
    return torch.empty(
        (a.size(0), weight.size(0)), dtype=torch.bfloat16, device=a.device
    )


direct_register_custom_op(
    op_name="rdna4_fp8_block_scaled_mm",
    op_func=_rdna4_fp8_block_scaled_mm_impl,
    mutates_args=[],
    fake_impl=_rdna4_fp8_block_scaled_mm_fake,
)


class RDNA4Fp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Hybrid block-FP8 kernel for RDNA4."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_rocm():
            return False, "RDNA4 block-FP8 kernels require ROCm"
        from vllm.platforms.rocm import on_rdna4

        if not on_rdna4():
            return False, "RDNA4 block-FP8 kernels require gfx1200 or gfx1201"
        if not hasattr(torch.ops._rocm_C, "rdna4_fp8_block_scaled_mm_decode"):
            return False, "vLLM was built without the RDNA4 block-FP8 extension"
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        can_implement, reason = super().can_implement(config)
        if not can_implement:
            return can_implement, reason
        if config.input_dtype != torch.bfloat16:
            return False, "RDNA4 block-FP8 supports only BF16 input"
        if config.out_dtype != torch.bfloat16:
            return False, "RDNA4 block-FP8 supports only BF16 output"
        if config.activation_quant_key.dtype != torch.float8_e4m3fn:
            return False, "RDNA4 block-FP8 requires OCP FP8 activations"
        if config.weight_quant_key.dtype != torch.float8_e4m3fn:
            return False, "RDNA4 block-FP8 requires OCP FP8 weights"
        if config.activation_quant_key.scale.dtype != torch.float32:
            return False, "RDNA4 block-FP8 requires FP32 activation scales"
        if config.weight_quant_key.scale.dtype != torch.float32:
            return False, "RDNA4 block-FP8 requires FP32 weight scales"
        if config.activation_quant_key.scale.group_shape != GroupShape(1, 128):
            return False, "RDNA4 block-FP8 requires 1x128 activation scales"
        if config.weight_quant_key.scale.group_shape != GroupShape(128, 128):
            return False, "RDNA4 block-FP8 requires 128x128 weight scales"
        n, k = config.weight_shape
        if n <= 0 or n % 128 != 0 or k <= 0 or k % 128 != 0:
            return False, "RDNA4 block-FP8 requires N and K divisible by 128"
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.rdna4_fp8_block_scaled_mm(A, B, As, Bs)
