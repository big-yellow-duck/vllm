# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import lru_cache

import torch

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


def should_use_rdna4_flydsl(m: int) -> bool:
    """Select the FlyDSL stack for its validated positive-M range."""
    return m >= 1


@lru_cache(maxsize=1)
def _load_rdna4_flydsl_mm():
    """Load the patched FlyDSL kernel lazily after vLLM initializes PyTorch."""
    from kernels.gemm.rdna4_fp8_blockscale import rdna4_fp8_block_scaled_mm

    return rdna4_fp8_block_scaled_mm


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
    m = a.shape[0]
    specialized_layout = _supports_specialized_layout(a, weight, a_scale, weight_scale)
    if specialized_layout and should_use_rdna4_flydsl(m):
        return _load_rdna4_flydsl_mm()(a, weight, a_scale, weight_scale)
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
        try:
            _load_rdna4_flydsl_mm()
        except (ImportError, OSError) as exc:
            return False, f"patched RDNA4 FlyDSL stack is unavailable: {exc}"
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
