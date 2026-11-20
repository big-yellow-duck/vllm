# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL device functions shared by the Qwen GDN kernels."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Float32, T

CONV_WIDTH = 4
HEAD_DIM = 128
VALUES_PER_KEY = 3
WAVE_SIZE = 32
VECTOR_WIDTH = HEAD_DIM // WAVE_SIZE


def lds_barrier():
    """Synchronize a gfx12 workgroup at a shared-memory handoff."""
    # Other waves consume only LDS here. Kernel completion publishes the
    # convolution and recurrent cache stores to the next dispatch.
    local = ir.Attribute.parse('#llvm.mmra_tag<"amdgpu-synchronize-as":"local">')
    release = llvm.fence(llvm.AtomicOrdering.release, syncscope="workgroup")
    release.attributes["llvm.mmra"] = local
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    acquire = llvm.fence(llvm.AtomicOrdering.acquire, syncscope="workgroup")
    acquire.attributes["llvm.mmra"] = local


def buffer_ptr(addr, dtype):
    ptr_type = fx.PointerType.get(dtype, fx.AddressSpace.Global, dtype.width // 8)
    return rocdl.make_buffer_ptr(fx.inttoptr(ptr_type, addr))


def load_fp32_vec(ptr):
    vec_type = T.vec(VECTOR_WIDTH, ptr.element_type.ir_type)
    return fx.Vector(fx.ptr_load(ptr, vec_type)).to(Float32)


def load_conv_weights(ptr, channel):
    return fx.Vector(fx.ptr_load(ptr + channel * CONV_WIDTH, T.vec(CONV_WIDTH, T.bf16)))


def sum_lanes(value, width):
    for shift in range_constexpr(width.bit_length() - 2, -1, -1):
        value = value + value.shuffle_xor(1 << shift, WAVE_SIZE)
    return value


def exp(value):
    return Float32(rocdl.exp2(T.f32, (value * 1.4426950408889634).ir_value()))


def sigmoid(value):
    denominator = Float32(1.0) + exp(-value)
    return Float32(rocdl.rcp(T.f32, denominator.ir_value()))


def rsqrt(value):
    return Float32(rocdl.rsq(T.f32, value.ir_value()))


def conv_result(h0, h1, h2, x, weights):
    # Match the native BF16 multiply used by Triton on RDNA4.
    zero = fx.BFloat16(0.0)
    products = []
    for value, weight in zip((h0, h1, h2, x), weights, strict=True):
        lhs = fx.Vector.from_elements((fx.BFloat16(value), zero), fx.BFloat16)
        rhs = fx.Vector.from_elements((weight, zero), fx.BFloat16)
        product = rocdl.fdot2_bf16_bf16_(
            T.bf16, lhs.ir_value(), rhs.ir_value(), zero.ir_value()
        )
        products.append(Float32(fx.BFloat16(product)))
    result = products[0] + products[1] + products[2] + products[3]
    return fx.BFloat16(result * sigmoid(result))
