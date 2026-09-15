#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source this file before Python commands in this checkout.
context_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${context_root}:${context_root}/.venv/lib/python3.14/site-packages:/app/vllm/cmake-build-release/_deps/triton_kernels-src/python/triton_kernels:/opt/python/lib/python3.14/site-packages"
export VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE="${VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE:-1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${context_root}/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${context_root}/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${context_root}/.cache/torchinductor}"
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_CUSTOM_AR=0
export VLLM_USAGE_STATS_SERVER=disabled
export NCCL_PROTO=Simple
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_GDN_DECODE_KERNEL=triton
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1
