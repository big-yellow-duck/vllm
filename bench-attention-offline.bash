#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
variant=${1:?usage: bench-attention-offline.bash vanilla|aiter|ours bf16|fp8 [args]}
kv=${2:?missing KV dtype}
shift 2
case "$variant" in
    vanilla|aiter) source_root=/app/vllm-bench-rdna4-attention-vanilla ;;
    ours) source_root=$task_root ;;
    *) exit 2 ;;
esac
export PYTHONPATH="$source_root:$task_root/benchmarks/kernels:/app/FlyDSL-cvt-f32-fp8/python:/app/tps-rdna4-qwen38-tp2/aiter:$task_root/.venv/lib/python3.14/site-packages:/app/vllm/cmake-build-release/_deps/triton_kernels-src/python/triton_kernels:/opt/python/lib/python3.14/site-packages"
export LD_LIBRARY_PATH="/app/FlyDSL-cvt-f32-fp8/build-fly/python_packages/flydsl/_mlir/_mlir_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=0,1
export NCCL_PROTO=Simple
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USAGE_STATS_SERVER=disabled
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_CUSTOM_AR=0
export VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_AITER_MOE=0 VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_TRITON_ROPE=0
export VLLM_ROCM_USE_AITER_FP8BMM=0 VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_GDN_DECODE_KERNEL=triton VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1
export VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE=0 VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL=0
unset VLLM_ROCM_RDNA4_QWEN38_GDN_LAUNCH VLLM_ROCM_ENABLE_CUDAGRAPH
if [[ $variant == ours ]]; then
    export VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE=1 VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL=1
fi
if [[ $variant == aiter ]]; then
    export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
fi
if [[ $variant != ours ]]; then
    unset VLLM_ROCM_CONTEXT_ATTENTION_AUTOTUNE VLLM_ROCM_USE_RDNA4_SPLITKV_FLYDSL
fi
export VLLM_CACHE_ROOT="$task_root/.cache/offline-attention"
export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE_ROOT/torchinductor"
export FLYDSL_RUNTIME_CACHE_DIR="$VLLM_CACHE_ROOT/flydsl"
output_dir="$task_root/results/offline-attention/${RUN_TAG:-run1}-$variant-$kv"
mkdir -p "$output_dir"
cd "$source_root"
"$task_root/.venv/bin/python" "$task_root/benchmarks/kernels/bench_attention_offline.py" \
    --variant "$variant" --kv "$kv" --output-json "$output_dir/result.json" "$@" \
    2>&1 | tee "$output_dir/run.log"
