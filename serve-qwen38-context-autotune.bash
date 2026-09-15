#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
launcher_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=context-attention-env.bash
source "${launcher_root}/context-attention-env.bash"
unset VLLM_ROCM_ENABLE_CUDAGRAPH
cd "${launcher_root}"
exec .venv/bin/python -m vllm.entrypoints.cli.main serve "${MODEL:-Qwen/Qwen3.8-27B-FP8}" \
    --tensor-parallel-size 2 --language-model-only --load-format fastsafetensors \
    --linear-backend triton --attention-backend ROCM_ATTN --kv-cache-dtype auto \
    --max-model-len "${MAX_MODEL_LEN:-2048}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" \
    --max-num-seqs "${MAX_NUM_SEQS:-1}" \
    --gpu-memory-utilization 0.90 --disable-custom-all-reduce \
    --no-enable-prefix-caching --enforce-eager \
    --host 127.0.0.1 --port "${PORT:-8000}" "$@"
