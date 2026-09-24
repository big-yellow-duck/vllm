#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen3.8-27B-FP8 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --kv-cache-dtype fp8 \
  --load-format fastsafetensors \
  --attention-backend ROCM_SEGMENTED_ATTN \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":5,"attention_backend":"ROCM_SEGMENTED_ATTN"}'
