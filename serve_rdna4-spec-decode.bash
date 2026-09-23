export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ROCM_SEGMENTED_ATTN_AUTOTUNE=1

vllm serve Qwen/Qwen3.8-27B-FP8 \
  --tensor-parallel-size 4 \
  --attention-backend ROCM_SEGMENTED_ATTN \
  --kv-cache-dtype fp8 \
  --max-model-len auto \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.938 \
  --speculative-config '{
    "method": "dflash",
    "model": "incoai/Qwen3.8-27B-DFlash2",
    "num_speculative_tokens": 5,
    "attention_backend": "ROCM_SEGMENTED_ATTN"
  }'
