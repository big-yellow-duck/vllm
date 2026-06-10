#!/usr/bin/env bash
set -euo pipefail

# Bench Qwen3.5-9B long-context latency on a Slurm GPU node.
# Defaults are chosen for a quick single-request latency sweep:
#   input lengths: 4k, 8k, 16k, 32k, using binary K = 1024
#   output length: 4k
#   max model length: 36k

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${ROOT_DIR}/benchmarks/$(basename "${BASH_SOURCE[0]}")"
cd "${ROOT_DIR}"

if [[ "${VLLM_BENCH_ON_SLURM:-0}" != "1" ]]; then
  echo "Submitting benchmark to a Slurm GPU node..."
  exec srun --gres=gpu:1 zsh -lc \
    "cd '${ROOT_DIR}' && VLLM_BENCH_ON_SLURM=1 '${SCRIPT_PATH}'"
fi

MODEL_PATH="${MODEL_PATH:-/data/Competitions/PRA26/Qwen3.5-9B}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results/qwen35_9b_long_context_$(date +%Y%m%d_%H%M%S)}"

INPUT_LENS="${INPUT_LENS:-4096 8192 16384 32768}"
OUTPUT_LEN="${OUTPUT_LEN:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Extra args are split by the shell intentionally so callers can pass normal
# vLLM CLI flags, for example:
#   EXTRA_VLLM_ARGS="--enforce-eager --max-num-seqs 1"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  echo "Run this script from the repository with the uv-managed .venv present." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Results: ${RESULT_DIR}"
echo "Input lengths: ${INPUT_LENS}"
echo "Output length: ${OUTPUT_LEN}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "Batch size: ${BATCH_SIZE}"
echo "Warmup/iters: ${NUM_ITERS_WARMUP}/${NUM_ITERS}"
echo

summary_tsv="${RESULT_DIR}/summary.tsv"
printf "input_len\toutput_len\tbatch_size\tavg_latency_s\tp50_s\tp90_s\tp99_s\trequest_per_s\toutput_tokens_per_s\ttotal_tokens_per_s\tjson\n" \
  > "${summary_tsv}"

for input_len in ${INPUT_LENS}; do
  if (( input_len + OUTPUT_LEN > MAX_MODEL_LEN )); then
    echo "Skipping input_len=${input_len}: input+output exceeds max model len." >&2
    continue
  fi

  output_json="${RESULT_DIR}/latency_in${input_len}_out${OUTPUT_LEN}.json"
  echo "=== input_len=${input_len}, output_len=${OUTPUT_LEN} ==="

  # shellcheck disable=SC2086
  "${PYTHON}" -m vllm.entrypoints.cli.main bench latency \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --trust-remote-code \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --input-len "${input_len}" \
    --output-len "${OUTPUT_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-iters-warmup "${NUM_ITERS_WARMUP}" \
    --num-iters "${NUM_ITERS}" \
    --disable-detokenize \
    --output-json "${output_json}" \
    ${EXTRA_VLLM_ARGS}

  "${PYTHON}" - "${output_json}" "${summary_tsv}" "${input_len}" \
      "${OUTPUT_LEN}" "${BATCH_SIZE}" <<'PY'
import json
import sys

json_path, summary_path, input_len, output_len, batch_size = sys.argv[1:]
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
percentiles = data["percentiles"]
input_len_i = int(input_len)
output_len_i = int(output_len)
batch_size_i = int(batch_size)
avg_latency = float(data["avg_latency"])
request_per_s = batch_size_i / avg_latency
output_tokens_per_s = batch_size_i * output_len_i / avg_latency
total_tokens_per_s = batch_size_i * (input_len_i + output_len_i) / avg_latency
row = [
    input_len,
    output_len,
    batch_size,
    f"{avg_latency:.6f}",
    f"{percentiles['50']:.6f}",
    f"{percentiles['90']:.6f}",
    f"{percentiles['99']:.6f}",
    f"{request_per_s:.6f}",
    f"{output_tokens_per_s:.6f}",
    f"{total_tokens_per_s:.6f}",
    json_path,
]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY

  echo
done

echo "Summary:"
cat "${summary_tsv}"
