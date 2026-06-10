#!/usr/bin/env bash
#SBATCH --job-name=qwen35-gdn-decode256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=benchmark_results/slurm-qwen35-gdn-decode256-%j.out
#SBATCH --error=benchmark_results/slurm-qwen35-gdn-decode256-%j.err

set -euo pipefail

. /data/spack/share/spack/setup-env.sh
spack load cuda@12.9.1

cd /home/xsl/pra26/vllm-cuda

OUT_DIR="benchmark_results/gdn_vllm_profile_32k_decode256_${SLURM_JOB_ID}"

.venv/bin/python benchmarks/profile_qwen35_gdn.py \
  --input-len 32768 \
  --output-len 257 \
  --warmup-input-len 32768 \
  --warmup-output-len 8 \
  --max-model-len 33032 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --output-dir "${OUT_DIR}"

.venv/bin/python benchmarks/analyze_qwen35_decode_profile.py "${OUT_DIR}" \
  --phase decode \
  --range-mode all \
  --json-output "${OUT_DIR}/decode256_average_breakdown.json" \
  --markdown-output "${OUT_DIR}/decode256_average_breakdown.md" \
  --top 30
