#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for round in 1 2; do
    if [[ $round == 1 ]]; then
        workloads=(vanilla:bf16 aiter:bf16 ours:bf16 vanilla:fp8 aiter:fp8 ours:fp8)
        extra_args=()
    else
        workloads=(ours:fp8 aiter:fp8 vanilla:fp8 ours:bf16 aiter:bf16 vanilla:bf16)
        extra_args=(--reverse)
    fi
    for workload in "${workloads[@]}"; do
        RUN_TAG="round$round" "$task_root/bench-attention-offline.bash" \
            "${workload%:*}" "${workload#*:}" "${extra_args[@]}" "$@"
    done
done
