#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
validation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${validation_root}"
# shellcheck source=context-attention-env.bash
source context-attention-env.bash
unset VLLM_ROCM_ENABLE_CUDAGRAPH
validation_output="${VALIDATION_OUTPUT:-${validation_root}/results/context-attention-restart}"
mkdir -p "${validation_output}"
.venv/bin/python -u benchmarks/kernels/validate_context_attention_startup.py \
    --output "${validation_output}/first.json" > "${validation_output}/first.log" 2>&1
CONTEXT_TUNING_FORBID=1 .venv/bin/python -u benchmarks/kernels/validate_context_attention_startup.py \
    --output "${validation_output}/restart.json" > "${validation_output}/restart.log" 2>&1
.venv/bin/python - "${validation_output}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
first = json.loads((root / 'first.json').read_text())
restart = json.loads((root / 'restart.json').read_text())
assert first['caches'], 'No context attention cache was saved'
assert first['caches'] == restart['caches'], 'Cache contents or mtime changed'
assert first['outputs'] == restart['outputs'], 'Greedy output changed on restart'
assert 'tuned=0 loaded=' in (root / 'restart.log').read_text(), 'Missing cache-load log'
print(f"PASS: fresh engine reused unchanged cache and produced identical output. Logs: {root}")
PY
