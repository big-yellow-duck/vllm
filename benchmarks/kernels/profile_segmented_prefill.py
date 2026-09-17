# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short repeatable dispatch sequence for HSA and ATT collection."""

import argparse
import json
from pathlib import Path

import torch
from benchmark_segmented_prefill import make_call, make_inputs

p = argparse.ArgumentParser()
p.add_argument("--backend", default="segmented")
p.add_argument("--queries", type=int, default=32)
p.add_argument("--context", type=int, default=8192)
p.add_argument("--heads", type=int, default=12)
p.add_argument("--kv-heads", type=int, default=2)
p.add_argument("--fp8", action="store_true")
p.add_argument("--page", type=int)
p.add_argument("--config", type=json.loads)
p.add_argument("--dump", type=Path)
a = p.parse_args()
with torch.inference_mode():
    page = a.page or (1568 if a.fp8 else 784)
    data = make_inputs(
        [a.queries],
        [a.context],
        heads=a.heads,
        kv_heads=a.kv_heads,
        fp8=a.fp8,
        page=page,
    )
    run, out, _ = make_call(data, a.backend, a.config)
    for _ in range(6):
        run()
    torch.cuda.synchronize()

    if a.dump and a.backend == "segmented":
        from vllm.v1.attention.ops.segmented_prefill import (
            _segmented_prefill_reduce,
            _segmented_prefill_stage,
        )

        a.dump.mkdir(parents=True, exist_ok=True)
        resources = {}
        for label, fn in [
            ("stage", _segmented_prefill_stage),
            ("reduce", _segmented_prefill_reduce),
        ]:
            cache = fn.device_caches[torch.cuda.current_device()][0]
            for compiled in cache.values():
                resources[label] = dict(
                    shared=compiled.metadata.shared,
                    registers=compiled.n_regs,
                    spills=compiled.n_spills,
                )
                for kind, value in compiled.asm.items():
                    if isinstance(value, str):
                        (a.dump / (label + "." + kind)).write_text(value)
                    elif isinstance(value, bytes):
                        (a.dump / (label + "." + kind)).write_bytes(value)
        (a.dump / "resources.json").write_text(json.dumps(resources, indent=2))
