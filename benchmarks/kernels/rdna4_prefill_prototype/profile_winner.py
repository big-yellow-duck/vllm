# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
from pathlib import Path

import torch
from fixtures import inputs
from kernel import make_call
from validate import Q2_CONFIG, baseline

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["prototype", "aiter"], default="prototype")
p.add_argument("--dump", type=Path)
a = p.parse_args()
data = inputs([2], [65536], "bf16", 784)
if a.backend == "prototype":
    run, out = make_call(data, Q2_CONFIG, 65536)
else:
    run, out, _ = baseline(data, 2, 65536, "aiter")
for _ in range(6):
    run()
torch.cuda.synchronize()
if a.dump and a.backend == "prototype":
    a.dump.mkdir(exist_ok=True, parents=True)
    res = {}
    for name, k in [("stage", run.compiled), ("reduce", run.reduce_compiled)]:
        res[name] = dict(
            shared=k.metadata.shared, registers=k.n_regs, spills=k.n_spills
        )
        for kind in ("ttir", "ttgir", "llir", "amdgcn", "hsaco"):
            obj = k.asm[kind]
            path = a.dump / (name + "." + kind)
            if isinstance(obj, bytes):
                path.write_bytes(obj)
            else:
                path.write_text(obj)
        res[name]["hsaco_sha256"] = hashlib.sha256(k.asm["hsaco"]).hexdigest()
    (a.dump / "resources.json").write_text(json.dumps(res, indent=2) + "\n")
