# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the public prototype API with fragmented pages and eviction sizes."""

import json
from pathlib import Path

import torch
from eviction import ReadEviction
from fixtures import inputs
from kernel import prepare_attention
from tune import error, measure, reference


def main():
    records = []
    data = inputs([2], [65536])
    data["context"] = 65536
    run, out = prepare_attention(
        data["q"],
        data["kc"],
        data["vc"],
        data["kd"],
        data["vd"],
        data["table"],
        65536,
    )
    for _ in range(10):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for mib in (128, 256, 512):
        evict = ReadEviction(mib * 1024**2)
        for seed in (0, 17, 59, 109, 317):
            torch.manual_seed(seed)
            if seed:
                data["table"][0].copy_(
                    torch.randperm(data["table"].numel(), device="cuda")
                )
            else:
                data["table"][0].copy_(
                    torch.arange(data["table"].numel(), device="cuda")
                )
            data["q"].normal_()
            data["kd"].normal_()
            data["vd"].normal_()
            graph.replay()
            torch.cuda.synchronize()
            e = error(out, reference(data))
            assert e < 0.01 and torch.isfinite(out).all()
            latency = measure(graph, evict, True, samples=30)
            record = dict(
                eviction_mib=mib,
                seed=seed,
                fragmented=bool(seed),
                error=e,
                read_eviction_us=latency,
                ideal_roof_pct=209.76 / latency * 100,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    Path("robustness.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
