# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune/query persistent ROCm context buckets and validate odd query lengths."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.triton_utils import triton
from vllm.v1.attention.ops import prefix_prefill_tuning as tuning
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd


@torch.inference_mode()
def probe(device, heads, kv_heads, dim, page, qlen, context):
    config = tuning.get_context_attention_config(
        device, heads, kv_heads, dim, page, 1, qlen, qlen + context, dim**-0.5
    )
    assert config is not None
    q, k, v, kc, vc, table, starts, lengths, one = tuning._make_inputs(
        device, heads, kv_heads, dim, page, 1, qlen, qlen + context
    )
    baseline, output = torch.empty_like(q), torch.empty_like(q)

    def run(launch, out):
        context_attention_fwd(
            q,
            k,
            v,
            out,
            "auto",
            kc,
            vc,
            table,
            starts,
            lengths,
            qlen + context,
            qlen,
            one,
            one,
            sm_scale=dim**-0.5,
            skip_decode=True,
            _launch_config=launch,
        )

    run(tuning._DEFAULT, baseline)
    run(config, output)
    torch.testing.assert_close(output, baseline, atol=0.01, rtol=0.01)
    relative_l2 = (
        (output.float() - baseline.float()).norm() / baseline.float().norm()
    ).item()
    assert relative_l2 <= 0.005
    # Independent FP32 reference at causal/page/tile boundaries without Q^2 storage.
    rows = sorted({0, min(31, qlen - 1), min(32, qlen - 1), qlen // 2, qlen - 1})
    context_k = (
        kc[table[0].long()].permute(0, 3, 1, 2, 4).reshape(-1, kv_heads, dim)[:context]
    )
    context_v = (
        vc[table[0].long()].permute(0, 3, 1, 2).reshape(-1, kv_heads, dim)[:context]
    )
    full_k = (
        torch.cat((context_k, k)).repeat_interleave(heads // kv_heads, dim=1).float()
    )
    full_v = (
        torch.cat((context_v, v)).repeat_interleave(heads // kv_heads, dim=1).float()
    )
    logits = (
        torch.bmm(q[rows].transpose(0, 1).float(), full_k.permute(1, 2, 0)) * dim**-0.5
    )
    mask = torch.arange(qlen + context, device=device)[None, :] > (
        torch.tensor(rows, device=device)[:, None] + context
    )
    logits.masked_fill_(mask[None, :, :], float("-inf"))
    reference = torch.bmm(logits.softmax(dim=-1), full_v.transpose(0, 1)).transpose(
        0, 1
    )
    errors = {}
    for name, actual in (("baseline", baseline), ("tuned", output)):
        per_row_l2 = (actual[rows].float() - reference).flatten(1).norm(
            dim=1
        ) / reference.flatten(1).norm(dim=1)
        errors[name] = per_row_l2.max().item()
        assert errors[name] <= 0.01, (name, errors[name])
    cache = (
        torch.empty(256 * 1024 * 1024, device=device, dtype=torch.int8)
        if qlen >= 8192
        else None
    )
    timings = {}
    for name, launch, out in (
        ("baseline", tuning._DEFAULT, baseline),
        ("tuned", config, output),
    ):
        if cache is None:
            timings[name] = (
                triton.testing.do_bench(
                    lambda launch=launch, out=out: run(launch, out),
                    warmup=5,
                    rep=20,
                    return_mode="median",
                )
                * 1000
            )
        else:
            timings[name] = tuning._bench_long_config(
                lambda launch=launch, out=out: run(launch, out), device, cache
            )
    return {
        "query": qlen,
        "context": context,
        "config": config,
        "us": timings,
        "relative_l2_vs_baseline": relative_l2,
        "sampled_fp32_max_row_relative_l2": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--warm-short-engine", action="store_true")
    parser.add_argument("--probes", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    if args.load_only:

        def forbidden(*args):
            raise RuntimeError("Saved bucket attempted to retune")

        tuning._tune_workload = forbidden
    started = time.monotonic()
    tuning.warmup_context_attention(
        device, torch.bfloat16, 12, 2, 256, 784, 0.0625, 65536, 65536, 1
    )
    if args.warm_short_engine:
        tuning.warmup_context_attention(
            device, torch.bfloat16, 12, 2, 256, 784, 0.0625, 2048, 2048, 1
        )
    elapsed = time.monotonic() - started
    identity = tuning._identity(device, 12, 2, 256, 784, 0.0625)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention" / f"{digest}.json"
    contents = path.read_bytes()
    data = json.loads(contents)
    assert all(
        r["best"] == min(r["results"], key=lambda x: x["us"])["config"]
        for r in data["records"]
    )
    result = {
        "elapsed_s": elapsed,
        "cache": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
        "data": data,
    }
    if args.probes:
        result["probes"] = [
            probe(device, 12, 2, 256, 784, q, c)
            for q, c in ((5155, 0), (5155, 8192), (32769, 0), (65535, 0))
        ]
    assert (
        tuning.get_context_attention_config(
            device, 12, 2, 256, 784, 1, 65537, 65537, 0.0625
        )
        is None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS buckets={len(data['records'])} startup={elapsed:.2f}s "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
