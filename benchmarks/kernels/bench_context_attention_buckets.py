# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune/query persistent ROCm context buckets and validate odd query lengths."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.triton_utils import triton
from vllm.v1.attention.ops import prefix_prefill_tuning as tuning
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd


@torch.inference_mode()
def probe(device, heads, kv_heads, dim, page, qlen, context, batch=1):
    config = tuning.get_context_attention_config(
        device, heads, kv_heads, dim, page, batch, qlen, qlen + context, dim**-0.5
    )
    assert config is not None
    q, k, v, kc, vc, table, starts, lengths, one = tuning._make_inputs(
        device, heads, kv_heads, dim, page, batch, qlen, qlen + context
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
    automatic = torch.empty_like(q)
    with patch.object(
        tuning,
        "get_context_attention_config",
        wraps=tuning.get_context_attention_config,
    ) as lookup:
        run(None, automatic)
        assert lookup.call_count == 1
        assert lookup.call_args.args[5:8] == (batch, qlen, qlen + context)
    torch.testing.assert_close(automatic, output, atol=0, rtol=0)
    torch.testing.assert_close(output, baseline, atol=0.01, rtol=0.01)
    relative_l2 = (
        (output.float() - baseline.float()).norm() / baseline.float().norm()
    ).item()
    assert relative_l2 <= 0.005
    # Independent FP32 reference at causal/page/tile boundaries without Q^2 storage.
    rows = sorted({0, min(31, qlen - 1), min(32, qlen - 1), qlen // 2, qlen - 1})
    errors = {"baseline": 0.0, "tuned": 0.0}
    for b in range(batch):
        context_k = (
            kc[table[b].long()]
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, kv_heads, dim)[:context]
        )
        context_v = (
            vc[table[b].long()].permute(0, 3, 1, 2).reshape(-1, kv_heads, dim)[:context]
        )
        full_k = (
            torch.cat((context_k, k[b * qlen : (b + 1) * qlen]))
            .repeat_interleave(heads // kv_heads, dim=1)
            .float()
        )
        full_v = (
            torch.cat((context_v, v[b * qlen : (b + 1) * qlen]))
            .repeat_interleave(heads // kv_heads, dim=1)
            .float()
        )
        indices = [b * qlen + row for row in rows]
        logits = (
            torch.bmm(q[indices].transpose(0, 1).float(), full_k.permute(1, 2, 0))
            * dim**-0.5
        )
        mask = torch.arange(qlen + context, device=device)[None, :] > (
            torch.tensor(rows, device=device)[:, None] + context
        )
        logits.masked_fill_(mask[None, :, :], float("-inf"))
        reference = torch.bmm(logits.softmax(dim=-1), full_v.transpose(0, 1)).transpose(
            0, 1
        )
        for name, actual in (("baseline", baseline), ("tuned", output)):
            per_row_l2 = (actual[indices].float() - reference).flatten(1).norm(
                dim=1
            ) / reference.flatten(1).norm(dim=1)
            errors[name] = max(errors[name], per_row_l2.max().item())
            assert errors[name] <= 0.01, (name, errors[name])
        del full_k, full_v, logits, reference
    cache = torch.empty(256 * 1024 * 1024, device=device, dtype=torch.int8)
    baseline_us = tuning._bench_long_config(
        lambda: run(tuning._DEFAULT, baseline), device, cache
    )
    timings = {}
    for name, launch, out in (
        ("baseline", tuning._DEFAULT, baseline),
        ("tuned", config, output),
    ):
        if baseline_us < 1000:
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
        "batch": batch,
        "query": qlen,
        "context": context,
        "config": config,
        "us": timings,
        "relative_l2_vs_baseline": relative_l2,
        "automatic_matches_saved": True,
        "sampled_fp32_max_row_relative_l2": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--warm-short-engine", action="store_true")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--short-probes", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=524288)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--sample-long-context", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    if args.load_only:

        def forbidden(*args):
            raise RuntimeError("Saved bucket attempted to retune")

        tuning._tune_workload = forbidden
    sampled = [
        (1, 2, 262144),
        (1, 4, 262144),
        (1, 8, 262144),
        (1, 32, 131072),
        (1, 256, 65536),
        (1, 8192, 65536),
        (2, 16, 65536),
        (2, 4096, 32768),
        (4, 8, 32768),
        (4, 2048, 16384),
        (8, 4, 16384),
        (16, 4, 8192),
        (32, 4, 4096),
        (32, 32, 1024),
        (32, 256, 8192),
    ]
    budget = tuning._memory_budget(device)
    raw = list(tuning._workloads(args.max_tokens, args.max_model_len, args.max_seqs))
    planned = list(
        tuning._workloads(
            args.max_tokens,
            args.max_model_len,
            args.max_seqs,
            memory_budget_bytes=budget,
        )
    )
    original_workloads = tuning._workloads
    if args.sample_long_context:
        selected = {(b, q, q + c) for b, q, c in sampled}
        assert selected <= set(planned)

        def filtered(*args, **kwargs):
            return (w for w in original_workloads(*args, **kwargs) if w in selected)

        tuning._workloads = filtered
    started = time.monotonic()
    tuning.warmup_context_attention(
        device,
        torch.bfloat16,
        12,
        2,
        256,
        784,
        0.0625,
        args.max_tokens,
        args.max_model_len,
        args.max_seqs,
    )
    tuning._workloads = original_workloads
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
        "limits": vars(args) | {"output": str(args.output)},
        "plan": {
            "raw": len(raw),
            "scratch_pruned": len(raw) - len(planned),
            "scratch_budget_bytes": budget,
            "feasible": len(planned),
        },
        "cache": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "mtime_ns": path.stat().st_mtime_ns,
        "data": data,
    }
    if args.sample_long_context:
        result["sampled_probes"] = [
            probe(device, 12, 2, 256, 784, q, c, b) for b, q, c in sampled
        ] + [
            probe(device, 12, 2, 256, 784, q, c, b)
            for b, q, c in ((1, 3, 260001), (3, 5, 30001), (31, 3, 4097))
        ]
    if args.probes:
        result["probes"] = [
            probe(device, 12, 2, 256, 784, q, c)
            for q, c in ((5155, 0), (5155, 8192), (32769, 0), (65535, 0))
        ]
    if args.short_probes:
        result["short_probes"] = [
            probe(device, 12, 2, 256, 784, q, c)
            for q in (2, 3, 4, 5, 8, 9, 16, 17)
            for c in (0, 4096, 8192)
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
