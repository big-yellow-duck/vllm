# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness, graph timing, and profiling for the RDNA4 HIP Octave seed."""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from octave_hip import FORMATS, HEAD, HipOctave, build_library
from octave_reference import (
    OctaveFormat,
    format_ops,
    reference_cache_store,
    reference_decode,
    reference_dequantize,
    reference_prefill,
)

ROOT = Path(__file__).resolve().parent
K3, K4, COMPACT = FORMATS
SCALE = 1 / 16
ATTENTION_LIMIT = 0.02
STORE_LIMIT = 0.01
DECODE_REGRESSION_LIMIT = 0.05


def verify_frozen_inputs():
    manifest = os.environ.get("OCTAVE_FROZEN_META")
    if not manifest:
        return
    hashes = json.loads(Path(manifest).read_text())["sha256"]
    for relative_path, expected in hashes.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen GEAK input changed: {relative_path}")
    expected = hashes["oracle/vllm/v1/attention/ops/rocm_octave.py"]
    if hashlib.sha256(Path(format_ops.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Torch loaded a different Octave format source")


@dataclass
class PagedInputs:
    key: torch.Tensor
    value: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    slots: torch.Tensor
    signs: torch.Tensor
    randn: Callable[[int, int], torch.Tensor]


def paged_inputs(lengths, page_size, kv_heads, fmt, dtype, token_major, seed):
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(seed)
    counts = [(n + page_size - 1) // page_size for n in lengths]
    pages = torch.randperm(sum(counts) + 2, generator=generator, device=device)
    table = torch.full(
        (len(lengths), max(counts) + 3), -1, device=device, dtype=torch.int32
    )[:, : max(counts)]
    slots, offset = [], 0
    for req, (length, count) in enumerate(zip(lengths, counts)):
        ids = pages[offset : offset + count]
        table[req, :count] = ids
        positions = torch.arange(length, device=device)
        slots.append(ids[positions // page_size] * page_size + positions % page_size)
        offset += count
    if token_major:
        cache = torch.full(
            (pages.numel(), page_size, kv_heads, fmt.slot_bytes),
            255,
            device=device,
            dtype=torch.uint8,
        ).transpose(1, 2)
    else:
        cache = torch.full(
            (pages.numel(), kv_heads, page_size, fmt.slot_bytes),
            255,
            device=device,
            dtype=torch.uint8,
        )

    def randn(tokens, heads):
        # Leave padding after every head to exercise real projection strides.
        return torch.randn(
            tokens, heads, HEAD + 8, generator=generator, device=device, dtype=dtype
        )[..., :HEAD]

    key, value = randn(sum(lengths), kv_heads), randn(sum(lengths), kv_heads)
    key[..., 100] *= 8
    return PagedInputs(
        key,
        value,
        cache,
        table,
        torch.cat(slots),
        format_ops.sign_bits(HEAD).to(device),
        randn,
    )


def int32(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def relative_error(out, ref):
    diff = (out - ref).norm().item()
    denominator = ref.norm().item()
    return diff / denominator if denominator else (0 if diff == 0 else math.inf)


def graph_for(call, repetitions):
    call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            call()
    return graph


def check_output(out, ref, label, limit):
    # Catch a bad token independently of its influence on the global norm.
    diff = (out - ref).norm(dim=-1).norm(dim=-1)
    denominator = ref.norm(dim=-1).norm(dim=-1)
    errors = torch.where(denominator > 0, diff / denominator, diff)
    if torch.count_nonzero(diff[denominator == 0]).item():
        raise AssertionError(f"{label}: a zero reference row produced nonzero output")
    error = errors.max().item()
    if not math.isfinite(error) or error >= limit:
        raise AssertionError(
            f"{label}: max per-token relative L2 {error:.6g} >= {limit}"
        )
    return error


def correctness_case(hip, cache_dtype, dtype, layout, page_size, group, seed, long):
    fmt = OctaveFormat.from_cache_dtype(cache_dtype, HEAD, 64)
    lengths = [1, 17, 2, 4097 if long else 113]
    chunks = [1, 4, 0, 129 if long else 19]
    kv_heads = 2
    inp = paged_inputs(lengths, page_size, kv_heads, fmt, dtype, layout, seed)
    ref_cache = inp.cache.clone()
    reference_cache_store(inp.key, inp.value, ref_cache, inp.slots, fmt)
    # Write zero to one live row, leave one mapped row padded, and retain guards.
    inp.key[0].zero_()
    inp.value[0].zero_()
    write_slots = inp.slots.clone()
    write_slots[2] = -1
    store_expected = inp.cache.clone()
    reference_cache_store(inp.key, inp.value, store_expected, write_slots, fmt)
    hip.store(inp.key, inp.value, inp.cache, write_slots, inp.signs, cache_dtype)
    live = write_slots[write_slots >= 0]
    pages, offsets = live // page_size, live % page_size
    mask = torch.ones(inp.cache.shape[0], page_size, device="cuda", dtype=torch.bool)
    mask[pages, offsets] = False
    if not torch.equal(
        inp.cache.transpose(1, 2)[mask], store_expected.transpose(1, 2)[mask]
    ):
        raise AssertionError("Cache store touched an unmapped slot")
    actual_bytes, reference_bytes = (
        inp.cache[pages, :, offsets],
        store_expected[pages, :, offsets],
    )
    if not torch.equal(actual_bytes[0], reference_bytes[0]):
        raise AssertionError("Zero input cache slot bytes differ")
    actual_k, actual_v = reference_dequantize(actual_bytes, fmt)
    ref_k, ref_v = reference_dequantize(reference_bytes, fmt)
    store_error = max(relative_error(actual_k, ref_k), relative_error(actual_v, ref_v))
    if not math.isfinite(store_error) or store_error >= STORE_LIMIT:
        raise AssertionError(
            f"Cache store relative L2 {store_error:.6g} >= {STORE_LIMIT}"
        )

    # Test attention against the same Torch-packed cache, including a padding
    # row with an invalid request ID, shuffled pages, and causal MTP lengths.
    reqs, lens = [0, 1, 1, 1], [1, 13, 14, 15]
    reqs += [3] * 4 + [len(lengths) + 100]
    lens += list(range(lengths[3] - 3, lengths[3] + 1)) + [0]
    query = inp.randn(len(reqs), kv_heads * group).mul_(2)
    q_to_req, q_to_klen = int32(reqs), int32(lens)
    d_ref = reference_decode(
        query, ref_cache, inp.table, q_to_req, q_to_klen, fmt, SCALE
    )
    zero_lens = torch.zeros_like(q_to_klen)
    zero = hip.decode(
        query,
        ref_cache,
        inp.table,
        q_to_req,
        zero_lens,
        inp.signs,
        cache_dtype,
        splits=7,
    )
    if torch.count_nonzero(zero).item():
        raise AssertionError("An all-padding decode must output exact zeros")

    starts, key_parts, value_parts, offset = [0], [], [], 0
    for length, chunk in zip(lengths, chunks):
        end = offset + length
        key_parts.append(inp.key[end - chunk : end])
        value_parts.append(inp.value[end - chunk : end])
        starts.append(starts[-1] + chunk)
        offset = end
    q_chunk = inp.randn(starts[-1], kv_heads * group).mul_(2)
    k_chunk, v_chunk = torch.cat(key_parts), torch.cat(value_parts)
    query_starts, seq_lens = int32(starts), int32(lengths)
    p_ref = reference_prefill(
        q_chunk,
        k_chunk,
        v_chunk,
        ref_cache,
        inp.table,
        query_starts,
        seq_lens,
        fmt,
        SCALE,
    )
    decode = partial(
        hip.decode,
        query,
        ref_cache,
        inp.table,
        q_to_req,
        q_to_klen,
        inp.signs,
        cache_dtype,
    )
    prefill = partial(
        hip.prefill,
        q_chunk,
        k_chunk,
        v_chunk,
        ref_cache,
        inp.table,
        query_starts,
        seq_lens,
        inp.signs,
        cache_dtype,
    )
    errors = {"decode": 0.0, "prefill": 0.0}
    for splits in (1, 7) if not long else (17,):
        for name, operation, ref in (
            ("decode", decode, d_ref),
            ("prefill", prefill, p_ref),
        ):
            out = torch.empty_like(ref)
            scratch = torch.empty((*ref.shape[:2], splits, HEAD + 2), device="cuda")
            call = partial(operation, out=out, scratch=scratch, splits=splits)
            if long or splits == 7:
                graph = graph_for(call, 1)
                out.fill_(float("nan"))
                graph.replay()
            else:
                call()
            err = check_output(out, ref, name, ATTENTION_LIMIT)
            errors[name] = max(errors[name], err)
    return {"store": store_error, **errors}


def correctness(hip, suite):
    cases = []
    for name in FORMATS:
        for dtype in (torch.float16, torch.bfloat16):
            configs = [(False, 16, 1), (True, 32, 6)]
            if suite == "smoke":
                configs = configs[:1]
            for token_major, page, group in configs:
                error = correctness_case(
                    hip, name, dtype, token_major, page, group, 17, False
                )
                cases.append(
                    {
                        "format": name,
                        "dtype": str(dtype),
                        "page": page,
                        "group": group,
                        "token_major": token_major,
                        **error,
                    }
                )
    if suite == "full":
        for name, group in ((K3, 4), (K4, 8), (COMPACT, 2)):
            error = correctness_case(
                hip, name, torch.bfloat16, True, 16, group, 59, True
            )
            cases.append(
                {
                    "format": name,
                    "dtype": "torch.bfloat16",
                    "long": True,
                    "group": group,
                    **error,
                }
            )
    return {
        "status": "pass",
        "cases": cases,
        "max_relative_l2": {
            op: max(case[op] for case in cases) for op in ("store", "decode", "prefill")
        },
    }


@dataclass(frozen=True)
class PerfCase:
    case: str
    op: str
    fmt: str
    dtype: torch.dtype
    length: int
    query: int = 1
    batch: int = 1
    splits: int = 1


PERF_CASES = (
    PerfCase("store_1_k3v3", "store", K3, torch.bfloat16, 1),
    PerfCase("store_128_compact", "store", COMPACT, torch.float16, 128),
    PerfCase("store_2048_k3v4", "store", K4, torch.bfloat16, 2048),
    PerfCase("decode_1x128_k3v3", "decode", K3, torch.bfloat16, 128),
    PerfCase("decode_1x1024_k3v4", "decode", K4, torch.float16, 1024, splits=8),
    PerfCase(
        "decode_1x8192_compact", "decode", COMPACT, torch.bfloat16, 8192, splits=32
    ),
    PerfCase("decode_4x8192_k3v3", "decode", K3, torch.bfloat16, 8192, 4, splits=32),
    PerfCase(
        "decode_b8x2048_k3v4", "decode", K4, torch.float16, 2048, batch=8, splits=8
    ),
    PerfCase("decode_1x32768_k3v3", "decode", K3, torch.bfloat16, 32768, splits=128),
    PerfCase("prefill_129_no_prefix_k3v3", "prefill", K3, torch.float16, 129, 129),
    PerfCase(
        "prefill_129x1024_compact",
        "prefill",
        COMPACT,
        torch.bfloat16,
        1153,
        129,
        splits=4,
    ),
    PerfCase(
        "prefill_256x4096_k3v4", "prefill", K4, torch.float16, 4352, 256, splits=8
    ),
)


def benchmark_calls(baseline, candidate, case):
    fmt = OctaveFormat.from_cache_dtype(case.fmt, HEAD, 64)
    inp = paged_inputs([case.length] * case.batch, 16, 2, fmt, case.dtype, True, 251)
    if case.op == "store":

        def bind(hip):
            return lambda: hip.store(
                inp.key, inp.value, inp.cache, inp.slots, inp.signs, case.fmt
            )
    else:
        baseline.store(inp.key, inp.value, inp.cache, inp.slots, inp.signs, case.fmt)
        query = inp.randn(case.query * case.batch, 12)
        out = torch.empty(query.shape, dtype=torch.float32, device="cuda")
        scratch = torch.empty((*query.shape[:2], case.splits, HEAD + 2), device="cuda")
        if case.op == "decode":
            mapping = int32([i for i in range(case.batch) for _ in range(case.query)])
            lengths = int32(
                list(range(case.length - case.query + 1, case.length + 1)) * case.batch
            )

            def bind(hip):
                return lambda: hip.decode(
                    query,
                    inp.cache,
                    inp.table,
                    mapping,
                    lengths,
                    inp.signs,
                    case.fmt,
                    splits=case.splits,
                    out=out,
                    scratch=scratch,
                )
        else:
            starts = int32([i * case.query for i in range(case.batch + 1)])
            lengths = int32([case.length] * case.batch)
            k = inp.key.view(case.batch, case.length, 2, HEAD)[
                :, -case.query :
            ].reshape(-1, 2, HEAD)
            v = inp.value.view(case.batch, case.length, 2, HEAD)[
                :, -case.query :
            ].reshape(-1, 2, HEAD)

            def bind(hip):
                return lambda: hip.prefill(
                    query,
                    k,
                    v,
                    inp.cache,
                    inp.table,
                    starts,
                    lengths,
                    inp.signs,
                    case.fmt,
                    splits=case.splits,
                    out=out,
                    scratch=scratch,
                )

    return bind(baseline), bind(candidate)


def timings(graphs, inner):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    generator = random.Random(83)
    samples = [[], []]
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    for _ in range(5):
        for i in generator.sample(range(2), 2):
            start.record()
            graphs[i].replay()
            stop.record()
            stop.synchronize()
            samples[i].append(start.elapsed_time(stop) / inner)
    return [statistics.median(s) for s in samples]


def performance(baseline, candidate, selected):
    cases = []
    for case in selected:
        base_call, candidate_call = benchmark_calls(baseline, candidate, case)
        inner = 3 if case.op == "prefill" else 12
        graphs = [graph_for(call, inner) for call in (base_call, candidate_call)]
        base_ms, candidate_ms = timings(graphs, inner)
        if not all(math.isfinite(ms) and ms > 0 for ms in (base_ms, candidate_ms)):
            raise RuntimeError(f"Invalid latency on {case.case}")
        cases.append(
            {
                "case": case.case,
                "op": case.op,
                "baseline_ms": base_ms,
                "optimized_ms": candidate_ms,
                "speedup": base_ms / candidate_ms,
            }
        )
    return {
        "cases": cases,
        "baseline_geomean_ms": statistics.geometric_mean(
            c["baseline_ms"] for c in cases
        ),
        "speedup_geomean": statistics.geometric_mean(c["speedup"] for c in cases),
        "decode_guard_pass": all(
            c["optimized_ms"] <= c["baseline_ms"] * (1 + DECODE_REGRESSION_LIMIT)
            for c in cases
            if c["op"] == "decode"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("compile", "correctness", "performance", "profile")
    )
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--case", choices=[case.case for case in PERF_CASES])
    parser.add_argument("--candidate", type=Path, default=ROOT / "kernel_src")
    parser.add_argument("--baseline", type=Path, default=ROOT / "baseline_src")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arch", choices=("gfx1200", "gfx1201"), default="gfx1201")
    args = parser.parse_args()
    verify_frozen_inputs()
    if (
        os.environ.get("OCTAVE_FROZEN_META")
        and args.baseline.resolve() != ROOT / "baseline_src"
    ):
        raise ValueError("GEAK timing must use the frozen HIP seed in baseline_src/")
    lib = build_library(args.candidate, ROOT / "build", args.arch)
    if args.mode == "compile":
        print(lib)
        return
    prop = torch.cuda.get_device_properties(0)
    if prop.gcnArchName.split(":")[0] != args.arch:
        raise RuntimeError(f"Requested {args.arch} but device is {prop.gcnArchName}")
    hip = HipOctave(lib)
    if args.mode == "correctness":
        result = correctness(hip, args.suite)
    else:
        baseline_src = args.baseline if args.baseline.is_dir() else args.candidate
        baseline = HipOctave(build_library(baseline_src, ROOT / "build", args.arch))
        cases = [
            case for case in PERF_CASES if args.case is None or case.case == args.case
        ]
        if args.suite == "smoke" and args.case is None:
            cases = [cases[1], cases[5], cases[10]]
        if args.mode == "profile":
            _, call = benchmark_calls(baseline, hip, cases[0])
            for _ in range(8):
                call()
            torch.accelerator.synchronize()
            result = {"profile_case": cases[0].case}
        else:
            result = performance(baseline, hip, cases)
    result["gpu"] = {
        "name": prop.name,
        "arch": prop.gcnArchName,
        "visible": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    result["library"] = str(lib)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
