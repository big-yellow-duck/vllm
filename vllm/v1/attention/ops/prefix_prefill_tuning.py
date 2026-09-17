# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-only, persistent launch tuning for ROCm context attention."""

import hashlib
import itertools
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
from filelock import FileLock

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)
_TABLES: dict[tuple, dict] = {}
_QUERY_BUCKETS = tuple(2**exponent for exponent in range(1, 17))
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)
_CONTEXT_BUCKETS = (0, *(2**exponent for exponent in range(10, 19)))
_DEFAULT = dict(
    BLOCK_M=128,
    BLOCK_N=64,
    num_unroll_cache=4,
    num_unroll_request=1,
    num_warps=4,
    num_stages=1,
)
_CONFIGS = [
    dict(
        BLOCK_M=m,
        BLOCK_N=n,
        num_unroll_cache=u,
        num_unroll_request=1,
        num_warps=w,
        num_stages=1,
    )
    for m, n, u, w in itertools.product((32, 64, 128), (32, 64), (1, 4), (4, 8))
]


def _key(device, heads, kv_heads, dim, page, scale, kv_dtype=torch.bfloat16):
    return (torch.device(device).index, heads, kv_heads, dim, page, scale, kv_dtype)


def _identity(device, heads, kv_heads, dim, page, scale, kv_dtype=torch.bfloat16):
    props = torch.cuda.get_device_properties(device)
    from triton._C.libtriton import get_cache_invalidating_env_vars

    return {
        "schema": 2,
        "gpu": props.name,
        "arch": props.gcnArchName,
        "compute_units": props.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "compiler_env": get_cache_invalidating_env_vars(),
        "kernel_sha256": hashlib.sha256(
            Path(__file__).with_name("prefix_prefill.py").read_bytes()
        ).hexdigest(),
        "tuner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "page": page,
        "scale": scale,
        "dtype": "bfloat16",
        "kv_dtype": str(kv_dtype),
        "configs": _CONFIGS,
    }


def _memory_budget(device):
    free, total = torch.accelerator.get_memory_info(device)
    available = int(min(free // 2, total // 4))
    return 1 << (available.bit_length() - 1) if available else 0


def _query_lengths(batch, query_len, max_tokens):
    if batch * query_len <= max_tokens:
        return [query_len] * batch
    # A scheduler can mix one long prefill with single-token decode requests.
    return [min(query_len, max_tokens - batch + 1)] + [1] * (batch - 1)


def _scratch_bytes(
    heads, kv_heads, dim, page, batch, query_len, seq_len, max_tokens=None
):
    queries = _query_lengths(batch, query_len, max_tokens or batch * query_len)
    context = max(page, triton.cdiv(seq_len - min(queries), page) * page)
    # Q/output/reference and temporary FP32 comparisons, dense K/V, paged K/V.
    return (
        sum(queries) * dim * (32 * heads + 4 * kv_heads)
        + batch * context * kv_heads * dim * 4
        + 256 * 1024 * 1024
    )


def _workloads(
    max_tokens,
    max_len,
    max_seqs,
    *,
    memory_budget_bytes=None,
    heads=12,
    kv_heads=2,
    dim=256,
    page=784,
    cache_layouts=(),
    cache_budget_bytes=None,
):
    limit = min(max_tokens, max_len)
    queries = sorted({min(q, limit) for q in _QUERY_BUCKETS if limit >= 2})
    for index, q in enumerate(queries):
        previous_query = queries[index - 1] if index else 1
        batch_limit = min(32, max_seqs, max_tokens - previous_query)
        batches = sorted({min(b, batch_limit) for b in _BATCH_BUCKETS})
        if batch_limit < 1:
            continue
        contexts = sorted({min(c, max_len - q) for c in _CONTEXT_BUCKETS})
        for b, c in itertools.product(batches, contexts):
            if (
                memory_budget_bytes is not None
                and _scratch_bytes(heads, kv_heads, dim, page, b, q, q + c, max_tokens)
                > memory_budget_bytes
            ):
                continue
            if cache_budget_bytes is not None:
                cache_bytes = b * sum(
                    triton.cdiv(q + c, block) * size if block else size
                    for block, size in cache_layouts
                )
                if cache_bytes > cache_budget_bytes:
                    continue
            yield b, q, q + c


def _make_inputs(
    device,
    heads,
    kv_heads,
    dim,
    page,
    batch,
    query_len,
    seq_len,
    kv_dtype=torch.bfloat16,
    *,
    query_lengths=None,
):
    generator = torch.Generator(device=device).manual_seed(1234)

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.bfloat16, device=device, generator=generator
        )

    queries = query_lengths if query_lengths is not None else [query_len] * batch
    q = randn(sum(queries), heads, dim)
    k = randn(sum(queries), kv_heads, dim)
    v = randn(sum(queries), kv_heads, dim)
    blocks = max(1, triton.cdiv(seq_len - min(queries), page))
    x = 16 // kv_dtype.itemsize
    scale_value = 1.0 if kv_dtype == torch.bfloat16 else 0.125
    kc = (randn(batch * blocks, kv_heads, dim // x, page, x).float() / scale_value).to(
        kv_dtype
    )
    vc = (randn(batch * blocks, kv_heads, dim, page).float() / scale_value).to(kv_dtype)
    table = torch.arange(batch * blocks, device=device, dtype=torch.int32).view(
        batch, blocks
    )
    starts = torch.tensor(
        [0, *itertools.accumulate(queries)], device=device, dtype=torch.int32
    )
    lengths = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    scale = torch.full((), scale_value, device=device)
    return q, k, v, kc, vc, table, starts, lengths, scale


def _bench_long_config(run, device, cache):
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(3):
        cache.zero_()
        start.record(stream)
        run()
        end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def _tune_workload(
    device, heads, kv_heads, dim, page, scale, kv_dtype, max_tokens, workload
):
    from .prefix_prefill import context_attention_fwd

    b, qlen, slen = workload
    q, k, v, kc, vc, table, starts, lengths, one = _make_inputs(
        device,
        heads,
        kv_heads,
        dim,
        page,
        b,
        qlen,
        slen,
        kv_dtype,
        query_lengths=_query_lengths(b, qlen, max_tokens),
    )
    reference, output = torch.zeros_like(q), torch.zeros_like(q)

    def run(config, out=output):
        context_attention_fwd(
            q,
            k,
            v,
            out,
            "auto" if kv_dtype == torch.bfloat16 else "fp8",
            kc,
            vc,
            table,
            starts,
            lengths,
            slen,
            qlen,
            one,
            one,
            sm_scale=scale,
            skip_decode=True,
            _launch_config=config,
        )

    run(_DEFAULT, reference)
    torch.accelerator.synchronize(device)
    reference_norm = reference.float().norm()
    cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    baseline_us = _bench_long_config(lambda: run(_DEFAULT, reference), device, cache)
    results = []
    for config in _CONFIGS:
        try:
            run(config)
            torch.accelerator.synchronize(device)
            torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
            relative_l2 = (
                (output.float() - reference.float()).norm() / reference_norm
            ).item()
            if relative_l2 > 0.005:
                raise AssertionError(f"Relative L2 error {relative_l2} exceeds 0.5%")
            if baseline_us >= 1000:
                microseconds = _bench_long_config(
                    lambda config=config: run(config), device, cache
                )
            else:
                microseconds = (
                    triton.testing.do_bench(
                        lambda config=config: run(config),
                        warmup=5,
                        rep=20,
                        return_mode="median",
                    )
                    * 1000
                )
            results.append(
                {"config": config, "us": microseconds, "relative_l2": relative_l2}
            )
        except (triton.OutOfResources, triton.CompilationError, AssertionError) as exc:
            logger.warning("Rejected context attention config %s: %s", config, exc)
    if not results:
        raise RuntimeError("No valid ROCm context attention launch configurations")
    winner = min(results, key=lambda item: item["us"])
    logger.info(
        "ROCm context attention autotune: B=%d Q=%d seq=%d best=%s %.2f us",
        b,
        qlen,
        slen,
        winner["config"],
        winner["us"],
    )
    return {
        "workload": list(workload),
        "query_lengths": _query_lengths(b, qlen, max_tokens),
        "best": winner["config"],
        "results": results,
    }


def _save(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def warmup_context_attention(
    device,
    dtype,
    heads,
    kv_heads,
    dim,
    page,
    scale,
    max_tokens,
    max_len,
    max_seqs,
    *,
    memory_budget_bytes=None,
    cache_layouts=(),
    cache_budget_bytes=None,
    kv_dtype=torch.bfloat16,
):
    """Benchmark missing buckets before KV allocation; share results across ranks."""
    props = torch.cuda.get_device_properties(device)
    if (
        dtype != torch.bfloat16
        or kv_dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or not props.gcnArchName.startswith("gfx1201")
        or dim not in (64, 128, 256)
    ):
        return
    key = _key(device, heads, kv_heads, dim, page, scale, kv_dtype)
    if memory_budget_bytes is None:
        memory_budget_bytes = _memory_budget(device)
    workloads = list(
        _workloads(
            max_tokens,
            max_len,
            max_seqs,
            memory_budget_bytes=memory_budget_bytes,
            heads=heads,
            kv_heads=kv_heads,
            dim=dim,
            page=page,
            cache_layouts=cache_layouts,
            cache_budget_bytes=cache_budget_bytes,
        )
    )
    pruned = len(list(_workloads(max_tokens, max_len, max_seqs))) - len(workloads)
    logger.info(
        "ROCm context attention tuning plan: workloads=%d pruned=%d "
        "scratch_budget=%d MiB cache_budget=%s",
        len(workloads),
        pruned,
        memory_budget_bytes // 2**20,
        cache_budget_bytes,
    )
    warmed = {tuple(r["workload"]) for r in _TABLES.get(key, {}).get("records", [])}
    if all(workload in warmed for workload in workloads):
        return
    identity = _identity(device, heads, kv_heads, dim, page, scale, kv_dtype)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(envs.VLLM_CACHE_ROOT) / "rocm_context_attention" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tuned = loaded = 0
    with FileLock(str(path) + ".lock"):
        data = {"identity": identity, "records": []}
        if path.exists():
            try:
                saved = json.loads(path.read_text())
                if saved["identity"] == identity:
                    # Reject malformed/unknown launch parameters before use.
                    for record in saved["records"]:
                        if (
                            len(record["workload"]) != 3
                            or any(
                                type(x) is not int or x < 1 for x in record["workload"]
                            )
                            or record["best"] not in _CONFIGS
                        ):
                            raise ValueError("Invalid launch configuration record")
                    data = saved
            except (ValueError, KeyError, TypeError, AssertionError):
                logger.warning(
                    "Ignoring invalid context attention tuning cache %s", path
                )
        records = {tuple(r["workload"]): r for r in data["records"]}
        for workload in workloads:
            if workload in records:
                loaded += 1
                continue
            records[workload] = _tune_workload(
                device,
                heads,
                kv_heads,
                dim,
                page,
                scale,
                kv_dtype,
                max_tokens,
                workload,
            )
            tuned += 1
            data["records"] = list(records.values())
            _save(path, data)
    _TABLES[key] = data
    logger.info(
        "ROCm context attention autotune ready: tuned=%d loaded=%d "
        "elapsed=%.2fs cache=%s",
        tuned,
        loaded,
        time.monotonic() - start,
        path,
    )


def get_context_attention_config(
    device,
    heads,
    kv_heads,
    dim,
    page,
    batch,
    query_len,
    seq_len,
    scale,
    kv_dtype=torch.bfloat16,
):
    """Look up an already warmed bucket; inference never benchmarks or reads disk."""
    data = _TABLES.get(_key(device, heads, kv_heads, dim, page, scale, kv_dtype))
    if data is None:
        return None
    records = data["records"]
    batches = sorted({r["workload"][0] for r in records})
    queries = sorted({r["workload"][1] for r in records})
    b = next((x for x in batches if x >= batch), None)
    q = next((x for x in queries if x >= query_len), None)
    if b is None or q is None:
        return None
    eligible = [
        r
        for r in records
        if r["workload"][0] == b
        and r["workload"][1] == q
        and r["workload"][2] >= seq_len
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r["workload"][2])["best"]


def warmup_rocm_context_attention(config, device):
    """Tune outside model memory profiling so temporary KV cannot inflate peaks."""
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    specs = get_kv_cache_spec(config)
    layouts = tuple(
        (spec.block_size, spec.page_size_bytes)
        if isinstance(spec, FullAttentionSpec)
        else (0, spec.max_memory_usage_bytes(config))
        for spec in specs.values()
    )
    budget = _memory_budget(device)
    for layer in config.compilation_config.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if isinstance(impl, RocmAttentionImpl):
            impl._warmup_context_attention(
                layer,
                device,
                config.model_config.dtype,
                memory_budget_bytes=budget,
                cache_layouts=layouts,
                cache_budget_bytes=budget,
            )
