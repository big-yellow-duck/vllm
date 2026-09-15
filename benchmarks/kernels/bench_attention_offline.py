# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 offline attention comparison with fixed tokens and separate warmups."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def tuning_cache_snapshot():
    root = Path(os.environ["VLLM_CACHE_ROOT"]) / "rocm_context_attention"
    return {
        str(path): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns,
            "kv_dtype": json.loads(path.read_text())["identity"]["kv_dtype"],
            "records": len(json.loads(path.read_text())["records"]),
        }
        for path in sorted(root.glob("*.json"))
    }


def worker_inventory(worker):
    import torch

    config = worker.vllm_config
    layers = []
    for name, layer in config.compilation_config.static_forward_context.items():
        impl = getattr(layer, "impl", None)
        if impl is None:
            continue
        spec = layer.get_kv_cache_spec(config)
        layers.append(
            {
                "name": name,
                "impl": type(impl).__name__,
                "page": spec.block_size,
                "cache_dtype": str(spec.dtype),
                "heads": impl.num_heads,
                "kv_heads": impl.num_kv_heads,
                "dim": impl.head_size,
                "k_scale": layer._k_scale.item(),
                "v_scale": layer._v_scale.item(),
                "query_quantized": layer.query_quant is not None,
                "q_scale": layer._q_scale.item(),
                "fused_rope": impl.fused_rope_kvcache_supported(),
                "fused_qk_norm_rope": getattr(
                    impl, "fused_qk_norm_rope_kvcache_supported", lambda: False
                )(),
            }
        )
    props = torch.cuda.get_device_properties(worker.device)
    import inspect

    unified_files = sorted(
        {
            inspect.getfile(layer.impl.unified_attention)
            for layer in config.compilation_config.static_forward_context.values()
            if hasattr(getattr(layer, "impl", None), "unified_attention")
        }
    )
    return {
        "rank": worker.rank,
        "gpu": props.name,
        "arch": props.gcnArchName,
        "layers": layers,
        "unified_attention_files": unified_files,
    }


def start_probe(worker):
    import torch

    worker._attention_bench_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    )
    worker._attention_bench_profiler.start()


def stop_probe(worker):
    profiler = worker._attention_bench_profiler
    profiler.stop()
    kernels = {}
    for event in profiler.events():
        if str(event.device_type).endswith("CUDA"):
            kernels[event.name] = kernels.get(event.name, 0) + 1
    del worker._attention_bench_profiler
    return {"rank": worker.rank, "gpu_kernel_counts": kernels}


class AttentionBenchWorkerExtension:
    def attention_bench_inventory(self):
        return worker_inventory(self)

    def attention_bench_start_probe(self):
        start_probe(self)

    def attention_bench_stop_probe(self):
        return stop_probe(self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["vanilla", "aiter", "ours"], required=True
    )
    parser.add_argument("--kv", choices=["bf16", "fp8"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    backend = "ROCM_AITER_UNIFIED_ATTN" if args.variant == "aiter" else "ROCM_ATTN"
    settings = dict(
        model="Qwen/Qwen3.8-27B-FP8",
        tensor_parallel_size=2,
        language_model_only=True,
        load_format="fastsafetensors",
        linear_backend="triton",
        attention_backend=backend,
        kv_cache_dtype="fp8" if args.kv == "fp8" else "auto",
        max_model_len=9216,
        max_num_batched_tokens=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.90,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
        seed=1234,
        worker_extension_cls="bench_attention_offline.AttentionBenchWorkerExtension",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
    )
    source = Path(vllm.__file__).parent.parent
    result = {
        "variant": args.variant,
        "kv": args.kv,
        "settings": settings,
        "source": str(source),
        "commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("VLLM_ROCM_")
            or k
            in (
                "NCCL_PROTO",
                "HIP_VISIBLE_DEVICES",
                "VLLM_GDN_DECODE_KERNEL",
                "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
                "VLLM_CACHE_ROOT",
            )
        },
        "rows": [],
        "tuning_cache_before_startup": tuning_cache_snapshot(),
    }
    before = time.perf_counter()
    llm = LLM(**settings)
    result["engine_startup_s"] = time.perf_counter() - before
    result["tuning_cache_after_startup"] = tuning_cache_snapshot()
    result["workers"] = llm.collective_rpc("attention_bench_inventory")
    save(args.output_json, result)
    if args.variant == "ours":
        expected_kv_dtype = (
            "torch.float8_e4m3fn" if args.kv == "fp8" else "torch.bfloat16"
        )
        assert any(
            cache["kv_dtype"] == expected_kv_dtype and cache["records"] > 0
            for cache in result["tuning_cache_after_startup"].values()
        )
    expected_impl = (
        "RocmAiterUnifiedAttentionImpl"
        if args.variant == "aiter"
        else "RocmAttentionImpl"
    )
    assert all(
        layer["impl"] == expected_impl
        for worker in result["workers"]
        for layer in worker["layers"]
    )
    assert all(
        not layer["fused_rope"] and not layer["fused_qk_norm_rope"]
        for worker in result["workers"]
        for layer in worker["layers"]
    )

    def prompts(input_len, batch):
        return [
            TokensPrompt(
                prompt_token_ids=[
                    1000 + ((i * 7919 + b * 101) % 9000) for i in range(input_len)
                ]
            )
            for b in range(batch)
        ]

    def generate(input_len, output_len, batch):
        return llm.generate(
            prompts(input_len, batch),
            SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=output_len,
                detokenize=False,
            ),
            use_tqdm=False,
        )

    if args.probe:
        generate(8192, 4, 1)
        llm.collective_rpc("attention_bench_start_probe")
        generate(8192, 4, 1)
        result["profile"] = llm.collective_rpc("attention_bench_stop_probe")
        save(args.output_json, result)

    pairs = [(128, 32), (128, 256), (2048, 32), (2048, 256), (8192, 32), (8192, 256)]
    if args.smoke:
        pairs = [(128, 4)]
    workloads = [(i, o, b) for i, o in pairs for b in (1, 2, 4)]
    random.Random(1234).shuffle(workloads)
    if args.reverse:
        workloads.reverse()
    for input_len, output_len, batch in workloads:
        for _ in range(args.warmups):
            generate(input_len, output_len, batch)
        row = {
            "input_len": input_len,
            "output_len": output_len,
            "batch": batch,
            "samples": [],
        }
        reference_ids = None
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = generate(input_len, output_len, batch)
            elapsed = time.perf_counter() - start
            ids = [output.outputs[0].token_ids for output in outputs]
            assert len(ids) == batch and all(len(x) == output_len for x in ids)
            if reference_ids is None:
                reference_ids = ids
            row["samples"].append(
                {
                    "latency_s": elapsed,
                    "output_tokens_per_s": batch * output_len / elapsed,
                    "ids": ids,
                    "matches_first_sample": ids == reference_ids,
                }
            )
        row["median_latency_s"] = statistics.median(
            sample["latency_s"] for sample in row["samples"]
        )
        row["median_output_tokens_per_s"] = batch * output_len / row["median_latency_s"]
        row["output_sha256"] = hashlib.sha256(
            json.dumps(reference_ids).encode()
        ).hexdigest()
        result["rows"].append(row)
        save(args.output_json, result)
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
    print(f"Saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
