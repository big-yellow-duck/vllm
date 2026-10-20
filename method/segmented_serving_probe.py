"""Inspect actual TP worker attention choices outside the timed workload."""

import hashlib
import json
from pathlib import Path


class SegmentedServingInspector:
    def inspect_segmented_serving(self):
        runner = self.model_runner
        config = runner.vllm_config
        layers = []
        for name, layer in config.compilation_config.static_forward_context.items():
            if not hasattr(layer, "attn_backend"):
                continue
            impl = layer.impl
            module = __import__(impl.__class__.__module__, fromlist=["__file__"])
            source = Path(module.__file__)
            layers.append(
                {
                    "name": name,
                    "backend": layer.attn_backend.get_name(),
                    "implementation": impl.__class__.__name__,
                    "heads": getattr(layer, "num_heads", None),
                    "kv_heads": getattr(layer, "num_kv_heads", None),
                    "head_dim": getattr(layer, "head_size", None),
                    "sliding_window": getattr(impl, "sliding_window", None),
                    "kv_cache_dtype": getattr(impl, "kv_cache_dtype", None),
                    "kv_sharing_target": getattr(
                        layer, "kv_sharing_target_layer_name", None
                    ),
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        from vllm.model_executor.warmup import (
            rocm_segmented_attn_autotune_warmup as tuning,
        )
        from vllm.v1.attention.ops import segmented_attention as ops

        tables = []
        for key, data in tuning._TABLES.items():
            tables.append(
                {
                    "key": [str(value) for value in key],
                    "identity": json.loads(
                        json.dumps(data.get("identity"), default=str)
                    ),
                    "records": len(data.get("records", [])),
                    "best_modes": sorted(
                        {
                            str(r["best"].get("kv_upcast", "legacy"))
                            for r in data.get("records", [])
                        }
                    ),
                }
            )
        return {
            "model": config.model_config.model,
            "tp": config.parallel_config.tensor_parallel_size,
            "dtype": str(config.model_config.dtype),
            "kv_cache_dtype": config.cache_config.cache_dtype,
            "attention_layers": layers,
            "tuning_tables": tables,
            "segmented_sources": {
                "ops": {
                    "path": ops.__file__,
                    "sha256": hashlib.sha256(
                        Path(ops.__file__).read_bytes()
                    ).hexdigest(),
                },
                "tuner": {
                    "path": tuning.__file__,
                    "sha256": hashlib.sha256(
                        Path(tuning.__file__).read_bytes()
                    ).hexdigest(),
                },
            },
            "speculative_method": config.speculative_config.method
            if config.speculative_config
            else None,
            "num_speculative_tokens": config.speculative_config.num_speculative_tokens
            if config.speculative_config
            else None,
        }
