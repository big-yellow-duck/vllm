"""Verify the MTP host-only fix preserves compiled kernels and old dispatch."""

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "logs/segmented-serving-2026-10-16"
REL = "vllm/v1/attention/ops/segmented_attention.py"
OLD = RUN / "source-after" / REL
NEW = ROOT / "vllm-segmented-attn-gemma" / REL
NAME = "vllm.v1.attention.ops.segmented_attention"


def load(path):
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    return module


new = load(NEW)
old = load(OLD)
sys.modules[NAME] = new
jit = {}
old_nodes = {n.name: n for n in ast.parse(OLD.read_text()).body if isinstance(n, ast.FunctionDef)}
new_nodes = {n.name: n for n in ast.parse(NEW.read_text()).body if isinstance(n, ast.FunctionDef)}
for name in ("_segmented_kv_upcast", "_segmented_attention_stage", "_segmented_attention_reduce"):
    a, b = getattr(old, name), getattr(new, name)
    assert ast.dump(old_nodes[name]) == ast.dump(new_nodes[name]), name
    assert old_nodes[name].lineno == new_nodes[name].lineno, name
    assert a.cache_key == b.cache_key, (name, a.cache_key, b.cache_key)
    jit[name] = {"ast_equal": True, "line_equal": True, "line": old_nodes[name].lineno, "triton_cache_key": b.cache_key}


class Meta:
    def __init__(self, shape, dtype, last_stride=1):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device("cuda:0")
        self.is_cuda = True
        self.last_stride = last_stride

    def stride(self, axis):
        return self.last_stride if axis in (-1, self.ndim - 1) else math.prod(self.shape[axis + 1:])

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()


scale = torch.ones((), dtype=torch.float32, device="cuda:0")
total = accepted = 0
for dim, ratio, page, dtype, kv_choice, batch, query, stride in itertools.product(
    (64, 128, 256, 512), (1, 2, 4, 8, 16, 17, 0, 3), (16, 32, 1568),
    (torch.bfloat16, torch.float16), range(4), (1, 3), (1, 2), (1, 2),
):
    kv_dtype = (dtype, torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float32)[kv_choice]
    q = Meta((batch * query, 2 * ratio, dim), dtype, stride)
    k = Meta((batch * query, 2, dim), dtype)
    out = Meta(q.shape, dtype)
    kc = Meta((7, page, 2, dim), kv_dtype)
    table = Meta((batch, 9), torch.int32)
    starts = Meta((batch + 1,), torch.int32)
    lengths = Meta((batch,), torch.int32)
    args = (q, k, k, out, kc, kc, table, starts, lengths, scale, scale)
    a, b = old.can_use_segmented_attention(*args), new.can_use_segmented_attention(*args)
    assert a == b, (dim, ratio, page, dtype, kv_dtype, batch, query, stride)
    total += 1
    accepted += int(b)

result = {
    "before_ops_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
    "after_ops_sha256": hashlib.sha256(NEW.read_bytes()).hexdigest(),
    "jit_functions": jit,
    "projected_kv_dispatch": {"cases": total, "identical": total, "accepted": accepted},
}
(RUN / "shared-kv-fix/math-equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
