# RDNA4 Octave KV bring-up

Standalone Torch reference, HIP kernels, correctness checks, and measured tuning
results for Octave paged attention on RDNA4. This directory is independent of
the RDNA3 kernel and is not registered as a vLLM serving backend.

## Interface

`TunedHipOctave.build(head_dim, rope_dim)` builds all dispatch paths before graph
capture. It supports dense causal paged decode and chunked prefill with post-RoPE
Q/K, equal logical Q/K/V dimensions, FP16/BF16 inputs, and FP32 attention output.
Logical head dimensions are 1–1024 when separately padded RoPE/NoPE segments fit
1024. Positive integer GQA ratios, MHA, and MQA are supported.

With this directory on `PYTHONPATH`:

```python
from octave_tuned import TunedHipOctave

hip = TunedHipOctave.build(head_dim=128, rope_dim=128)
layout = hip.layout("octave_k3v4")
signs = layout.sign_bits().cuda()
# Cache: uint8 (pages, KV heads, page size, layout.slot_bytes).
# Q/K/V: (tokens, heads, 128), contiguous within each logical head.
# Preallocated FP32 scratch: (tokens, query_heads, splits, hip.head_dim + 2).
# hip.store(...), hip.decode(...), hip.prefill(...).
```

Builders use `hipcc` from `PATH` and content-addressed local build directories.
Preallocate buffers and warm up before graph capture. Automatic split selection
uses tensor metadata and a default 2 GiB workspace budget; explicit integer
splits bypass that budget. Kernels read live sequence lengths on every replay.
Callers supply valid page indices and unique live store slots; negative store
slots are padding. Both page strides and page sizes 16/32 were tested.

The three cache formats are `octave_k3v3`, `octave_k3v4`, and
`octave_k3v3_compact`. The original codebooks and wire precision are preserved;
no FP8 conversion is introduced. The D256/R64 fast path uses native INT8 WMMA
QK where eligible, FP16 WMMA PV, and FP32 softmax/accumulation. Generic geometries
use FP16 WMMA with FP32 range recovery. D<32 uses FP32 directly.

Unequal QK/V MLA, sliding windows, softcap, additional bias/mask semantics, and
model integration require separate work. Gfx1201 is measured; gfx1200 was
compile-tested only.

## Reference and tests

The eager Torch reference loads
[the format source in this checkout](../../../vllm/v1/attention/ops/rocm_octave.py)
directly, without importing compiled vLLM extensions. `OCTAVE_VLLM_ROOT` can
select another checkout. Runtime HIP/layout/sign metadata needs no reference
module. The validated format source has SHA-256
`2f1990ccbcbaeb14e82d7d50d2fc29b96b91f7ef977ad12edfa8f22e5a3c3696`.

From the repository root, using a ROCm Torch environment with pytest installed:

```bash
# CPU reference, geometry, scheduling, and dispatch contracts.
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q benchmarks/kernels/rdna4_octave

# Include ROCm reference tests; serialize with other GPU measurements.
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q benchmarks/kernels/rdna4_octave

# D256/R64 native bring-up checks (fixed legacy source family).
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  .venv/bin/python benchmarks/kernels/rdna4_octave/task_runner.py correctness \
  --candidate benchmarks/kernels/rdna4_octave/tuned_kernel_src
```

`task_runner.py` also provides fixed-split graph timing and profiling. Its
twelve-case D256/R64 suite is an earlier bring-up fixture; the broad release
measurements below use the tuned geometry and automatic scheduling policy.

## Measured results

The final release covers 27 logical dimensions / 32 head/RoPE combinations,
18 GQA ratios through 127, three formats, both activation dtypes/cache strides,
ragged/empty metadata, and graphs. There are 6972 attention comparisons against
independent FP32 Torch attention over the same packed KV. Maximum per-head
relative L2 is 0.364%; 13 full-output endpoint fixtures reach 0.496%, including
8192 query tokens over a 256K prefix. The package suite passed 210 tests.

On one Radeon AI PRO R9700 (`gfx1201`), 346 paired configurations give 1.428x
geometric-mean speedup over the preceding head-dimension release; 64 fresh
held-out cases give 1.546x. The D1024 maximum-prefix endpoint improves from
74.63 s to 23.31 s. These are synthetic per-layer measurements, not model
throughput or task-quality results. Codec loss is separate: the original
D256/R64 synthetic attention fixture showed roughly 13–22% error against
original KV, depending on cache format and operation.

[Final performance, roofline, accuracy, and profiler report](reports/20261109/report.md)
includes timings, charts, tuning decisions, compact validation records, and
measurement provenance. Raw profiler captures and generated binaries remain
outside Git. [Publication notes](reports/20261109/publication.md) distinguish
historical measurements from checkout portability changes.
