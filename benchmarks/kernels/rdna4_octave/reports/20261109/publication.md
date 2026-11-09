# Publication notes

This directory publishes the 9 November 2026 `22_hybrid_scalar` release from
the RDNA4 Octave tuning workspace. It contains the standalone HIP sources,
Torch reference, runtime wrappers, tests, final report, timing/accuracy records,
roof controls, ISA/resource summaries, and compact profiler reviews.

## Source and measurement provenance

The three HIP source files are copied byte for byte from the measured public
package. The original packed wire format comes from this repository's
`vllm/v1/attention/ops/rocm_octave.py`; its hash matches the source used by the
tuning run. The existing RDNA3 implementation and vLLM serving dispatch are
unchanged.

Python publication changes add license headers, apply repository formatting,
discover the containing vLLM checkout for reference imports, and use
`torch.accelerator` for device contexts and synchronization. The geometry,
split policy, native sources, and numerical algorithms are preserved.
The timing CSV uses LF line endings; its values are unchanged.
Checkout validation is recorded in `checkout-validation.json`.

Publication checks pass 189 CPU tests (21 GPU skips), all 210 tests with ROCm,
and 48 additional native decode/prefill fixtures across D1/7/64/96/128/256/512/1024.
The native fixtures cover all formats, both activation dtypes, both cache
strides, automatic splits, and graph replay after query mutation. Repository
pre-commit checks pass. The geometry and scheduling Python ASTs also match the
measured package exactly.

`manifest.json` hashes the files actually included in this publication.
`original-workspace-manifest.json`, `release-inputs.json`, profile reviews,
and raw validation records retain the tuning workspace's original paths and
hashes. Those are historical provenance, not instructions to resolve those
paths in this checkout. Original Python hashes predate publication formatting.

The performance and accuracy numbers in `report.md`, `summary.json`, and the
release JSONL files are the recorded tuning results. They have not been rerun
as a new broad benchmark during publication. The report describes reuse of
unchanged native measurements and the final scalar retiming explicitly.

## Included evidence and reproduction scope

The compact records include the main/held-out/store timings, full-output
checks, range/graph/partition/interleaved probes, per-kernel code identity,
ISA/resources, profile review, and empirical roof controls. The 15 GiB tuning
workspace's raw ATT captures, generated binaries, intermediate candidate
trees, and full stage-generation scripts are excluded. Some paths inside the
original records therefore refer to artifacts outside this publication.

The package README gives commands for the portable correctness suite and
fixed D256/R64 bring-up harness. These commands do not recreate the entire
historical broad tuning campaign. Runtime use remains standalone; model
integration, task-quality evaluation, and production serving measurements
are separate work.
