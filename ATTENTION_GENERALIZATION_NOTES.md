# Four-hour attention implementation notes

Start: 2026-09-15 15:25:18 UTC. Deadline: 19:25:18 UTC.

## Scope and completion evidence

Implement the agreed layered approach: a useful general Triton short-prefill
path, tuned shape/range routing with existing direct/long-prefill and decode
fallbacks, a profile-driven FlyDSL specialization experiment, and promotion
only where correctness and matched kernel/engine measurements support it.
Keep the full scope visible; a single-shape microbenchmark is not completion.

Required evidence:

- Ragged batches, arbitrary prefixes/tails, causal fresh current K/V, BF16/FP16
  and required FP8 cache behavior, changed-input graph replay, bounded workspace.
- Disjoint ownership with decode and long-prefill paths, unsupported-feature
  fallback, representative shape-range tuning and regression checks.
- Compare to selected and additionally tuned AITER; retain read/write eviction
  and reuse results. No altered FLOP/byte accounting to manufacture wins.
- Profile and implement a concrete FlyDSL candidate using existing SplitKV
  code as a reference; benchmark it against Triton and AITER. Promote only a
  supported winning range; document any rejected candidate and remaining gap.
- Engine integration checks and model/output evaluation appropriate to changes.
- Reproducible code, raw data, notes, and a requirement-by-requirement audit.

## Initial authoritative state

vLLM branch `perf/rdna4-prefill-autotune-splitkv`, HEAD 338603e46b.
Existing reports and isolated prototype are untracked user-session work.
Production has tuned context prefill and optional FlyDSL SplitKV decode;
the isolated short-prefill prototype has not yet been integrated.
AITER is the patched checkout at `/app/tps-rdna4-qwen38-tp2/aiter`.
Use the requested `.venv` and the established combined source PYTHONPATH.
GPU jobs run sequentially; no subagents are authorized.

## 15:29 — implementation decisions

New segmented Triton path owns eligible short-prefill rows, while decode
remains on its current route and long-prefill remains in the existing context
kernel. Both kernels must enforce the same per-request query-length boundary.
Use GPU query-start/sequence metadata without host synchronization. Support
fresh current K/V independently of the cached copy. Empty segments require
explicit zero partials and negative-infinity LSE; masked cache tails must not
load beyond their block-table row. Workspace is bounded by short query slots,
not total tokens in a mixed long-prefill batch.

The prior 80% result applies to Q2/C65536 under read eviction. Q32/C8192 is
already faster than default AITER but below its roof, making it a candidate
for a specialization study. Preserve those facts, including dirty-cache losses.

## 15:40 — first general implementation and test design

Added a GPU-metadata-driven segmented kernel with arbitrary prefix/tail masks,
FP8 per-tensor scaling, strided Q/current K/V, and an explicit empty-segment
representation. Added opt-in integration and startup workspace reservation.
The initial query limit is 128; decode ownership follows `skip_decode`.

Tests extend the existing prefix-prefill suite. Contract: eligible output rows
must equal causal FP32 attention using cached prefix plus fresh current K/V;
non-owned rows must remain untouched. Guard against stale/overlapping writes,
empty segments, page-table OOB, FP8 scale mistakes, and route changes during
graph replay. Cheapest meaningful check is direct GPU kernel/public-wrapper
comparison, with CPU-only workspace/routing boundary tests where sufficient.
Existing cache-packing and context input helpers are reused.

## 15:55 — first range evidence and FlyDSL candidate

All six ragged BF16/FP16/FP8 cases pass, including changed graph ownership.
The CPU workspace lock test covers every batch 1..32 and query cap 1..128
for D128/G16 and D256/G6 without growth. Five persistent context-tuner tests
pass after adding scheduler-realizable ragged fixtures for large query buckets.
The previous uniform-only generation excluded B4/Q8192 even when a batch of
[8189,1,1,1] fits the 8192-token scheduler budget. Cache memory accounting now
uses the smallest query length's prefix; decode output rows are initialized
consistently for candidate comparison.

First matrix: 60 shapes, 120 successful backend/reference checks; BF16 and FP8
KV, Q2/8/32/64/128, C0/512/8192, B1/4. Default heuristic still loses some small
BF16 cases. The FP8 AITER rows use BF16 Q for matched input precision, which
is not its native engine FP8-Q path; add that comparison before claiming
native FP8 performance superiority. Raw matrix-v1.json is in the result dir.

General masking initially regressed Q32/C8192 to 127 us. Peeling the masked
prefix tail restores 91 us in the strided, shuffled-page fixture, compared
with 79 us in the earlier restricted contiguous prototype. The four-wave
FlyDSL candidate compiles and passes Q32 correctness, but first timing is
233 us, so it is not eligible for promotion. It uses direct register loads
for Q/K/V, separate scores and weights in LDS, and balanced four-wave QK/PV.
Inspect resources and instruction waits before making the next change.

## HSA/ATT of first FlyDSL candidate

Stage launches 192 workgroups of 128 threads; HSA reports 216 VGPRs,
6656 bytes LDS and zero scratch. The ATT analyzer does not recognize RDNA4:
it labels gfx942 and classifies s_wait_loadcnt as "other". Ignore its occupancy
estimate; actual architecture is gfx1201. Raw top instructions are overwhelmingly
s_wait_loadcnt and s_wait_loadcnt_dscnt. This supports grouping independent
K/V loads ahead of their WMMA consumers. Current sequential load/WMMA pairs
expose global-memory latency despite spill-free register allocation. Preserve
this baseline capture at att-fly-v3 before experimenting with grouped loads.

## 16:12 — resource-guided search changes

The exhaustive initial launch search checkpoint contains 259 completed
candidates. Resume skips these exact case/config pairs. Add eight-warp launches
because general Q32 uses scratch where the restricted prototype was spill-free.
Prune screens with fewer than 16 or more than 768 stage workgroups, and BM64
when fewer than 17 useful flattened rows exist; these are bounded search choices,
not proof of global optimality. Include split count 2, missing from the first
coarse grid. Original results are preserved in tune-triton-v2.log/json.

FlyDSL v4 (grouped K loads) improved Q32 from 233 to 176 us. Grouping all four
V fragments as well regressed to 190 us; HSA reports 256 VGPRs and 120 bytes
scratch. The 32-row candidate passes correctness but takes 219 us with 376
bytes scratch and 30 KiB LDS. Both remain experiments, not production routes.
A further candidate streams Q fragments in groups of four to reduce their
live register footprint; it has not yet run. D128 also passes in both candidates.

Prepared a detached baseline worktree at 338603e46b for engine comparisons.
Extended the existing engine benchmark with configurable scheduler token budget,
explicit workload lists, source hashes, short-prefill route probes and a small
12-prompt deterministic output/log-probability diagnostic. This is a regression
check, not a broad model-quality benchmark. Engine runs are still outstanding.

## 16:28 — large-cache address overflow found and fixed

Full prefix suite and isolated nonstandard-page test aborted in HSA. Static
analysis identifies an actual physical-allocation OOB in the new stage:
block_id(int32) * K0(int32) can exceed 2**31 elements before pointer addition.
For 640 physical blocks, page544, HK64, D128, the cache has 2,852,126,720
BF16 elements; some valid block bases wrap negative. Existing context code
already casts physical block IDs to int64. The new stage now does the same;
FlyDSL experiments also widen physical IDs before stride multiplication.

Added a focused address-boundary regression in the existing test file: two
logical prefix pages live in a cache with physical stride 2**30+4096 and
block ID 2. It allocates roughly 8 GiB total but initializes only a few KiB,
so it exercises the overflow without the expensive broad fixture setup.
The focused eight-case segmented suite is running. Keep the full failing log;
do not count the aborted full suite as a pass. Earlier benchmark cache offsets
were below the boundary and their numerical results are valid, but final timing
must use widened-address code. New rows record the stage source hash.

## 2026-09-16 11:26 UTC — continuation audit

The prior four-hour deadline (2026-09-15 19:25 UTC) has passed. No GPU/test jobs
were live at resumption. Do not claim completion within that budget. The fixed
full prefix suite completed: 125 passed, 112 skipped in 520.94 seconds;
test-prefix-full-address64.log is the authoritative result. The source still
has the opt-in general path and unpromoted FlyDSL experiments. Engine/model
validation and final tuned-AITER comparisons remain incomplete. Stop expanding
the experimental design and close those checks using existing candidates.

## User-requested pause — 2026-09-16

Paused on the user's instruction. Corrected comparison: Q32/C8192 BF16
Triton 80.32 us, AITER default 158.48 us, prior tuned context 491.05 us;
ragged [7,33]/[137,1024]: Triton 36.08 us, AITER 51.50 us, context 76.22 us.
These are a single comparison pass, not a tuned-AITER/repeated-round final win.
FlyDSL streaming-Q candidate is 103.12 us on Q32, slower than Triton.
Q2/C512 still loses to AITER (23.20 vs 18.62 us); Q3/C0 loses to prior context
(17.46 vs 11.96 us). Keep opt-in default off and do not promote FlyDSL.
Full corrected prefix suite: 125 passed, 112 skipped. Production-source Ruff
checks pass after formatting. Engine baseline attempt failed because the new
detached checkout lacks the shared native vLLM extensions; it produced no engine
performance or model-evaluation results. Resume by wiring the same verified
prebuilt extensions as the existing comparison checkout, then closing engine
checks and tuned-AITER comparisons. The requested four-hour deadline was missed.
