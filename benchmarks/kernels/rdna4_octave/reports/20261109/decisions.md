# Profiling-guided tuning decisions

Every source is isolated under `stages/`; no public kernel promotion occurs
until the final release gates pass. All GPU work uses physical GPU 0 and the
shared lock. Stage-specific `bench.jsonl`, `profiles/review.json`, collection logs
and generated sources retain the evidence. Ratios below are targeted paired
samples, not products to be presented as an overall speedup.

| Stage | Finding and decision |
| --- | --- |
| 00 baseline | Freeze kernel and Python policy. Validate SQ_WAVES; reject zero instruction/memory counters. Working ATT shows fragment loads and LDS/load waits. |
| 01 LDS vectors | Use 128-bit fragments, aligned Q/K and transposed V. D1024 sample improves 1.32x; smaller heads need further work. |
| 02 LDS barriers | Narrow matrix exchanges to shared-memory barriers. D1024 adds about 1.05x. Store synchronization stays intact. |
| 03 query reuse | Doubling narrow rows without rebalancing waves increases register pressure and slows several cases. Rejected. |
| 04 request tiles | Cover ragged boundaries with disjoint regular/prefix tiles. Sample improves 1.55x; FP32 guard cost falls from 1.76 ms to 0.014 ms. |
| 05 request specialization | Compile away request routing where the tile cannot cross requests; recover single-request overhead. |
| 06 KV range guard | Finite large raw/cached values expose native FP16 overflow. Explicit per-KV checks fix 36 fixtures but cost 3–14%. |
| 07 guard protocol | Reuse the query decision, detect invalid live logits/output, and communicate via an output marker. Retain FP32 recovery with lower overhead. |
| 08 native exchange | Replace generic XOR shuffles with gfx12 register permutations. Samples improve about 1.20x on prefill and 1.34x on raw prefill. Matrix ATT has no ds_bpermute. |
| 09 packed words | Loads fall sharply, but extra register exchanges slow several common heads. Rejected as a broad replacement. |
| 10 legacy recovery | Fix 13 failing range fixtures in preserved 256/64 WMMA paths using safe merge/repair. Scalar paths keep their launch count. |
| 11 split validation | Validate actual split statistics without assuming per-query partition width. Check non-power-of-two splits and empty/request-boundary cases. |
| 12 wide streaming | Alias K/V staging across phases, use 16 query rows, crop QK partial storage and transform eight output rows at a time. D1024 hot matrix scratch drops from 80 B/thread to zero; traced matrix time is about 18.2 to 11.5 ms. |
| 13 V padding | Broad padding regresses small/384 heads. A separate 30-case component screen identifies physical D512 as useful (18 cases, geometric mean 1.057x). |
| 14 parallel merge | Cooperate over splits with eight waves for small grids. Long-decode samples improve 1.32x at D128 and 1.77x at D1024. ATT is unavailable for this small-grid capture; successful kernel traces and wave counts provide the review. |
| 15 selected padding | Select V padding only at physical D512. Retain the unpadded path elsewhere. |
| 16 output range | 120 tiny-value comparisons previously fail. Recover tiny normalized native results in FP32; all 240 tiny-value comparisons pass. Adjacent-head marker ordering reduces the large-batch guard from about 0.21 to 0.057 ms. |
| 17 policy | 63 split sweeps and 48 engine fixtures support fewer tiny-matrix splits, more scalar-batch splits, and the geometry engine for eligible medium 256/64 prefills. Full original 1817-configuration accuracy coverage passes; expanded D1 coverage fails, so this candidate is superseded. |
| 18 tiny heads | Route logical D<32 through FP32 and adjust its split block estimate. All 72 targeted tiny-head diagnostic configurations pass. Normal D128/D1024 controls retain identical outputs. Freeze for the final broad release run. |

The first Stage 11 generator attempt aborted before applying its intended edit;
an already-started diagnostic ran against the copied parent. The raw diagnostic
files contain that first run followed by the corrected complete run. They are
not used as final release counts. Stage 18's release files have their own source
freeze and are the authoritative final validation artifacts.

Stage 19 retains the native binaries and corrects tiny MHA overpartitioning.
The earlier 17-query off-grid case confirmed a regression; a seven-count split
sweep measured S32 at 0.090 ms versus S128 at 0.136 ms. The final tiny-MHA
occupancy target is 128. Main/full/store measurements whose execution is
unchanged retain their actual Stage 18 provenance; a metadata-only engine/split
comparison verifies each inherited attention fixture. Final accuracy, graph and
new held-out workloads run against Stage 19.

Stage 20 addresses the second off-grid set's large regressions: extra scalar
splits stay inside the fast-merge output limit, and short K3V3 / thin MHA cases
retain the existing packed engine. A third off-grid set identifies two remaining
scalar tails at equal splits. Stage 21 tests the original shuffle reduction:
it helps those cases but slows single-query MHA. Stage 22 compiles both modes
and selects original reduction lowering only for physical D<=256 and more than
256 scalar decode outputs. Matrices, guards, stores and merges retain their
instruction bytes/resources, verified from ELF function bytes. Scalar timing
cases are remeasured; final accuracy/graph gates and a new off-grid set run on
the combined candidate.
