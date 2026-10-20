---
name: guidellm-benchmark-compare
description: Compare two GuideLLM multi-turn benchmark JSON reports with per-concurrency TPS, TTFT, and E2E charts plus a percentage-based Markdown analysis. Use when evaluating two serving configurations, kernels, backends, baselines, or optimization cases from GuideLLM results.
---

# GuideLLM Benchmark Compare

Produce a consistent, request-level comparison of exactly two GuideLLM cases.

## Resolve the two cases

Infer the baseline and candidate only when the user's wording and available
paths identify exactly two cases. If either case is ambiguous, pause before
plotting or writing and ask one concise question: which two result cases should
be compared, and which is the baseline? Do not silently choose among multiple
plausible reports.

Accept explicit JSON paths or resolve case names to
`agent-multiturn-benchmarks.json` files with `rg --files`. Keep unrelated
working-tree changes intact.

## Validate before comparing

Read both reports and verify:

- model and workload;
- configured streams;
- prompt/output tokens, turns, and seed;
- successful, errored, and incomplete request counts.

Call out meaningful mismatches. Do not present percentage differences as an
apples-to-apples optimization result when workload-defining fields differ.

Use means over all `requests.successful` records for both the charts and the
table. Do not mix these with GuideLLM's `benchmark.metrics` summaries:
warmup/cooldown exclusion can differ across reports and previously produced
tables that contradicted the charts.

## Generate charts

Run the bundled chart script:

```bash
python <skill-dir>/scripts/plot_compare_multiturn.py \
  --report "Baseline label=/path/to/baseline.json" \
  --report "Candidate label=/path/to/candidate.json" \
  --title "Descriptive comparison title" \
  --output-dir /path/to/comparison-plots
```

It writes TPS, TTFT, and E2E figures. Each figure uses a 2×2 layout with one
subplot for streams 1, 2, 4, and 8 and two lines per subplot. Visually inspect
at least the TPS and TTFT outputs. If the reports have other concurrency levels,
adapt the layout rather than dropping data.

## Build the analysis

Use the bundled summary helper as the numeric source:

```bash
python <skill-dir>/scripts/summarize_guidellm.py \
  /path/to/baseline.json /path/to/candidate.json
```

Create a Markdown report near the results unless the user specifies another
location. Include:

- setup and compatibility checks;
- the generated baseline → candidate table;
- embedded links to all three figures;
- a concise interpretation of TPS, TTFT, and E2E changes;
- caveats supported by the data, such as achieved-concurrency differences or
  inconsistent request counts.

State that positive TPS changes improve performance and negative TTFT/E2E
changes improve performance. Bold improvements in the table. Describe these as
end-to-end serving results unless the inputs are specifically kernel
microbenchmarks.

Regenerate charts after data changes. Verify scripts compile, Markdown image
paths resolve, outputs exist, and `git diff --check` passes.
