# PR #59132 gfx1151 serving figures

Figures copied from the Qwen3.8-27B W4A16 AutoRound ROCM_ATTN versus ROCM_SEGMENTED_ATTN comparison in vllm-bench-collection at f4c1fb5bb104e6852c4a86d832c1b6ca4f692b44.

GuideLLM 0.7.3 simple-agent-multiturn-prefix-cache workload: 2,048 configured prompt tokens, 512 output tokens, eight turns, seed 20260715, and 1/2/4/8 configured streams. Images report mean per-request output tokens/s, time to first token, and end-to-end latency by turn. Realized prompt lengths, achieved concurrency, Linux, and Python versions differed between reports; the PR body records these limitations.
