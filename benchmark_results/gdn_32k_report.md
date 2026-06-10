# Qwen3.5-9B GDN 32K Profile Report

Date: 2026-06-10
Device: NVIDIA L40
Model: `/data/Competitions/PRA26/Qwen3.5-9B`
vLLM commit/worktree: local `/home/xsl/pra26/vllm-cuda`
CUDA/PyTorch: CUDA 12.9 via `spack load cuda@12.9.1`, PyTorch `2.11.0+cu129`

## Model/GDN shape

From `config.json`:

- layers: 32 total, 24 `linear_attention` GDN layers
- hidden size: 4096
- GDN heads: `linear_num_key_heads=16`, `linear_num_value_heads=32`
- head dims: `linear_key_head_dim=128`, `linear_value_head_dim=128`
- GDN core packed qkv dim: `16*128*2 + 32*128 = 8192`
- conv kernel: 4
- activation/model dtype: bf16
- GDN state dtype: float32 (`mamba_ssm_dtype=float32`)
- selected prefill backend on L40: Triton/FLA

## Commands

Standalone GDN core:

```bash
srun --gres=gpu:1 zsh -lc '. /data/spack/share/spack/setup-env.sh && spack load cuda@12.9.1 && cd /home/xsl/pra26/vllm-cuda && .venv/bin/python benchmarks/benchmark_qwen35_gdn_core.py --prefill-tokens 32768 --decode-context 32768 --prefill-iters 3 --decode-iters 50 --warmups 2 --profile --output-dir benchmark_results/gdn_core_32k'
```

Full vLLM profile with an unprofiled 32K warmup request:

```bash
srun --gres=gpu:1 zsh -lc '. /data/spack/share/spack/setup-env.sh && spack load cuda@12.9.1 && cd /home/xsl/pra26/vllm-cuda && .venv/bin/python benchmarks/profile_qwen35_gdn.py --input-len 32768 --output-len 2 --warmup-input-len 32768 --warmup-output-len 2 --max-model-len 32776 --max-num-batched-tokens 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 --output-dir benchmark_results/gdn_vllm_profile_32k_warm32k'
```

## Standalone GDN core tensor metadata

Prefill, one 32K request:

- `mixed_qkv`: `[32768, 8192]`, `torch.bfloat16`, contiguous
- `b`: `[32768, 32]`, `torch.bfloat16`, contiguous
- `a`: `[32768, 32]`, `torch.bfloat16`, contiguous
- `core_attn_out`: `[32768, 32, 128]`, `torch.bfloat16`, contiguous
- `prefill_query_start_loc`: `[0, 32768]`, `torch.int32`
- `chunk_indices`: `[512, 2]`, `torch.int32`
- `chunk_offsets`: `[0, 512]`, `torch.int64`
- `prefill_has_initial_state`: `[False]`

Decode, batch 1 with 32K context:

- `mixed_qkv`: `[1, 8192]`, `torch.bfloat16`, contiguous
- `b`: `[1, 32]`, `torch.bfloat16`, contiguous
- `a`: `[1, 32]`, `torch.bfloat16`, contiguous
- `core_attn_out`: `[1, 32, 128]`, `torch.bfloat16`, contiguous
- `non_spec_query_start_loc`: `[0, 1]`, `torch.int32`

## Standalone GDN core timing

From `benchmark_results/gdn_core_32k/summary.json`:

- prefill GDN core CUDA-event average: `13.4205 ms` per layer call
- decode GDN core CUDA-event average: `0.5362 ms` per layer call

Prefill profiler self CUDA times for one layer call:

- `standalone_gdn_core/prefill`: `13.324 ms`
- `ChunkGatedDeltaRuleFunction`: `8.731 ms` self CUDA, `8.908 ms` CUDA total
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`: `2.658 ms`
- `recompute_w_u_fwd_kernel`: `2.241 ms`
- `chunk_fwd_kernel_o`: `2.223 ms`
- `_causal_conv1d_fwd_kernel`: `1.854 ms`
- `_fused_post_conv_kernel`: `1.785 ms`
- `merge_16x16_to_64x64_inverse_kernel`: `0.853 ms`
- `chunk_scaled_dot_kkt_fwd_kernel`: `0.723 ms`
- `chunk_local_cumsum_scalar_kernel`: `0.032 ms`

Decode profiler kernel self CUDA for one layer call:

- `fused_sigmoid_gating_delta_rule_update_kernel`: `3.456 us`
- `_causal_conv1d_update_kernel`: `1.631 us`
- `aten::cat` CUDA kernel: `1.504 us`
- `Memcpy DtoD`: `1.056 us`

The decode CUDA-event average includes launch/scheduling gaps and Python driver overhead around many tiny kernels; the profiler kernel self times above are the actual GPU kernel durations.

## Full vLLM profile

Results:

- output dir: `benchmark_results/gdn_vllm_profile_32k_warm32k`
- profiler table: `benchmark_results/gdn_vllm_profile_32k_warm32k/torch_profiler/profiler_out_0.txt`
- trace: `benchmark_results/gdn_vllm_profile_32k_warm32k/torch_profiler/rank0.1781074976606780412.pt.trace.json.gz`
- generated tokens: 2
- latency including profiler overhead: `6.3476 s`
- profiled prefill forward range: `execute_context_1(32768)_generation_0(0)` self CUDA `5.317 s`
- profiled decode forward range: `execute_context_0(0)_generation_1(1)` self CUDA `25.047 ms`

GDN-related rows from the full vLLM profiler:

- `ChunkGatedDeltaRuleFunction`: 24 calls, `233.276 ms` self CUDA, `237.543 ms` CUDA total, `9.898 ms` average CUDA total
- `vllm::qwen_gdn_attention_core`: 24 calls, `90.005 ms` self CUDA, `346.132 ms` CUDA total, `14.422 ms` average CUDA total
- `chunk_fwd_kernel_o`: 24 calls, `70.035 ms` total, `2.918 ms` avg
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`: 24 calls, `58.577 ms` total, `2.441 ms` avg
- `recompute_w_u_fwd_kernel`: 24 calls, `54.166 ms` total, `2.257 ms` avg
- `_causal_conv1d_fwd_kernel`: 24 calls, `46.767 ms` total, `1.949 ms` avg
- `_fused_post_conv_kernel`: 24 calls, `43.238 ms` total, `1.802 ms` avg
- `merge_16x16_to_64x64_inverse_kernel`: 24 calls, `32.331 ms` total, `1.347 ms` avg
- `chunk_scaled_dot_kkt_fwd_kernel`: 24 calls, `17.307 ms` total, `0.721 ms` avg
- `fused_recurrent_gated_delta_rule_packed_decode_kernel`: 24 calls, `221.665 us` total, `9.236 us` avg
- `_causal_conv1d_update_kernel`: 24 calls, `62.146 us` total, `2.589 us` avg

## Prefill forward breakdown

The prefill breakdown uses the GPU annotation range
`execute_context_1(32768)_generation_0(0)`, whose CUDA time is
`5.316855 s`. For module-level percentages, GDN uses the profiler row
`vllm::qwen_gdn_attention_core` CUDA total, while full attention uses the
FlashAttention prefill kernels inside the prefill range.

Analysis command for strict kernel-name buckets:

```bash
.venv/bin/python benchmarks/analyze_qwen35_decode_profile.py benchmark_results/gdn_vllm_profile_32k_warm32k --phase prefill --json-output benchmark_results/gdn_vllm_profile_32k_warm32k/prefill_breakdown.json --markdown-output benchmark_results/gdn_vllm_profile_32k_warm32k/prefill_breakdown.md
```

Results for one 32K prefill forward:

| category | time | pct of prefill forward |
| --- | ---: | ---: |
| prefill forward total | `5316.855008 ms` | `100.00%` |
| full attention | `650.487388 ms` | `12.23%` |
| GDN core, profiler CUDA total | `346.132 ms` | `6.51%` |
| other, residual to forward total | `4320.235620 ms` | `81.25%` |

The strict kernel-name bucket file reports named GDN kernels at
`323.281031 ms`, `6.08%`; the gap to the profiler CUDA-total GDN number is
CUDA time attributed to the GDN record-function range but not to the explicit
GDN kernel-name list.

Dominant prefill kernels:

- BF16 GEMM kernels: `3976.407966 ms`, `74.79%` of prefill forward
- FlashAttention prefill split-kv: `650.487388 ms`, `12.23%`
- GDN named kernels: `323.281031 ms`, `6.08%`
- `triton_poi_fused_mul_silu_slice_3`: `87.338357 ms`, `1.64%`

## Decode forward breakdown

The decode breakdown was computed from the torch profiler trace by selecting the
GPU annotation range `execute_context_0(0)_generation_1(1)` and summing
CUDA `kernel`, `gpu_memcpy`, and `gpu_memset` events fully inside that range.

Analysis command:

```bash
.venv/bin/python benchmarks/analyze_qwen35_decode_profile.py benchmark_results/gdn_vllm_profile_32k_warm32k --json-output benchmark_results/gdn_vllm_profile_32k_warm32k/decode_breakdown.json --markdown-output benchmark_results/gdn_vllm_profile_32k_warm32k/decode_breakdown.md
```

Classification:

- full attention: `flash_fwd_splitkv*` kernels
- GDN: `fused_recurrent_gated_delta_rule_packed_decode_kernel` and `_causal_conv1d_update_kernel`
- other: remaining CUDA kernels/copies/sets; the residual profiler gap inside the decode GPU annotation is also counted as other for the end-to-end percentage

Results for one decode forward after the 32K warmup/prefill:

| category | time | pct of decode forward |
| --- | ---: | ---: |
| decode forward total | `25.046954 ms` | `100.00%` |
| full attention | `1.854617 ms` | `7.40%` |
| GDN | `0.283811 ms` | `1.13%` |
| other CUDA events | `22.807012 ms` | `91.06%` |
| unattributed profiler gap | `0.101514 ms` | `0.41%` |
| other including gap | `22.908526 ms` | `91.46%` |

Dominant decode kernels:

- cuBLAS GEMV kernels: `22.046970 ms`, `88.02%` of decode forward
- FlashAttention split-kv: `1.719449 ms`, `6.86%`
- FlashAttention combine: `0.135168 ms`, `0.54%`
- GDN recurrent delta-rule decode: `0.221665 ms`, `0.88%`
- GDN conv update: `0.062146 ms`, `0.25%`

## Notes

- The full vLLM process uses spawn workers, so the script-level monkey patch did not capture in-worker GDN tensor metadata. The metadata above comes from the standalone GDN-core script using the same `_forward_core` path and vLLM metadata builder.
- The second full vLLM run used a 32K unprofiled request before `start_profile()`. Triton JIT warnings appeared before profiler start and are therefore excluded from the profiler table.
- Full-model CUDA time is dominated by GEMM and full-attention layers; GDN prefill core accounts for roughly `346 ms / 5317 ms = 6.5%` of the 32K prefill CUDA total in this run.
