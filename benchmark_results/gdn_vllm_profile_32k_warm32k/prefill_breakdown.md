# Qwen3.5 Prefill Profile Breakdown

- trace: `benchmark_results/gdn_vllm_profile_32k_warm32k/torch_profiler/rank0.1781074976606780412.pt.trace.json.gz`
- range: `execute_context_1(32768)_generation_0(0)`
- forward GPU range: `5316.855008 ms`
- CUDA kernel/memcpy/memset sum: `5309.388148 ms`
- unattributed gap inside range: `7.466860 ms`

## Buckets

| bucket | time (ms) | pct of forward range | calls |
| --- | ---: | ---: | ---: |
| full_attention | 650.487388 | 12.23% | 8 |
| gdn | 323.281031 | 6.08% | 192 |
| other | 4335.619729 | 81.54% | 705 |
| other_including_unattributed_gap | 4343.086589 | 81.69% | - |

## Top Events

| category | time (ms) | pct of forward range | calls | event |
| --- | ---: | ---: | ---: | --- |
| other | 2987.609883 | 56.19% | 96 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| other | 988.798083 | 18.60% | 32 | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| full_attention | 650.487388 | 12.23% | 8 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<256, 64, 64, 4, cutlass::bfloat16_t> >, true, false, false, false, true, false, false, false>(flash::Flash_fwd_params)` |
| other | 87.338357 | 1.64% | 24 | `triton_poi_fused_mul_silu_slice_3` |
| gdn | 70.035045 | 1.32% | 24 | `chunk_fwd_kernel_o` |
| gdn | 58.576949 | 1.10% | 24 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| other | 57.712995 | 1.09% | 24 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_4` |
| gdn | 54.166161 | 1.02% | 24 | `recompute_w_u_fwd_kernel` |
| gdn | 46.766718 | 0.88% | 24 | `_causal_conv1d_fwd_kernel` |
| gdn | 43.238182 | 0.81% | 24 | `_fused_post_conv_kernel` |
| gdn | 32.330713 | 0.61% | 24 | `merge_16x16_to_64x64_inverse_kernel` |
| other | 30.482940 | 0.57% | 24 | `triton_per_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_silu_view_0` |
| other | 27.982358 | 0.53% | 24 | `triton_red_fused__to_copy_add_fused_add_rms_norm_2` |
| other | 27.770035 | 0.52% | 8 | `triton_poi_fused_mul_silu_slice_2` |
| other | 19.268496 | 0.36% | 24 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_1` |
| other | 18.815944 | 0.35% | 8 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3` |
| other | 18.709158 | 0.35% | 27 | `Memcpy DtoD (Device -> Device)` |
| gdn | 17.307250 | 0.33% | 24 | `chunk_scaled_dot_kkt_fwd_kernel` |
| other | 11.323823 | 0.21% | 24 | `ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn` |
| other | 9.454676 | 0.18% | 8 | `triton_poi_fused_mul_sigmoid_view_0` |
