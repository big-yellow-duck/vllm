# Qwen3.5 Decode Profile Breakdown

- trace: `benchmark_results/gdn_vllm_profile_32k_warm32k/torch_profiler/rank0.1781074976606780412.pt.trace.json.gz`
- range: `execute_context_0(0)_generation_1(1)`
- forward GPU range: `25.046954 ms`
- CUDA kernel/memcpy/memset sum: `24.945440 ms`
- unattributed gap inside range: `0.101514 ms`

## Buckets

| bucket | time (ms) | pct of forward range | calls |
| --- | ---: | ---: | ---: |
| full_attention | 1.854617 | 7.40% | 16 |
| gdn | 0.283811 | 1.13% | 48 |
| other | 22.807012 | 91.06% | 469 |
| other_including_unattributed_gap | 22.908526 | 91.46% | - |

## Top Events

| category | time (ms) | pct of forward range | calls | event |
| --- | ---: | ---: | ---: | --- |
| other | 17.570799 | 70.15% | 120 | `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>)` |
| other | 4.476171 | 17.87% | 33 | `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>)` |
| full_attention | 1.719449 | 6.86% | 8 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<256, 64, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, true, false, true, false>(flash::Flash_fwd_params)` |
| gdn | 0.221665 | 0.88% | 24 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| full_attention | 0.135168 | 0.54% | 8 | `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<256, 64, 64, 4, cutlass::bfloat16_t> >, 4, 6, true>(flash::Flash_fwd_params)` |
| other | 0.134432 | 0.54% | 24 | `triton_per_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_silu_view_0` |
| other | 0.083200 | 0.33% | 24 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_4` |
| other | 0.069280 | 0.28% | 24 | `triton_red_fused__to_copy_add_fused_add_rms_norm_2` |
| gdn | 0.062146 | 0.25% | 24 | `_causal_conv1d_update_kernel` |
| other | 0.040800 | 0.16% | 24 | `triton_poi_fused_mul_silu_slice_3` |
| other | 0.037408 | 0.15% | 24 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_1` |
| other | 0.034500 | 0.14% | 8 | `triton_poi_fused_clone_copy_index_select_slice_split_7` |
| other | 0.033185 | 0.13% | 24 | `triton_poi_fused_5` |
| other | 0.032640 | 0.13% | 4 | `_compute_slot_mapping_kernel` |
| other | 0.027229 | 0.11% | 8 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3` |
| other | 0.024512 | 0.10% | 8 | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| other | 0.022592 | 0.09% | 8 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*, int)` |
| other | 0.021315 | 0.09% | 8 | `triton_poi_fused__to_copy_add_cat_clone_mul_rms_norm_slice_split_split_with_sizes_sub_unsqueeze_view_8` |
| other | 0.020160 | 0.08% | 8 | `triton_poi_fused__to_copy_add_cat_mul_rms_norm_slice_split_split_with_sizes_sub_unsqueeze_view_10` |
| other | 0.018913 | 0.08% | 16 | `triton_poi_fused_zeros_6` |
