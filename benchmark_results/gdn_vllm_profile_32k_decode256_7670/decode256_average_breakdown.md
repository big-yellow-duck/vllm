# Qwen3.5 Decode Profile Breakdown

- trace: `benchmark_results/gdn_vllm_profile_32k_decode256_7670/torch_profiler/rank0.1781085195168744862.pt.trace.json.gz`
- range mode: `all`
- range count: `256`
- first range: `execute_context_0(0)_generation_1(1)`
- last range: `execute_context_0(0)_generation_1(1)`
- average forward GPU range: `24.271828 ms`
- min/max forward GPU range: `24.240860 ms` / `25.065174 ms`
- total forward GPU range: `6213.588001 ms`
- average CUDA kernel/memcpy/memset sum: `24.198709 ms`
- average unattributed gap inside range: `0.073119 ms`

## Average Buckets

| bucket | avg time (ms) | pct of total forward ranges | total time (ms) | total calls | avg calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_attention | 1.727108 | 7.12% | 442.139521 | 4096 | 16.00 |
| gdn | 0.204471 | 0.84% | 52.344612 | 12288 | 48.00 |
| other | 22.267130 | 91.74% | 5700.385352 | 120037 | 468.89 |
| other_including_unattributed_gap | 22.340249 | 92.04% | 5719.103868 | - | - |

## Top Events

| category | avg time (ms) | pct of total forward ranges | total calls | avg calls | event |
| --- | ---: | ---: | ---: | ---: | --- |
| other | 17.372844 | 71.58% | 30720 | 120.00 | `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, false, true, true, false, 7, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>)` |
| other | 4.407325 | 18.16% | 8448 | 33.00 | `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, false, true, true, false, 6, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStridedBatched<__nv_bfloat16>, float>)` |
| full_attention | 1.635107 | 6.74% | 2048 | 8.00 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<256, 64, 64, 4, cutlass::bfloat16_t> >, false, false, false, false, true, false, true, false>(flash::Flash_fwd_params)` |
| gdn | 0.157216 | 0.65% | 6144 | 24.00 | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| full_attention | 0.092000 | 0.38% | 2048 | 8.00 | `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<256, 64, 64, 4, cutlass::bfloat16_t> >, 4, 6, true>(flash::Flash_fwd_params)` |
| other | 0.078438 | 0.32% | 6144 | 24.00 | `triton_per_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_silu_view_0` |
| other | 0.049921 | 0.21% | 6144 | 24.00 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_4` |
| gdn | 0.047255 | 0.19% | 6144 | 24.00 | `_causal_conv1d_update_kernel` |
| other | 0.042299 | 0.17% | 6144 | 24.00 | `triton_red_fused__to_copy_add_fused_add_rms_norm_2` |
| other | 0.028252 | 0.12% | 6144 | 24.00 | `triton_poi_fused_mul_silu_slice_3` |
| other | 0.025295 | 0.10% | 6144 | 24.00 | `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_view_1` |
| other | 0.023574 | 0.10% | 6144 | 24.00 | `triton_poi_fused_5` |
| other | 0.021772 | 0.09% | 2048 | 8.00 | `triton_poi_fused_clone_copy_index_select_slice_split_7` |
| other | 0.019234 | 0.08% | 1024 | 4.00 | `_compute_slot_mapping_kernel` |
| other | 0.016591 | 0.07% | 2048 | 8.00 | `triton_red_fused__to_copy_add_copy__fused_add_rms_norm_3` |
| other | 0.015399 | 0.06% | 2048 | 8.00 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*, __nv_bfloat16*, long const*, long, long, long, long, long, int, int, int, float const*, float const*, int)` |
| other | 0.014930 | 0.06% | 2048 | 8.00 | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| other | 0.013564 | 0.06% | 4096 | 16.00 | `triton_poi_fused_zeros_6` |
| other | 0.013246 | 0.05% | 2048 | 8.00 | `triton_poi_fused__to_copy_add_cat_clone_mul_rms_norm_slice_split_split_with_sizes_sub_unsqueeze_view_8` |
| other | 0.012966 | 0.05% | 2048 | 8.00 | `triton_poi_fused__to_copy_add_cat_mul_rms_norm_slice_split_split_with_sizes_sub_unsqueeze_view_10` |
