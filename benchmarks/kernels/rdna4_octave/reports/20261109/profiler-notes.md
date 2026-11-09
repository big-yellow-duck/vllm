# Profiler validation

The frozen baseline is traced with rocprofv3 1.3.5 on physical GPU 0 (gfx1201).
ROCTx selected regions contain only preallocated attention invocations. Builds,
cache setup and correctness oracles are outside those regions.

The 1024/128 K3V4 BF16 sample (256 queries, 4096 prefix, GQA8) measures a median
27.914 ms in the matrix kernel, 0.070 ms in the range guard and 0.055 ms in merge.
`SQ_WAVES` matches the launch geometry exactly: 8192 matrix waves, 8192 guard
waves and 4096 merge waves per call. Instruction and FetchSize counters return
zeros on active work and are excluded.

AMD documents that AUTO mode can gate PMC clocks on RDNA3/4, and that
`profile_standard` enables the affected counters. This installation denies
unprivileged performance-level access; noninteractive sudo is unavailable.
No device performance settings or permissions were changed. The observed zero
pattern is consistent with that documented restriction, not proof of its cause.
[AMD profiler configuration](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)

Advanced Thread Trace works using an explicit decoder-library directory for
this ROCm wheel installation. The first default-path probe failed to load the
decoder and was terminated after it entered its error handler; its raw capture
is retained separately. The explicit-path capture decoded successfully with no
reported lost/incomplete data. Sampled instruction timing places LDS-load waits,
load completion waits, and scalar FP16 fragment loads among the main hotspots.
This is a sampled WGP/SIMD view, not whole-GPU utilization.
[AMD thread-trace documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-thread-trace.html)

Each tuning stage retains graph timings, numerical checks, kernel traces,
runtime resource fields and a review. Static ISA audits cover selected milestones
and all release geometries. Most stages also have a decoded instruction trace.
Stage 14's two-block parallel-merge ATT capture timed out without decoded data;
that decision uses the successful kernel trace, validated wave counts and paired
timings. Its merge median falls from about 0.23 ms to 0.017 ms. No instruction-level
or achieved-utilization claim is made for that capture.

ATT latency/stall counts are sampled at one WGP/SIMD and should not be compared
as whole-device percentages. Runtime VGPR allocation fields and static compiler
register counts are retained separately. Zero scratch in the selected wide WMMA
matrix does not imply zero scratch in its FP32 recovery kernel.
