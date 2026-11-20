# Qwen GDN monokernel on RDNA4

This package fuses causal convolution, gated delta recurrence, gated RMSNorm,
and group-128 FP8 quantization for a Qwen3.8-27B-FP8 GDN layer. The kernels use
FlyDSL's typed ROCm buffer pointers and operate on the convolution and recurrent
caches allocated by vLLM.

- `dispatch.py` binds the kernels to the packed Qwen GDN entry point and selects
  the path from the forward metadata and cache layout.
- `gdn_decode.py` advances one token per request.
- `gdn_verify.py` verifies a DFlash2 token group and saves each possible
  accepted prefix. It selects the lane layout from the verification shape.
- `gdn_norm.py` contains the FP8 epilogue and a standalone kernel used with the
  standard GDN core.
- `common.py` contains the shared FlyDSL device functions.

Set `VLLM_ROCM_USE_FLYDSL_QWEN_FP8=1` with a ROCm build, FlyDSL installed, and
`VLLM_ROCM_USE_AITER=0`. The model selects the path for BF16 activations, TP1
or TP2, and a block-FP8 output projection on RDNA4. Ordinary decode is fused
up to 16 tokens per batch. DFlash2 verification is fused up to seven drafts per
request and 64 verification positions per batch. Prefill, mixed batches, and
wider batches use the standard GDN core.

The model-local dispatch and kernel layout follow the specialized decode
structure. [vLLM RFC #60904](https://github.com/vllm-project/vllm/issues/60904)

See the [benchmark and serving commands](../../../../../docs/benchmarking/flydsl_qwen_gdn.md).
