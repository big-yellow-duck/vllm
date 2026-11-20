# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## FlyDSL GDN decode on RDNA4

Set `VLLM_ROCM_USE_FLYDSL_QWEN_FP8=1` for Qwen3.8-27B-FP8 on RDNA4.
The decode kernel updates the convolution and recurrent caches, applies gated
RMSNorm, and emits group-128 FP8 activations for the output projection in one
launch per GDN layer.

Use a ROCm build with FlyDSL installed, `VLLM_ROCM_USE_AITER=0`, and a
Triton block-FP8 linear backend. The path
supports BF16 activations and tensor parallel sizes 1 and 2. It fuses ordinary
decode for up to 16 tokens per batch and DFlash verification for up to seven
draft tokens per request and 64 verification positions per batch. Prefill,
mixed prefill/verification batches, and larger batches use the standard GDN
core with a fused FP8 output norm.

See the [two-GPU benchmark](../benchmarking/flydsl_qwen_gdn.md).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
