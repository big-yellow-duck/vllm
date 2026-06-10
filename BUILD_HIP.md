# HIP vLLM 编译安装指南

## 环境信息

- GPU: AMD Radeon AI PRO R9700 (位于 login node hepnode0)
- OS: Linux 6.11.0-29-generic
- Python: 3.12 (通过 uv 管理)
- PyTorch: 2.12.0+rocm7.2
- ROCm: 7.2

## 安装步骤

### 1. 创建 Python 虚拟环境

```bash
cd /home/xsl/pra26/vllm-hip
uv venv --python 3.12
source .venv/bin/activate
```

### 2. 安装 PyTorch (ROCm 版本)

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

注意：不要使用 `--torch-backend=auto`，它会错误地安装 CPU 版本。

### 3. 安装 ROCm 平台检测依赖

vLLM 通过 `amdsmi` 检测 ROCm GPU：

```bash
uv pip install amdsmi
```

验证：

```python
import amdsmi
amdsmi.amdsmi_init()
print(amdsmi.amdsmi_get_processor_handles())  # 应返回 GPU handle 列表
amdsmi.amdsmi_shut_down()
```

### 4. 安装构建依赖

vLLM 的构建系统需要以下额外依赖：

```bash
uv pip install setuptools_scm setuptools_rust
```

同时需要 setuptools >= 77.0.3：

```bash
uv pip install "setuptools>=77.0.3,<81.0.0"
```

### 5. 编译安装 vLLM (HIP 模式)

```bash
VLLM_GPU_LANGUAGES=hip uv pip install -e . --no-build-isolation
```

关键参数说明：
- `VLLM_GPU_LANGUAGES=hip`：指定编译 HIP/ROCm 的 C 扩展
- `--no-build-isolation`：使用当前环境的依赖，而非创建隔离构建环境
- `-e`：editable 模式安装，方便开发调试

### 6. 修复 torchvision 版本

编译 vLLM 可能会把 torchvision 替换为 CPU 版本。如果发生这种情况，重新安装 ROCm 版本：

```bash
uv pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

## 验证安装

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/Competitions/PRA26/Qwen3.5-9B",
    trust_remote_code=True,
    max_model_len=512,
    gpu_memory_utilization=0.8,
)
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.7, max_tokens=64))
print(outputs[0].outputs[0].text)
```

## 常见问题

### `Device string must not be empty`

原因：vLLM 未检测到 ROCm 平台，回退到 `UnspecifiedPlatform`。

解决：安装 `amdsmi`（见步骤 3）。

### `'_OpNamespace' '_C' object has no attribute 'silu_and_mul'`

原因：C 扩展未编译，vLLM 以纯 Python 模式运行。

解决：以 `VLLM_GPU_LANGUAGES=hip` 重新编译（见步骤 5）。

### `RuntimeError: Only one platform plugin can be activated, but got: ['rocm', 'cpu']`

原因：源码目录残留旧的 `vllm.egg-info`，版本号包含 "cpu" 标记。

解决：

```bash
rm -rf vllm.egg-info
VLLM_GPU_LANGUAGES=hip uv pip install -e . --no-build-isolation
```

### `operator torchvision::nms does not exist`

原因：torchvision 是 CPU 版本，与 ROCm 版 PyTorch 不兼容。

解决：重新安装 torchvision 的 ROCm 版本（见步骤 6）。

### `ModuleNotFoundError: No module named 'setuptools_rust'`

原因：缺少构建依赖。

解决：`uv pip install setuptools_rust setuptools_scm`（见步骤 4）。
