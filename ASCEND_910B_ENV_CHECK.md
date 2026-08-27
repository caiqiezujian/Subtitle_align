# D910B + vLLM 0.18.0 环境检查

本文档用于在服务器或 Docker 容器中检查 D910B、CANN、PyTorch NPU、vLLM Ascend 等环境是否正常。

这些命令不会启动字幕对齐服务，也不会加载 Qwen3-ASR 模型。请在以后实际运行字幕服务的同一个容器内，按照 1～6 的顺序执行。

## 1. 检查 D910B 是否被容器识别

```bash
npu-smi info
```

正常情况下应当能够看到：

- Ascend 910B 设备；
- NPU 设备编号；
- 显存占用；
- 设备健康状态。

如果提示 `npu-smi: command not found`，或者看不到任何 NPU，说明容器尚未正确映射昇腾设备或驱动。

## 2. 检查 Python 和关键组件版本

```bash
python3 --version
```

```bash
python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = [
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "qwen-asr",
    "transformers",
]

for package in packages:
    try:
        print(f"{package:15s} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:15s} NOT INSTALLED")
PY
```

vLLM 0.18.0 推荐重点核对下面的组合：

```text
Python          3.10 或 3.11
vllm            0.18.0
vllm-ascend     0.18.0
torch           2.9.0
torch-npu       2.9.0.post2
```

`vllm` 和 `vllm-ascend` 必须同时存在，并且版本保持一致。只有 `vllm==0.18.0` 不能证明已经支持昇腾 NPU。

## 3. 检查 Python 依赖冲突

```bash
python3 -m pip check
```

没有冲突时应当输出：

```text
No broken requirements found.
```

如果出现类似下面的信息：

```text
qwen-asr requires vllm==0.14.0, but you have vllm 0.18.0
```

说明当前 `qwen-asr` 与 vLLM 0.18.0 存在依赖版本冲突。此时先保留完整输出，不要直接升级、降级或重新安装依赖。

## 4. 检查 torch-npu 和基础 NPU 运算

下面只执行一个非常小的 NPU 张量运算，不会加载模型：

```bash
python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())

x = torch.tensor([1.0, 2.0, 3.0]).npu()
print("NPU calculation:", (x * 2).cpu().tolist())
PY
```

正确结果应当包含：

```text
NPU available: True
NPU count: 1
NPU calculation: [2.0, 4.0, 6.0]
```

`NPU count` 可以大于 1，具体数量取决于容器映射了多少张卡。

## 5. 检查 vLLM 是否激活 Ascend 插件

```bash
python3 - <<'PY'
import vllm
import vllm_ascend
from vllm.platforms import current_platform

print("Current platform:", current_platform)
print("Device type:", current_platform.device_type)
PY
```

正常情况下，日志或输出中应当看到类似内容：

```text
Platform plugin ascend is activated
Device type: npu
```

如果设备类型是 `cuda`、`cpu`，或者无法导入 `vllm_ascend`，说明当前 Python 环境不是完整的昇腾 vLLM 环境。

## 6. 检查 CANN 版本

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

如果该文件不存在，运行：

```bash
find /usr/local/Ascend -maxdepth 4 \( -name "version.cfg" -o -name "version.info" \) -print
```

然后查看找到的版本文件，例如：

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/*/version.info
```

vLLM Ascend 0.18.0 当前官方安装说明推荐使用 CANN 9.0.0。实际环境必须让 CANN、torch、torch-npu、vLLM 和 vllm-ascend 形成一套相互匹配的版本，不能只判断其中一个版本。

## 检查完成后需要保留的输出

请保留并反馈以下四部分结果：

1. 第 2 步的全部组件版本；
2. 第 3 步的 `pip check` 结果；
3. 第 4 步的 NPU 可用状态和计算结果；
4. 第 5 步的 vLLM 平台识别结果。

基础检查全部通过，只能证明 D910B 和 vLLM Ascend 环境可用。Qwen3-ASR 与 Qwen3-ForcedAligner 是否完整兼容，还需要在完成代码的 NPU 适配后进行一次小文件模型测试。

当前项目核心代码包含 `torch.cuda`、`cuda:0` 和 `CUDA_VISIBLE_DEVICES` 等 CUDA 专用逻辑，因此不能在 D910B 上原样启动。后续需要将设备检测、模型加载、设备编号和显存处理改为 Ascend NPU 逻辑。
