from __future__ import annotations

import os
import shutil
from importlib.metadata import PackageNotFoundError, version

from simple_srt_service.settings import settings


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"缺少 Python 包：{name}") from exc


def version_family(value: str) -> tuple[int, int]:
    numeric = value.split("+", 1)[0].split("rc", 1)[0]
    pieces = numeric.split(".")
    try:
        return int(pieces[0]), int(pieces[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"无法识别软件版本：{value}") from exc


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，请先安装 ffmpeg")

    installed = {
        name: package_version(name)
        for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "qwen-asr")
    }
    if version_family(installed["vllm"]) != (0, 14):
        raise RuntimeError(
            f"当前 vllm={installed['vllm']}，本服务当前按 vLLM 0.14 适配"
        )
    if version_family(installed["vllm-ascend"]) != (0, 14):
        raise RuntimeError(
            "vllm-ascend 必须与 vLLM 使用 0.14 系列，当前为 "
            f"{installed['vllm-ascend']}"
        )
    if version_family(installed["torch"]) != version_family(installed["torch-npu"]):
        raise RuntimeError(
            "torch 与 torch-npu 主次版本不一致："
            f"torch={installed['torch']}，torch-npu={installed['torch-npu']}"
        )

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = settings.npu_visible_devices
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    import torch
    import torch_npu  # noqa: F401
    import vllm  # noqa: F401
    import vllm_ascend  # noqa: F401
    from qwen_asr import Qwen3ASRModel

    if not callable(getattr(Qwen3ASRModel, "LLM", None)):
        raise RuntimeError("当前 qwen-asr 不提供 Qwen3ASRModel.LLM vLLM 后端")
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() 为 False")
    if settings.npu_logical_device_index >= torch.npu.device_count():
        raise RuntimeError(
            f"配置请求 {settings.npu_device}，但隔离后仅识别到 "
            f"{torch.npu.device_count()} 张 NPU"
        )

    required_models = [
        settings.model_root / "Qwen3-ASR-1.7B",
        settings.model_root / "Qwen3-ForcedAligner-0.6B",
    ]
    missing = [str(path) for path in required_models if not path.is_dir()]
    if missing:
        raise RuntimeError("缺少模型目录：" + "、".join(missing))

    torch.npu.set_device(settings.npu_device)
    probe = torch.tensor([1.0, 2.0], dtype=torch.float32).to(settings.npu_device)
    if (probe * 2).cpu().tolist() != [2.0, 4.0]:
        raise RuntimeError("NPU 基础张量计算结果异常")

    print("Ascend vLLM 环境预检通过")
    for name, value in installed.items():
        print(f"{name}={value}")
    print(f"physical_visible_npu={settings.npu_visible_devices}")
    print(f"logical_device={settings.npu_device}")
    print(f"model_root={settings.model_root}")


if __name__ == "__main__":
    main()
