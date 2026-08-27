from __future__ import annotations

import shutil
import os
from importlib.metadata import PackageNotFoundError, version

from simple_srt_service_ascend.settings import settings


EXPECTED_QWEN_ASR = "0.0.6"
EXPECTED_TRANSFORMERS = "4.57.6"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"缺少 Python 包：{name}") from exc


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，请先安装 ffmpeg")

    qwen_version = package_version("qwen-asr")
    transformers_version = package_version("transformers")
    torch_version = package_version("torch")
    torch_npu_version = package_version("torch-npu")

    if qwen_version != EXPECTED_QWEN_ASR:
        raise RuntimeError(
            f"qwen-asr 版本为 {qwen_version}，本服务锁定并验证 {EXPECTED_QWEN_ASR}"
        )
    if transformers_version != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            "transformers 版本为 "
            f"{transformers_version}，本服务为避免 ASR 精度风险锁定 "
            f"{EXPECTED_TRANSFORMERS}"
        )

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = settings.npu_visible_devices

    import torch
    import torch_npu  # noqa: F401
    import qwen_asr  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() 为 False")
    if settings.npu_logical_device_index >= torch.npu.device_count():
        raise RuntimeError(
            f"配置请求 {settings.npu_device}，但容器仅识别到 "
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

    print("Ascend 环境预检通过")
    print(f"torch={torch_version}")
    print(f"torch-npu={torch_npu_version}")
    print(f"qwen-asr={qwen_version}")
    print(f"transformers={transformers_version}")
    print(f"device={settings.npu_device}")
    print(f"model_root={settings.model_root}")


if __name__ == "__main__":
    main()
