from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SERVICE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SERVICE_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 YAML 对象：{path}")
    return value


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config.yaml 中的 {name} 必须是对象")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _path(value: Any, *, relative_to_service: bool = False) -> Path:
    path = Path(str(value)).expanduser()
    if relative_to_service and not path.is_absolute():
        path = SERVICE_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class SimpleSettings:
    server_host: str
    server_port: int
    npu_visible_devices: str
    npu_logical_device_index: int
    model_root: Path
    data_dir: Path
    max_upload_mb: int
    source_language: str
    engine_gpu_memory_utilization: float
    engine_max_inference_batch_size: int
    engine_asr_batch_size: int
    engine_forced_aligner_batch_size: int
    engine_max_new_tokens: int
    engine_max_model_len: int
    engine_enforce_eager: bool
    engine_attention_implementation: str
    engine_startup_timeout_seconds: int

    @property
    def npu_device(self) -> str:
        return f"npu:{self.npu_logical_device_index}"


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return result


def build_settings(config: dict[str, Any] | None = None) -> SimpleSettings:
    config = load_config() if config is None else config
    server = _section(config, "server")
    npu = _section(config, "npu")
    # 兼容旧版 simple_srt_service/config.yaml，升级后无需立刻改掉 gpu 节。
    legacy_gpu = _section(config, "gpu")
    engine = _section(config, "alignment_engine")
    models = _section(config, "models")
    storage = _section(config, "storage")
    language = str(engine.get("language", "auto")).strip()
    if language not in {"auto", "Chinese", "English", "Japanese"}:
        raise ValueError(
            "alignment_engine.language 仅支持 auto、Chinese、English、Japanese"
        )

    logical_device_index = int(npu.get("logical_device_index", 0))
    if logical_device_index < 0:
        raise ValueError("npu.logical_device_index 不能小于 0")

    memory_utilization = float(engine.get("gpu_memory_utilization", 0.85))
    if not 0 < memory_utilization < 1:
        raise ValueError("gpu_memory_utilization 必须大于 0 且小于 1")

    attention = str(engine.get("attention_implementation", "eager")).strip()
    if attention not in {"eager", "sdpa"}:
        raise ValueError("attention_implementation 仅支持 eager 或 sdpa")

    return SimpleSettings(
        server_host=str(server.get("host", "0.0.0.0")),
        server_port=int(server.get("port", 12046)),
        npu_visible_devices=str(
            npu.get("visible_devices", legacy_gpu.get("visible_devices", "6"))
        ),
        npu_logical_device_index=logical_device_index,
        model_root=_path(models.get("root", "/data/yb/model")),
        data_dir=_path(storage.get("data_dir", "./data"), relative_to_service=True),
        max_upload_mb=_positive_int(storage.get("max_upload_mb", 4096), "max_upload_mb"),
        source_language=language,
        engine_gpu_memory_utilization=memory_utilization,
        engine_max_inference_batch_size=_positive_int(
            engine.get("max_inference_batch_size", 4),
            "max_inference_batch_size",
        ),
        engine_asr_batch_size=_positive_int(
            engine.get("asr_batch_size", 4), "asr_batch_size"
        ),
        engine_forced_aligner_batch_size=_positive_int(
            engine.get("forced_aligner_batch_size", 4),
            "forced_aligner_batch_size",
        ),
        engine_max_new_tokens=_positive_int(
            engine.get("max_new_tokens", 1024), "max_new_tokens"
        ),
        engine_max_model_len=_positive_int(
            engine.get("max_model_len", 4096), "max_model_len"
        ),
        engine_enforce_eager=_as_bool(engine.get("enforce_eager", False)),
        engine_attention_implementation=attention,
        engine_startup_timeout_seconds=_positive_int(
            engine.get("startup_timeout_seconds", 1800),
            "startup_timeout_seconds",
        ),
    )


settings = build_settings()
