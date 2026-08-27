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
    cuda_visible_devices: str
    model_root: Path
    data_dir: Path
    max_upload_mb: int
    source_language: str
    engine_gpu_memory_utilization: float
    engine_max_inference_batch_size: int
    engine_max_new_tokens: int
    engine_flash_attention: bool
    engine_startup_timeout_seconds: int


def build_settings(config: dict[str, Any] | None = None) -> SimpleSettings:
    config = load_config() if config is None else config
    server = _section(config, "server")
    gpu = _section(config, "gpu")
    engine = _section(config, "alignment_engine")
    models = _section(config, "models")
    storage = _section(config, "storage")
    language = str(engine.get("language", "auto")).strip()
    if language not in {"auto", "Chinese", "English", "Japanese"}:
        raise ValueError(
            "alignment_engine.language 仅支持 auto、Chinese、English、Japanese"
        )
    return SimpleSettings(
        server_host=str(server.get("host", "0.0.0.0")),
        server_port=int(server.get("port", 12045)),
        cuda_visible_devices=str(gpu.get("visible_devices", "5")),
        model_root=_path(models.get("root", "/data/yb/Code/models")),
        data_dir=_path(storage.get("data_dir", "./data"), relative_to_service=True),
        max_upload_mb=int(storage.get("max_upload_mb", 4096)),
        source_language=language,
        engine_gpu_memory_utilization=float(
            engine.get("gpu_memory_utilization", 0.65)
        ),
        engine_max_inference_batch_size=int(
            engine.get("max_inference_batch_size", 32)
        ),
        engine_max_new_tokens=int(engine.get("max_new_tokens", 2048)),
        engine_flash_attention=_as_bool(engine.get("flash_attention", False)),
        engine_startup_timeout_seconds=int(
            engine.get("startup_timeout_seconds", 900)
        ),
    )


settings = build_settings()

