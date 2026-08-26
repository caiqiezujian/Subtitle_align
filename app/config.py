from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 YAML 对象：{path}")
    return value


CONFIG_PATH = Path(os.getenv("SUBALIGN_CONFIG", str(DEFAULT_CONFIG_PATH))).resolve()
_CONFIG = load_config_file(CONFIG_PATH)


def _config(section: str, key: str, default: Any) -> Any:
    group = _CONFIG.get(section, {})
    if not isinstance(group, dict):
        raise ValueError(f"config.yaml 中的 {section} 必须是对象")
    return group.get(key, default)


def _first_env(*names: str, default: Any = "") -> Any:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(*names: str, default: bool = False) -> bool:
    return _as_bool(_first_env(*names, default=default))


def _origins() -> tuple[str, ...]:
    env_value = os.getenv("SUBALIGN_ALLOW_ORIGINS")
    if env_value is not None:
        return tuple(x.strip() for x in env_value.split(",") if x.strip())
    value = _config("security", "allow_origins", [])
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    if not isinstance(value, list):
        raise ValueError("security.allow_origins 必须是 YAML 数组或逗号分隔字符串")
    return tuple(str(x).strip() for x in value if str(x).strip())


@dataclass(frozen=True)
class Settings:
    config_path: Path = CONFIG_PATH
    server_host: str = str(
        _first_env("SUBALIGN_HOST", default=_config("server", "host", "0.0.0.0"))
    )
    server_port: int = int(
        _first_env("SUBALIGN_PORT", default=_config("server", "port", 12045))
    )
    server_workers: int = int(
        _first_env("SUBALIGN_WORKERS", default=_config("server", "workers", 1))
    )
    data_dir: Path = Path(
        _first_env(
            "SUBALIGN_DATA_DIR", default=_config("storage", "data_dir", "./data")
        )
    ).expanduser().resolve()
    model_root: Path = Path(
        _first_env(
            "QWEN_MODEL_ROOT",
            default=_config("models", "root", "/data/yb/Code/models"),
        )
    ).expanduser()
    cuda_visible_devices: str = str(
        _first_env(
            "CUDA_VISIBLE_DEVICES", default=_config("gpu", "visible_devices", "5")
        )
    )
    max_upload_mb: int = int(
        _first_env(
            "SUBALIGN_MAX_UPLOAD_MB",
            default=_config("storage", "max_upload_mb", 4096),
        )
    )
    max_concurrent_jobs: int = int(
        _first_env(
            "SUBALIGN_MAX_CONCURRENT_JOBS",
            default=_config("gpu", "max_concurrent_jobs", 1),
        )
    )
    api_key: str = str(
        _first_env("SUBALIGN_API_KEY", default=_config("security", "api_key", ""))
    )
    allow_origins: tuple[str, ...] = _origins()
    flash_enabled: bool = _env_bool(
        "V4_FLASH_ENABLED",
        "V4_ENABLED",
        default=_as_bool(_config("v4_flash", "enabled", False)),
    )
    flash_base_url: str = str(
        _first_env(
            "V4_FLASH_BASE_URL",
            "V4_BASE_URL",
            default=_config(
                "v4_flash", "base_url", "http://10.185.1.71:8080/v1"
            ),
        )
    ).rstrip("/")
    flash_api_key: str = str(
        _first_env(
            "V4_FLASH_API_KEY",
            "V4_API_KEY",
            default=_config("v4_flash", "api_key", ""),
        )
    )
    flash_model: str = str(
        _first_env(
            "V4_FLASH_MODEL",
            "V4_MODEL",
            default=_config("v4_flash", "model", "dsv4"),
        )
    )
    flash_timeout_seconds: int = int(
        _first_env(
            "V4_FLASH_TIMEOUT_SECONDS",
            "V4_TIMEOUT_SECONDS",
            "REQUEST_TIMEOUT",
            default=_config("v4_flash", "timeout_seconds", 120),
        )
    )
    flash_retry_count: int = int(
        _first_env(
            "V4_FLASH_RETRY_COUNT",
            "RETRY_COUNT",
            default=_config("v4_flash", "retry_count", 2),
        )
    )
    flash_verify_ssl: bool = _env_bool(
        "V4_FLASH_VERIFY_SSL",
        default=_as_bool(_config("v4_flash", "verify_ssl", False)),
    )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


settings = Settings()
