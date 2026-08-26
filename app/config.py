from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("SUBALIGN_DATA_DIR", "./data")).resolve()
    model_root: Path = Path(
        os.getenv("QWEN_MODEL_ROOT", "/data/yb/Code/models")
    ).expanduser()
    cuda_visible_devices: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    max_upload_mb: int = _int("SUBALIGN_MAX_UPLOAD_MB", 4096)
    max_concurrent_jobs: int = _int("SUBALIGN_MAX_CONCURRENT_JOBS", 1)
    api_key: str = os.getenv("SUBALIGN_API_KEY", "")
    allow_origins: tuple[str, ...] = tuple(
        x.strip()
        for x in os.getenv("SUBALIGN_ALLOW_ORIGINS", "").split(",")
        if x.strip()
    )
    flash_enabled: bool = _bool(
        "V4_FLASH_ENABLED", _bool("V4_ENABLED", False)
    )
    flash_base_url: str = _first(
        "V4_FLASH_BASE_URL",
        "V4_BASE_URL",
        default="http://10.185.1.71:8080/v1",
    ).rstrip("/")
    flash_api_key: str = _first("V4_FLASH_API_KEY", "V4_API_KEY")
    flash_model: str = _first("V4_FLASH_MODEL", "V4_MODEL", default="dsv4")
    flash_timeout_seconds: int = int(
        _first(
            "V4_FLASH_TIMEOUT_SECONDS",
            "V4_TIMEOUT_SECONDS",
            "REQUEST_TIMEOUT",
            default="120",
        )
    )
    flash_retry_count: int = int(
        _first("V4_FLASH_RETRY_COUNT", "RETRY_COUNT", default="2")
    )
    flash_verify_ssl: bool = _bool("V4_FLASH_VERIFY_SSL", False)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


settings = Settings()
