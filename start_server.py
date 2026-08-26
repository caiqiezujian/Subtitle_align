from __future__ import annotations

import os
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "缺少 config.yaml。请先执行：cp config.example.yaml config.yaml，"
            "然后修改配置再启动。"
        )

    os.environ.setdefault("SUBALIGN_CONFIG", str(CONFIG_PATH))
    from app.config import settings

    if settings.server_workers != 1:
        raise SystemExit("GPU 对齐服务要求 server.workers: 1，避免重复占用显存。")

    print(f"Config: {settings.config_path}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={settings.cuda_visible_devices}")
    print(f"Server: http://{settings.server_host}:{settings.server_port}")
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        workers=1,
    )


if __name__ == "__main__":
    main()
