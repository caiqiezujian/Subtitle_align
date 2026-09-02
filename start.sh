#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "已创建 config.yaml。请填写模型路径和 v4-flash Key，然后重新执行 bash start.sh。"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 $PYTHON_BIN，请确认容器已经安装 Python 3。"
  exit 1
fi

# 后台重定向日志时也立即刷新启动进度，避免模型加载期间看起来“没有日志”。
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" start_server.py
