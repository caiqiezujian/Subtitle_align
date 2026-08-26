#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "已创建 config.yaml。请填写模型路径和 v4-flash Key，然后重新执行 bash start.sh。"
  exit 1
fi

exec python start_server.py
