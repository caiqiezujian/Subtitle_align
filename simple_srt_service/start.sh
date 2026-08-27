#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SERVICE_DIR/.." && pwd)"

if [[ ! -f "$SERVICE_DIR/config.yaml" ]]; then
  cp "$SERVICE_DIR/config.example.yaml" "$SERVICE_DIR/config.yaml"
  echo "已创建 simple_srt_service/config.yaml，请修改模型路径和 GPU 编号后重新启动。"
  exit 1
fi

cd "$PROJECT_ROOT"
exec python3 -m simple_srt_service.run

