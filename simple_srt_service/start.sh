#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SERVICE_DIR/.." && pwd)"

if [[ ! -f "$SERVICE_DIR/config.yaml" ]]; then
  cp "$SERVICE_DIR/config.example.yaml" "$SERVICE_DIR/config.yaml"
  echo "已创建 simple_srt_service/config.yaml，请修改模型路径和 NPU 编号后重新启动。"
  exit 1
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
python3 -m simple_srt_service.check_environment
exec python3 -m simple_srt_service.run
