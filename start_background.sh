#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/subtitle-align.pid"
LOG_FILE="$SCRIPT_DIR/subtitle-align.log"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STARTUP_WAIT_SECONDS="${SUBALIGN_STARTUP_WAIT_SECONDS:-1800}"

cd "$SCRIPT_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
  cp "$SCRIPT_DIR/config.example.yaml" "$CONFIG_FILE"
  echo "已创建 config.yaml。请填写模型路径等配置，然后重新执行 bash start_background.sh。"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 $PYTHON_BIN，请确认容器已经安装 Python 3。"
  exit 1
fi

if [[ ! "$STARTUP_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUBALIGN_STARTUP_WAIT_SECONDS 必须是正整数。"
  exit 1
fi

SERVER_PORT="$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
server = value.get("server") or {}
print(int(server.get("port", 12045)))
PY
)"
HEALTH_URL="http://127.0.0.1:${SERVER_PORT}/api/health"

health_ready() {
  "$PYTHON_BIN" - "$SERVER_PORT" <<'PY' >/dev/null 2>&1
import http.client
import json
import sys

connection = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=2)
try:
    connection.request("GET", "/api/health")
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
finally:
    connection.close()

if response.status != 200:
    raise SystemExit(1)
if payload.get("status") != "ok":
    raise SystemExit(1)
if payload.get("gpu_worker") != "ready" or payload.get("models_resident") is not True:
    raise SystemExit(1)
PY
}

if health_ready; then
  echo "服务已经就绪：$HEALTH_URL"
  echo "日志：$LOG_FILE"
  exit 0
fi

SERVICE_PID=""
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    SERVICE_PID="$EXISTING_PID"
    echo "检测到服务进程 PID=$SERVICE_PID，继续等待模型和接口就绪。"
  fi
fi

if [[ -z "$SERVICE_PID" ]]; then
  nohup bash "$SCRIPT_DIR/start.sh" > "$LOG_FILE" 2>&1 &
  SERVICE_PID=$!
  echo "$SERVICE_PID" > "$PID_FILE"
  echo "服务进程已启动，PID=$SERVICE_PID；正在等待模型加载。"
fi

STARTED_AT=$SECONDS
LAST_REPORT_AT=-10
while (( SECONDS - STARTED_AT < STARTUP_WAIT_SECONDS )); do
  if health_ready; then
    echo "服务已真正就绪，PID=$SERVICE_PID"
    echo "健康检查：$HEALTH_URL"
    echo "日志：$LOG_FILE"
    exit 0
  fi

  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "服务进程已经退出，启动失败。最近日志如下："
    tail -n 120 "$LOG_FILE" || true
    exit 1
  fi

  ELAPSED=$((SECONDS - STARTED_AT))
  if (( ELAPSED - LAST_REPORT_AT >= 10 )); then
    echo "等待模型与接口就绪：${ELAPSED}s / ${STARTUP_WAIT_SECONDS}s"
    tail -n 3 "$LOG_FILE" 2>/dev/null || true
    LAST_REPORT_AT=$ELAPSED
  fi
  sleep 2
done

echo "等待 ${STARTUP_WAIT_SECONDS}s 后服务仍未就绪，但进程 PID=$SERVICE_PID 仍在运行。"
echo "请检查配置和模型加载日志：tail -f '$LOG_FILE'"
echo "最近日志如下："
tail -n 120 "$LOG_FILE" || true
exit 1
