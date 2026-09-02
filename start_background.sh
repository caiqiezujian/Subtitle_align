#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/subtitle-align.pid"
LOG_FILE="$SCRIPT_DIR/subtitle-align.log"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "服务已经在运行，PID=$EXISTING_PID"
    echo "日志：$LOG_FILE"
    exit 0
  fi
fi

cd "$SCRIPT_DIR"
nohup bash "$SCRIPT_DIR/start.sh" > "$LOG_FILE" 2>&1 &
SERVICE_PID=$!
echo "$SERVICE_PID" > "$PID_FILE"

sleep 2
if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
  echo "服务启动失败，最近日志如下："
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

echo "服务已在后台启动，PID=$SERVICE_PID"
echo "日志：$LOG_FILE"
echo "查看日志：tail -f '$LOG_FILE'"
