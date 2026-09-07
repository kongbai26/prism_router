#!/bin/bash
# Prism Router 启动脚本（后台运行）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/prism_router.pid"
LOG_DIR="$PROJECT_DIR/logs/server"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_DIR"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Prism Router already running (PID: $PID)"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server.$(date +%Y-%m-%d).log"

echo "Prism Router starting..."
nohup $PYTHON -m prism_router server -y -c config.yaml >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 1
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Prism Router started (PID: $(cat "$PID_FILE")), log: $LOG_FILE"
else
    echo "Prism Router failed to start, check: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
