#!/bin/bash
# Prism Router 停止脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/prism_router.pid"

# 方式1：通过 PID 文件停止
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Prism Router (PID: $PID)..."
        kill "$PID"
        for i in $(seq 1 5); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
        echo "Prism Router stopped"
    else
        echo "Process $PID not found"
    fi
    rm -f "$PID_FILE"
    exit 0
fi

# 方式2：通过进程名查找停止
PIDS=$(pgrep -f "prism_router server" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Stopping Prism Router (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill
    sleep 2
    PIDS=$(pgrep -f "prism_router server" 2>/dev/null)
    [ -n "$PIDS" ] && echo "$PIDS" | xargs kill -9
    echo "Prism Router stopped"
else
    echo "Prism Router is not running"
fi
