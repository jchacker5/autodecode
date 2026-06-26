#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv/bin"
MODEL="$ROOT/models/ornith-9b-4bit"
PORT="${MLX_SERVER_PORT:-8080}"
HOST="${MLX_SERVER_HOST:-127.0.0.1}"
PIDFILE="$ROOT/.mlx-server.pid"
LOGFILE="$ROOT/mlx-server.log"

if [[ ! -d "$MODEL" ]]; then
  echo "MLX model not found at $MODEL" >&2
  exit 1
fi

if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "MLX server already running (pid $OLD_PID) on http://$HOST:$PORT"
    exit 0
  fi
  rm -f "$PIDFILE"
fi

nohup "$VENV/mlx_lm.server" \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --temp 0.05 \
  --prefill-step-size 2048 \
  --chat-template-args '{"enable_thinking":false}' \
  --max-tokens 4096 \
  >"$LOGFILE" 2>&1 &

echo $! >"$PIDFILE"
echo "Started MLX server pid $(cat "$PIDFILE")"
echo "URL: http://$HOST:$PORT/v1"
echo "Log: $LOGFILE"
echo "OpenCode: mlx/default_model | Hermes: custom:mlx-ornith:default_model"