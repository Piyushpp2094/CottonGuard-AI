#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PROJECT_DIR/artifacts/logs"
cd "$PROJECT_DIR"
echo $$ > /tmp/cotton_leaf_backend.pid
exec "$HOME/.venvs/cotton_leaf_ai/bin/python" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > "$PROJECT_DIR/artifacts/logs/backend.log" 2>&1
