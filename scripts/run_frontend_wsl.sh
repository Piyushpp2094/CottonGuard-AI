#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
mkdir -p "$PROJECT_DIR/artifacts/logs"
export PATH="$HOME/.local/node/bin:$PATH"

# Resolve Windows host IP dynamically for Vite proxy when backend runs in Windows .venv
if [ -z "$VITE_PROXY_TARGET" ]; then
    WIN_HOST_IP=$(ip route show default 2>/dev/null | awk '{print $3}')
    if [ -n "$WIN_HOST_IP" ]; then
        export VITE_PROXY_TARGET="http://${WIN_HOST_IP}:8000"
    else
        export VITE_PROXY_TARGET="http://127.0.0.1:8000"
    fi
fi

cd "$PROJECT_DIR/frontend"
echo $$ > /tmp/cotton_leaf_frontend.pid
exec npm run dev -- --host 0.0.0.0 --port 5173 2>&1 | tee -a "$PROJECT_DIR/artifacts/logs/frontend.log"
