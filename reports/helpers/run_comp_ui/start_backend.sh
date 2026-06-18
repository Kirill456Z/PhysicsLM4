#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo ">>> Setting up backend..."

if ! python -c "import fastapi, uvicorn, pydantic, yaml, dotenv" >/dev/null 2>&1; then
  pip install -q -r "$BACKEND_DIR/requirements.txt"
fi

echo ">>> Starting backend on http://localhost:8000 (docs: http://localhost:8000/docs)"
python -m uvicorn main:app --reload --port 8000 --app-dir "$BACKEND_DIR"
