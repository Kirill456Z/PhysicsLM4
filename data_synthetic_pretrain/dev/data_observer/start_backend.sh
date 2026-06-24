#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo ">>> Setting up backend..."
cd "$REPO_ROOT"

if ! command -v poetry >/dev/null 2>&1; then
  echo "ERROR: poetry is not installed or not in PATH." >&2
  exit 1
fi

# Install backend deps into the Poetry environment if they are missing.
if ! poetry run python -c "import fastapi, uvicorn, pydantic, yaml, numpy" >/dev/null 2>&1; then
  poetry run pip install -q -r "$BACKEND_DIR/requirements.txt"
fi

echo ">>> Starting backend on http://localhost:8000 (API docs: http://localhost:8000/docs)"
poetry run uvicorn main:app --reload --port 8000 --app-dir "$BACKEND_DIR"
