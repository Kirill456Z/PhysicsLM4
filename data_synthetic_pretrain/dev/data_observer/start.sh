#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Backend ──────────────────────────────────────────────────────────────────
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
poetry run uvicorn main:app --reload --port 8000 --app-dir "$BACKEND_DIR" &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────────────────────
echo ">>> Setting up frontend..."
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
  npm install
fi

echo ">>> Starting frontend on http://localhost:3000"
npm run dev &
FRONTEND_PID=$!

# ── Cleanup on exit ────────────────────────────────────────────────────────
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."

wait
