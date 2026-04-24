#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# ── Backend ──────────────────────────────────────────────────────────────────
echo ">>> Setting up backend..."
cd "$BACKEND_DIR"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

echo ">>> Starting backend on http://localhost:8000 (API docs: http://localhost:8000/docs)"
uvicorn main:app --reload --port 8000 &
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
