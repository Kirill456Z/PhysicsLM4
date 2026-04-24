#!/usr/bin/env bash
# Run tests under lingua_modified/tests and data_synthetic_pretrain/tests using the
# Docker training image, or the local Python when USE_DOCKER=0.
#
# Examples:
#   ./scripts/run_tests.sh
#   ./scripts/run_tests.sh data_synthetic_pretrain/tests -v
#   USE_DOCKER=0 ./scripts/run_tests.sh
#
# Env:
#   PHYSICSLM4_DOCKER_IMAGE  Image tag (default: physics4lm:local). Build: docker build -t physics4lm:local ...
#   DOCKER_PLATFORM            e.g. linux/amd64 on Apple Silicon (default: linux/amd64)
#   USE_DOCKER                 1 = docker (default), 0 = run pytest in current environment

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Default targets: lingua_modified/tests and data_synthetic_pretrain/tests. If the first
# arg is a flag (e.g. -v, -k), prepend those dirs. If you pass a path, only that path is used.
_DEFAULT_TESTS=(lingua_modified/tests data_synthetic_pretrain/tests)
if [[ $# -eq 0 ]]; then
  set -- "${_DEFAULT_TESTS[@]}"
elif [[ "${1-}" == -* ]]; then
  set -- "${_DEFAULT_TESTS[@]}" "$@"
fi
unset _DEFAULT_TESTS

USE_DOCKER="${USE_DOCKER:-1}"
export PYTHONPATH="${ROOT}/lingua_modified:${ROOT}"

if [[ "$USE_DOCKER" == "0" ]]; then
  exec python -m pytest "$@"
fi

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install Docker or set USE_DOCKER=0 with deps installed." >&2
  exit 1
fi

IMAGE="${PHYSICSLM4_DOCKER_IMAGE:-physics4lm:local}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

# Inner bash: dummy $0; "$@" is forwarded to pytest.
exec docker run --rm \
  --platform "$PLATFORM" \
  -v "$ROOT":/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace:/workspace/lingua_modified \
  "$IMAGE" \
  /bin/bash -c 'pip install -q --user pytest && export PATH="${HOME}/.local/bin:${PATH}" && exec python -m pytest "$@"' _ "$@"
