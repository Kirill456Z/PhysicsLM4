#!/usr/bin/env bash
# Run tests under lingua_modified/tests and data_synthetic_pretrain/tests using the
# Docker training image, or the local Python when USE_DOCKER=0.
#
# With Docker, pytest temp dirs, TMPDIR, and JUnit output are written under
#   ${TEST_ARTIFACTS_DIR} (default: <repo>/.test-artifacts)
# That directory is the same on the host because the repository root is
# bind-mounted to /workspace in the container (no separate docker cp step).
# After the run, open TEST_ARTIFACTS_DIR on your machine.
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
#   TEST_ARTIFACTS_DIR         Host path under the repo for basetemp, tmp, junit (default: <repo>/.test-artifacts)

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

TEST_ARTIFACTS_DIR="${TEST_ARTIFACTS_DIR:-${ROOT}/.test-artifacts}"
case "${TEST_ARTIFACTS_DIR}" in
  "$ROOT"/*)
    T_REL="${TEST_ARTIFACTS_DIR#"$ROOT"/}"
    ARTIFACTS_IN_CONTAINER="/workspace/${T_REL}"
    ;;
  *)
    echo "ERROR: TEST_ARTIFACTS_DIR must be a subdirectory of the repository: $ROOT" >&2
    exit 1
    ;;
esac

mkdir -p "${TEST_ARTIFACTS_DIR}/basetemp" "${TEST_ARTIFACTS_DIR}/tmp" "${TEST_ARTIFACTS_DIR}/junit"

_run_pytest_with_host_artifacts() {
  export TMPDIR="${TEST_ARTIFACTS_DIR}/tmp"
  exec python -m pytest \
    --basetemp "${TEST_ARTIFACTS_DIR}/basetemp" \
    --junitxml="${TEST_ARTIFACTS_DIR}/junit/junit.xml" \
    -o junit_family=legacy \
    "$@"
}

if [[ "$USE_DOCKER" == "0" ]]; then
  _run_pytest_with_host_artifacts "$@"
fi

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install Docker or set USE_DOCKER=0 with deps installed." >&2
  exit 1
fi

IMAGE="${PHYSICSLM4_DOCKER_IMAGE:-physics4lm:local}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

docker run --rm \
  --platform "$PLATFORM" \
  -v "$ROOT":/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace:/workspace/lingua_modified \
  -e "TEST_ARTIFACTS_IN_CONTAINER=${ARTIFACTS_IN_CONTAINER}" \
  -e "TMPDIR=${ARTIFACTS_IN_CONTAINER}/tmp" \
  "$IMAGE" \
  /bin/bash -c \
'set -euo pipefail
T="${TEST_ARTIFACTS_IN_CONTAINER:-/workspace/.test-artifacts}"
pip install -q --user pytest
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "$T/basetemp" "$T/tmp" "$T/junit"
export TMPDIR="$T/tmp"
exec python -m pytest \
  --basetemp "$T/basetemp" \
  --junitxml "$T/junit/junit.xml" \
  -o junit_family=legacy \
  "$@"' \
  _ "$@"

ec=$?
echo "Test artifacts on the host: ${TEST_ARTIFACTS_DIR}/" >&2
echo "  (junit, pytest basetemp, TMPDIR) — paths mirror ${ARTIFACTS_IN_CONTAINER} in the container." >&2
exit "$ec"
