#!/bin/bash
set -e

DATA_DIR="${PRETOKENIZE_DATA_DIR:-/scratch/zemlians/physics4lm/fineweb_edu/fineweb_edu}"
RANK="${PRETOKENIZE_RANK:-0}"
WORLD_SIZE="${PRETOKENIZE_WORLD_SIZE:-1}"

echo "========================================"
echo "Pre-tokenizing FineWeb-Edu data..."
echo "Data directory: ${DATA_DIR}"
echo "Rank: ${RANK} / World size: ${WORLD_SIZE}"
echo "========================================"

python scripts/pretokenize.py "${DATA_DIR}" --rank "${RANK}" --world-size "${WORLD_SIZE}"

echo "Pre-tokenization complete."
