#!/bin/bash
set -e

# Parse command line arguments (default CHECKPOINT_VERSION and CONFIG_NAME from env vars)
CHECKPOINT_VERSION="${CHECKPOINT_VERSION:-}"
CONFIG_NAME="${CONFIG_NAME:-depo_debug}"
CONFIG="apps/main/configs/${CONFIG_NAME}.yaml"
OUTPUT_DIR=""

for arg in "$@"; do
    case $arg in
        --checkpoint-version=*)
            CHECKPOINT_VERSION="${arg#*=}"
            ;;
        --config=*)
            CONFIG="${arg#*=}"
            ;;
        --output-dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 --checkpoint-version=VERSION [--config=PATH] [--output-dir=PATH]"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT_VERSION" ]; then
    echo "Error: --checkpoint-version is required"
    echo "Usage: $0 --checkpoint-version=VERSION [--config=PATH] [--output-dir=PATH]"
    exit 1
fi

echo "========================================"
echo "Canon Layer Snapshot"
echo "  Checkpoint version: $CHECKPOINT_VERSION"
echo "  Config:             $CONFIG"
echo "========================================"

cd lingua_modified

CMD="python -m lingua.log_from_trained --config $CONFIG --checkpoint-version $CHECKPOINT_VERSION"
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

$CMD

echo "Done!"
