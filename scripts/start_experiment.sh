#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_NAME=""
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --exp_name=*)
            EXP_NAME="${arg#*=}"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo "Usage: $0 --exp_name=NAME [additional rebuild_and_start_job.sh args]"
    echo "  Finds all .yaml configs in lingua_modified/apps/main/configs/exps/NAME"
    echo "  and submits one training job per config with wandb run name {exp_name}_{config_stem}."
    exit 1
fi

EXP_DIR="lingua_modified/apps/main/configs/exps/${EXP_NAME}"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment folder not found: $EXP_DIR"
    exit 1
fi

shopt -s nullglob
CONFIGS=("${EXP_DIR}"/*.yaml)
shopt -u nullglob

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: No .yaml files found in $EXP_DIR"
    exit 1
fi

echo "Starting experiment: $EXP_NAME"
echo "Found ${#CONFIGS[@]} config(s) in $EXP_DIR"

for config_path in "${CONFIGS[@]}"; do
    file_stem="$(basename "$config_path" .yaml)"
    wandb_name="${EXP_NAME}_${file_stem}"
    config_name="exps/${EXP_NAME}/${file_stem}"

    echo ""
    echo "--- Config: $file_stem | wandb run: $wandb_name ---"

    "$SCRIPT_DIR/rebuild_and_start_job.sh" \
        --no-testing \
        --config="${config_name}" \
        --wandb-base-name="${wandb_name}" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "Done! Submitted ${#CONFIGS[@]} job(s) for experiment: $EXP_NAME"
