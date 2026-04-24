#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Parse command line arguments
RUN_TESTS=true
INCREMENT_MINOR=false
CHECKPOINT_VERSION=""
DOWNLOAD_FINEWEB_EDU=false
DATASET_FRACTION="1.0"  # Download all by default
GPU="1.0"  # Default GPU count
NODE_POOL=""  # Optional: e.g. "h100" to request H100 GPUs
TASK="train_depo"  # Default task
CONFIG_NAME="depo_debug"  # Default config (without .yaml extension)
START_NEW_WANDB_RUN=false  # When set, start a new wandb run instead of continuing the checkpoint's run
WANDB_RUN_ID=""  # Explicit wandb run ID to resume (for runs created before deterministic IDs)
NAME_SUFFIX=""  # Optional suffix appended to the wandb run name
VALID_TASKS=("run_jupyter_notebook" "train_depo" "log_from_trained" "pretokenize")
PRETOKENIZE_JOBS=1  # Number of parallel pretokenize jobs (each processes a subset of chunks)
for arg in "$@"; do
    case $arg in
        --minor)
            INCREMENT_MINOR=true
            ;;
        --checkpoint-version=*)
            CHECKPOINT_VERSION="${arg#*=}"
            ;;
        --download-fineweb-edu)
            DOWNLOAD_FINEWEB_EDU=true
            ;;
        --dataset-fraction=*)
            DATASET_FRACTION="${arg#*=}"
            ;;
        --task=*)
            TASK="${arg#*=}"
            ;;
        --config=*)
            CONFIG_NAME="${arg#*=}"
            ;;
        --gpu=*)
            GPU="${arg#*=}"
            ;;
        --node-pool=*)
            NODE_POOL="${arg#*=}"
            ;;
        --start-new-wandb-run)
            START_NEW_WANDB_RUN=true
            ;;
        --wandb-run-id=*)
            WANDB_RUN_ID="${arg#*=}"
            ;;
        --name-suffix=*)
            NAME_SUFFIX="${arg#*=}"
            ;;
        --pretokenize-jobs=*)
            PRETOKENIZE_JOBS="${arg#*=}"
            ;;
        --no-testing)
            RUN_TESTS=false
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--minor] [--no-testing] [--checkpoint-version=VERSION] [--base-vocab-size=N] [--download-fineweb-edu] [--dataset-fraction=F] [--task=TASK] [--config=NAME] [--gpu=N] [--node-pool=POOL] [--start-new-wandb-run] [--wandb-run-id=ID] [--name-suffix=SUFFIX]"
            echo "  --no-testing: skip pytest (lingua_modified/tests + data_synthetic_pretrain/tests) before version bump and commit"
            echo "  TASK: run_jupyter_notebook, train_depo (default), log_from_trained, pretokenize"
            echo "  --pretokenize-jobs=N: number of parallel pretokenize jobs (default: 1)"
            echo "  CONFIG: config name without .yaml extension (default: depo_debug)"
            echo "  DATASET_FRACTION: fraction of dataset to download, 0.0-1.0 (default: 1.0)"
            echo "  GPU: number of GPUs to request (default: 1.0)"
            echo "  NODE_POOL: RunAI node pool for GPU type (e.g. h100 for H100 GPUs)"
            echo "  --start-new-wandb-run: use new version for wandb run name even when resuming from checkpoint"
            echo "  --wandb-run-id=ID: explicit wandb run ID to resume (overrides deterministic ID)"
            exit 1
            ;;
    esac
done

# Validate task
TASK_VALID=false
for valid in "${VALID_TASKS[@]}"; do
    if [ "$TASK" = "$valid" ]; then
        TASK_VALID=true
        break
    fi
done
if [ "$TASK_VALID" = false ]; then
    echo "Error: Invalid task '$TASK'. Must be one of: ${VALID_TASKS[*]}"
    exit 1
fi

# Run tests before any version or git changes. Pytest: 0 = all passed; non-zero = fail or error.
if [ "$RUN_TESTS" = true ]; then
    echo "Running tests via scripts/run_tests.sh (use --no-testing to skip)..."
    if ! "$REPO_ROOT/scripts/run_tests.sh"; then
        echo "ERROR: tests failed. Aborting (no version bump, no commit, no image push)." >&2
        exit 1
    fi
else
    echo "Skipping tests (--no-testing)."
fi

# Read current version from latest_tag.txt
CURRENT_TAG=$(cat latest_tag.txt)

# Parse the version (assuming format X.Y.Z)
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_TAG"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Increment version based on flag
if [ "$INCREMENT_MINOR" = true ]; then
    # Increment minor version and reset patch to 0
    MINOR=$((MINOR + 1))
    PATCH=0
else
    # Increment patch version
    PATCH=$((PATCH + 1))
fi

# Create new tag
NEW_TAG="${MAJOR}.${MINOR}.${PATCH}"

# Write new tag to latest_tag.txt
echo "$NEW_TAG" > latest_tag.txt

echo "Updated version from $CURRENT_TAG to $NEW_TAG"

# When --checkpoint-version is provided: use config from recipe_stashes instead of CONFIG_NAME
# When not: save current config to recipe_stashes with semver version
RECIPE_STASHES_DIR="recipe_stashes"
CONFIGS_DIR="lingua_modified/apps/main/configs"
RESUME_CONFIG_NAME=""
if [ -n "$CHECKPOINT_VERSION" ]; then
    # Pick config from recipe_stashes matching checkpoint version (*_VERSION.yaml)
    shopt -s nullglob
    STASH_CONFIGS=("${RECIPE_STASHES_DIR}"/*_"${CHECKPOINT_VERSION}".yaml)
    shopt -u nullglob
    if [ ${#STASH_CONFIGS[@]} -eq 0 ]; then
        echo "Error: No config found in recipe_stashes for checkpoint version ${CHECKPOINT_VERSION}"
        echo "Expected pattern: *_${CHECKPOINT_VERSION}.yaml"
        exit 1
    fi
    if [ ${#STASH_CONFIGS[@]} -gt 1 ]; then
        echo "Warning: Multiple configs found, using first: ${STASH_CONFIGS[0]}"
    fi
    RESUME_CONFIG_NAME="resume_${CHECKPOINT_VERSION}"
    echo "Using config from recipe_stashes: ${STASH_CONFIGS[0]} (will copy to configs as ${RESUME_CONFIG_NAME}.yaml)"
else
    # Save config copy to recipe_stashes with semver version
    CONFIG_SRC="${CONFIGS_DIR}/${CONFIG_NAME}.yaml"
    CONFIG_DST="${RECIPE_STASHES_DIR}/${CONFIG_NAME}_${NEW_TAG}.yaml"
    mkdir -p "$RECIPE_STASHES_DIR"
    if [ -f "$CONFIG_SRC" ]; then
        cp "$CONFIG_SRC" "$CONFIG_DST"
        echo "Saved config copy to $CONFIG_DST"
    else
        echo "Warning: Config file $CONFIG_SRC not found, skipping recipe stash"
    fi
fi

# Commit all files with the new tag version
echo "Committing changes..."
git add -A
git commit -m "Version ${NEW_TAG}"
echo "Committed changes with tag ${NEW_TAG}"

# When resuming from checkpoint: copy config from recipe_stashes to configs (after commit, so it's not committed)
# recipe_stashes is in dockerignore, so we need the config inside the build context
if [ -n "$RESUME_CONFIG_NAME" ]; then
    cp "${STASH_CONFIGS[0]}" "${CONFIGS_DIR}/${RESUME_CONFIG_NAME}.yaml"
    CONFIG_NAME="$RESUME_CONFIG_NAME"
    echo "Copied config to ${CONFIGS_DIR}/${RESUME_CONFIG_NAME}.yaml for Docker build"
fi

# Build Docker image
echo "Building Docker image..."
echo "Build context size: $(du -sh . 2>/dev/null | cut -f1) (excludes .dockerignore)"
docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/mlo-zemlians/physics4lm:${NEW_TAG} \
    --build-arg LDAP_GROUPNAME=mlo \
    --build-arg LDAP_GID=30133 \
    --build-arg LDAP_USERNAME=zemlians \
    --build-arg LDAP_UID=296871

# Remove copied config from configs folder (don't keep it locally)
if [ -n "$RESUME_CONFIG_NAME" ]; then
    rm -f "${CONFIGS_DIR}/${RESUME_CONFIG_NAME}.yaml"
    echo "Removed temporary config ${RESUME_CONFIG_NAME}.yaml from configs"
fi

# Push Docker image
echo "Pushing Docker image..."
docker push registry.rcp.epfl.ch/mlo-zemlians/physics4lm:${NEW_TAG}

# Submit runai job
echo "Submitting runai job..."
if [ -n "$CHECKPOINT_VERSION" ]; then
    echo "Using checkpoint version: $CHECKPOINT_VERSION"
fi
echo "Task: $TASK"
echo "Config: $CONFIG_NAME"
# Replace dots with underscores for job name
JOB_TAG=$(echo ${NEW_TAG} | tr '.' '-')

# Determine whether to continue existing wandb run or start a new one
# By default, when resuming from a checkpoint, continue the checkpoint's wandb run
CONTINUE_WANDB_RUN=0
if [ "$START_NEW_WANDB_RUN" = false ] && [ -n "$CHECKPOINT_VERSION" ]; then
    NAMING_TAG="${CHECKPOINT_VERSION}"
    CONTINUE_WANDB_RUN=1
    echo "Continuing wandb run: using checkpoint version ${CHECKPOINT_VERSION} for run naming"
else
    NAMING_TAG="${NEW_TAG}"
fi

# Build environment variables list
ENV_VARS="--environment WANDB_API_KEY=SECRET:wandb-secret,secret1 --environment HF_TOKEN=SECRET:hf-secret,secret1 --environment SEMVER_TAG=${NAMING_TAG}"
ENV_VARS="$ENV_VARS --environment CONTINUE_WANDB_RUN=${CONTINUE_WANDB_RUN}"
if [ -n "$WANDB_RUN_ID" ]; then
    ENV_VARS="$ENV_VARS --environment WANDB_RUN_ID=${WANDB_RUN_ID}"
fi
if [ -n "$NAME_SUFFIX" ]; then
    ENV_VARS="$ENV_VARS --environment NAME_SUFFIX=${NAME_SUFFIX}"
fi
if [ -n "$CHECKPOINT_VERSION" ]; then
    ENV_VARS="$ENV_VARS --environment CHECKPOINT_VERSION=${CHECKPOINT_VERSION}"
fi
if [ "$DOWNLOAD_FINEWEB_EDU" = true ]; then
    ENV_VARS="$ENV_VARS --environment DOWNLOAD_FINEWEB_EDU=1"
fi
ENV_VARS="$ENV_VARS --environment CONFIG_NAME=${CONFIG_NAME}"
ENV_VARS="$ENV_VARS --environment DATASET_FRACTION=${DATASET_FRACTION}"
if [ "$TASK" = "pretokenize" ]; then
    ENV_VARS="$ENV_VARS --environment PRETOKENIZE_DATA_DIR=/scratch/zemlians/physics4lm/fineweb_edu/fineweb_edu"
fi

# Task entrypoints in scripts/; pretokenize and log_from_trained stay at repo root
case "$TASK" in
    run_jupyter_notebook|train_depo)
        TASK_SCRIPT="scripts/${TASK}.sh"
        ;;
    *)
        TASK_SCRIPT="${TASK}.sh"
        ;;
esac

NODE_POOL_ARGS=""
if [ -n "$NODE_POOL" ]; then
    NODE_POOL_ARGS="--node-pools ${NODE_POOL}"
fi

# Pretokenize is CPU-only: use 0 GPUs
SUBMIT_GPU="${GPU}"
if [ "$TASK" = "pretokenize" ]; then
    SUBMIT_GPU="0"
fi

if [ "$TASK" = "pretokenize" ]; then
    echo "Submitting ${PRETOKENIZE_JOBS} pretokenize job(s)..."
    for job_idx in $(seq 0 $((PRETOKENIZE_JOBS - 1))); do
        JOB_ENV="$ENV_VARS --environment PRETOKENIZE_RANK=${job_idx} --environment PRETOKENIZE_WORLD_SIZE=${PRETOKENIZE_JOBS}"
        runai submit \
          --name pretokenize-${JOB_TAG}-${job_idx} \
          --image registry.rcp.epfl.ch/mlo-zemlians/physics4lm:${NEW_TAG} \
          --gpu ${SUBMIT_GPU} \
          $NODE_POOL_ARGS \
          $JOB_ENV \
          --existing-pvc claimname=mlo-scratch,path=/scratch \
          --existing-pvc claimname=home,path=/home/zemlians \
          --backoff-limit 0 \
          --command -- /bin/bash -lc "chmod a+x ${TASK_SCRIPT} && ./${TASK_SCRIPT}"
        echo "  Submitted: pretokenize-${JOB_TAG}-${job_idx}"
    done
else
    runai submit \
      --name depo-debug-train-${JOB_TAG} \
      --image registry.rcp.epfl.ch/mlo-zemlians/physics4lm:${NEW_TAG} \
      --gpu ${SUBMIT_GPU} \
      $NODE_POOL_ARGS \
      $ENV_VARS \
      --existing-pvc claimname=mlo-scratch,path=/scratch \
      --existing-pvc claimname=home,path=/home/zemlians \
      --backoff-limit 0 \
      --command -- /bin/bash -lc "chmod a+x ${TASK_SCRIPT} && ./${TASK_SCRIPT}"
fi

echo "Done! Job(s) submitted with tag ${NEW_TAG}"