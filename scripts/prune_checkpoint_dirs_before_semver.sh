#!/usr/bin/env bash
# Remove subdirectories whose names look like vMAJOR.MINOR.PATCH and sort before 0.2.0 (semver-like).
#
# Usage:
#   ./scripts/prune_checkpoint_dirs_before_semver.sh /path/to/mix_checkpoints              # dry-run (list only)
#   ./scripts/prune_checkpoint_dirs_before_semver.sh /path/to/mix_checkpoints --delete   # actually rm -rf
#
# Environment:
#   THRESHOLD — default 0.2.0 (dirs with version strictly less than this are removed)

set -euo pipefail

THRESHOLD="${THRESHOLD:-0.2.0}"
TARGET_DIR="${1:-.}"
DELETE=0
if [[ "${2:-}" == "--delete" ]]; then
  DELETE=1
fi
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Not a directory: $TARGET_DIR" >&2
  exit 1
fi

version_lt_threshold() {
  local ver="$1"
  # True if ver sorts strictly before THRESHOLD (sort -V).
  [[ "$(printf '%s\n' "$ver" "$THRESHOLD" | sort -V | head -n1)" == "$ver" && "$ver" != "$THRESHOLD" ]]
}

shopt -s nullglob
mapfile -t candidates < <(
  find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -name 'v*' -printf '%f\n' | sort -V
)

to_remove=()
for name in "${candidates[@]}"; do
  [[ "$name" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
  ver="${name#v}"
  if version_lt_threshold "$ver"; then
    to_remove+=("$TARGET_DIR/$name")
  fi
done

if ((${#to_remove[@]} == 0)); then
  echo "Nothing to remove under $TARGET_DIR (threshold < $THRESHOLD)"
  exit 0
fi

echo "Threshold: remove dirs with version < $THRESHOLD"
if [[ "$DELETE" -ne 1 ]]; then
  echo "DRY RUN — would remove ${#to_remove[@]} director(ies):"
  printf '  %s\n' "${to_remove[@]}"
  echo "Re-run with: $0 \"$TARGET_DIR\" --delete"
  exit 0
fi

echo "Removing ${#to_remove[@]} director(ies):"
printf '  %s\n' "${to_remove[@]}"
rm -rf -- "${to_remove[@]}"
echo "Done."
