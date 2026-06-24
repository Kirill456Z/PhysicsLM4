#!/bin/bash

# Suppress deprecation message
export SUPPRESS_DEPRECATION_MESSAGE=true

# Disable pager and force plain output - runai may not output when stdout is not a TTY
export PAGER=cat
export GIT_PAGER=cat
export TERM=dumb

# Project from runai config (override with RUNAI_PROJECT env var)
PROJECT="${RUNAI_PROJECT:-mlo-zemlians}"

[[ "$1" == "--debug" ]] && DEBUG=1 || DEBUG=0

echo "Fetching jobs (project: $PROJECT)..."

# Try JSON first (more reliable in scripts) - runai may not output table when not a TTY
json_output=$(runai list jobs --json -p "$PROJECT" 2>&1)
json_ok=$?

deleted_count=0
if [[ $json_ok -eq 0 ]] && command -v jq &>/dev/null; then
    # Parse JSON: handle both {items: [...]} and [...] structures, status at .status or .status.phase
    jobs=$(echo "$json_output" | jq -r '
        (if type == "array" then . else .items // [] end)[] |
        select((.status.phase // .status // "") != "Running") |
        .metadata.name // .name
    ' 2>/dev/null)
    [[ $DEBUG -eq 1 ]] && { echo "--- JSON jobs ---"; echo "$jobs"; echo "--- End ---"; }
fi

# Use table parsing if JSON failed or produced no jobs (jobs empty or whitespace-only)
jobs_trimmed="${jobs//[[:space:]]/}"
if [[ -z "$jobs_trimmed" ]] || [[ "$json_ok" -ne 0 ]] || ! command -v jq &>/dev/null; then
    output=$(runai list jobs -p "$PROJECT" 2>&1) || { echo "Failed to list jobs"; exit 1; }
    # Strip ANSI escape codes (colors) - they break status matching
    output=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')
    [[ $DEBUG -eq 1 ]] && { echo "--- Raw table output ---"; echo "$output"; echo "--- End ---"; }
    # Filter to job lines only (exclude header and non-job lines)
    job_lines=$(echo "$output" | grep -E '^[^ ]+ +(Succeeded|Failed|Suspended|Pending) +')
    [[ $DEBUG -eq 1 ]] && { echo "--- Filtered job lines ---"; echo "$job_lines"; echo "--- End ---"; }
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        job_name=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $2}')
        [[ -z "$job_name" || -z "$status" ]] && continue
        read -p "Delete job $job_name (status: $status)? [y/N] " -n 1 -r < /dev/tty
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            runai delete job "$job_name" -p "$PROJECT" && ((deleted_count++)) || echo "Failed to delete $job_name"
        fi
    done <<< "$job_lines"
else
    while IFS= read -r job_name; do
        [[ -z "$job_name" ]] && continue
        status=$(echo "$json_output" | jq -r --arg n "$job_name" '
            (if type == "array" then . else .items // [] end)[] |
            select((.metadata.name // .name) == $n) |
            .status.phase // .status // "Unknown"
        ' 2>/dev/null | head -1)
        read -p "Delete job $job_name (status: $status)? [y/N] " -n 1 -r < /dev/tty
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            runai delete job "$job_name" -p "$PROJECT" && ((deleted_count++)) || echo "Failed to delete $job_name"
        fi
    done <<< "$jobs"
fi

echo "Done. Deleted ${deleted_count:-0} job(s)."
