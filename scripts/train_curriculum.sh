#!/usr/bin/env bash
# Fall-fix curriculum (arm + ball, option (b)): train_cycles.sh on each task in turn until its 64-episode goal passes,
# the passing checkpoint starting the next task. Stops if a task does not pass within its cycles.
#   bash scripts/train_curriculum.sh <checkpoint.pt> <run_prefix> <cycles_per_task> <task>...
# GOAL_MID=k: the tasks before the last one pass at k of 64 (our own curriculum steps); the last one always needs 60
set -u
cd "$(dirname "$0")/.."
CKPT=$1
PRE=$2
N=$3
shift 3
for TASK in "$@"; do
    RUN="${PRE}_$(echo "$TASK" | sed 's/^AIB-Stage//; s/-v0$//' | tr 'A-Z' 'a-z')"
    echo "[curriculum] $TASK from $CKPT (run $RUN)"
    G=${GOAL_MID:-60}
    [ "$TASK" = "${!#}" ] && G=60
    bash scripts/train_cycles.sh "$TASK" "$CKPT" "$N" "$RUN" "${STEP:-150}" "$G" > "logs/chain_$RUN.txt" 2>&1
    cat "logs/chain_$RUN.txt" | grep "^\[cycles\]"
    grep -q "goal reached" "logs/chain_$RUN.txt" || { echo "[curriculum] $TASK did not pass - stopped"; exit 1; }
    CKPT=$(sed -n 's/^\[cycles\] [0-9]*: [0-9]* up (\(.*\))$/\1/p' "logs/chain_$RUN.txt" | tail -1)
    unset WARMUP RESET_NORM ZERO_HEAD ZERO_ARMS BEST0
done
echo "[curriculum] all passed: $CKPT"
