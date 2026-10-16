#!/usr/bin/env bash
# Chain train_video_cycle.sh: N cycles of STEP iterations each (report + video after each), stop early once
# the 64-pitch evaluation has >= GOAL pitches still up 1.5 s after the release.
#   bash scripts/train_cycles.sh <task> <checkpoint.pt> <cycles> <run_name> [step=150] [goal=60]
# KEEP_BEST=1: a cycle whose goal count is not above the best so far is dropped, the next resumes from the best
set -u
cd "$(dirname "$0")/.."
TASK=$1
CKPT=$2
N=$3
RUN=$4
STEP=${5:-150}
GOAL=${6:-60}
IT=$(basename "$CKPT" .pt | sed 's/model_//')
[ "$CKPT" = none ] && IT=0  # from scratch
BEST=${BEST0:--1}  # BEST0: the starting checkpoint's known goal count
BESTCK=$CKPT
echo "$TASK $RUN $IT $N $STEP" > logs/current_chain.txt  # read by train_progress.py --auto (status-line mod)
for _ in $(seq "$N"); do
    STOP=$((IT + STEP))
    [ -n "${KEEP_BEST:-}" ] && export SEED=$RANDOM  # a retry from the same checkpoint should not replay the same run
    bash scripts/train_video_cycle.sh "$TASK" "$CKPT" "$STOP" "$RUN" > "logs/cycle_${RUN}_$STOP.txt" 2>&1 || exit 1
    unset RESET_NORM ZERO_HEAD WARMUP ZERO_ARMS  # (ZERO_HEAD=1: zero the actor's output layer, first cycle only)
    # (RESET_NORM=1: refit the observation normalizers in the first cycle only - a new stage)
    CKPT=$(sed -n 's/^\[cycle\] next: //p' "logs/cycle_${RUN}_$STOP.txt")
    UP=$(sed -n 's/.*still up 1.5 s after the release: \([0-9]*\)\/.*/\1/p' "logs/cycle_${RUN}_$STOP.txt" | head -1)
    REST=$(sed -n 's/.*s after the release: \([0-9]*\)\/[0-9]* (failed:.*/\1/p' "logs/cycle_${RUN}_$STOP.txt" | head -1)
    [ -n "$REST" ] && UP=$REST  # stand2: at rest standing (stricter than "still up")
    GOALN=$(grep -o '^\[goal\] [0-9]*' "logs/cycle_${RUN}_$STOP.txt" | head -1 | cut -d' ' -f2)
    [ -n "$GOALN" ] && UP=$GOALN  # stage 1 steps (eval_stage.py)
    echo "[cycles] $STOP: $UP up ($CKPT)"
    [ -n "$UP" ] && [ "$UP" -ge "$GOAL" ] && { echo "[cycles] goal reached"; exit 0; }
    IT=$STOP
    if [ -n "${KEEP_BEST:-}" ]; then  # a cycle below the best: the next one resumes from the best checkpoint
        if [ -n "$UP" ] && [ "$UP" -gt "$BEST" ]; then BEST=$UP; BESTCK=$CKPT; else CKPT=$BESTCK; fi
        IT=$(basename "$CKPT" .pt | sed 's/model_//')
        echo "[cycles] best so far $BEST ($BESTCK)"
    fi
done
