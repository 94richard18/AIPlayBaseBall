#!/usr/bin/env bash
# Stage 1 plan A (approved 2026-10-12): reward v2 + survival shaping (AIB-PitchStand3-v0) from a checkpoint for
# 300 iterations; if it has no effect, train the same task from scratch without asking (the user's decision).
#   bash scripts/plan_a_then_scratch.sh <checkpoint.pt> [more_cycles=8] [scratch_cycles=10]
# Effect = any pitch at rest standing, or the median fall >= MIN_FALL_S after the release (v2 at 20500: +0.40 s).
set -u
cd "$(dirname "$0")/.."
CKPT=$1
MORE=${2:-8}
SCRATCH=${3:-10}
MIN_FALL_S=0.8
TASK=AIB-PitchStand3-v0
PY=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
IT=$(basename "$CKPT" .pt | sed 's/model_//')
# the value scale changes (+100 alive, graded fall): start the critic head from zero
START=logs/rsl_rl/aib1_pitch/$(date +%Y-%m-%d_%H-%M-%S)_v3start
mkdir -p "$START"
$PY scripts/reset_critic_head.py "$CKPT" "$START/model_$IT.pt"

bash scripts/train_cycles.sh "$TASK" "$START/model_$IT.pt" 2 v3 || exit 1
effect=0
for STOP in $((IT + 150)) $((IT + 300)); do
    R="logs/cycle_v3_$STOP.txt"
    REST=$(sed -n 's/.*s after the release: \([0-9]*\)\/[0-9]* (failed:.*/\1/p' "$R")
    FALL=$(sed -n 's/.*fall median +\([0-9.]*\) s after the release.*/\1/p' "$R" | head -1)
    echo "[plan A] $STOP: at rest $REST, fall median +${FALL:-none} s"
    { [ -n "$REST" ] && [ "$REST" -ge 1 ]; } && effect=1
    { [ -z "$FALL" ] || awk "BEGIN{exit !($FALL >= $MIN_FALL_S)}"; } && effect=1
done
LAST=$(sed -n 's/^\[cycle\] next: //p' "logs/cycle_v3_$((IT + 300)).txt")
if [ $effect = 1 ]; then
    echo "[plan A] effect: continue $MORE more cycles from $LAST"
    bash scripts/train_cycles.sh "$TASK" "$LAST" "$MORE" v3
else
    echo "[plan A] no effect: train $TASK from scratch, $SCRATCH cycles"
    bash scripts/train_cycles.sh "$TASK" none "$SCRATCH" v3scratch
fi
