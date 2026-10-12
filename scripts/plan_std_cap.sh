#!/usr/bin/env bash
# Stage 1 (approved 2026-10-13): no entropy bonus, action noise std capped at 0.5 (PitchStand3PPORunnerCfg), from a
# checkpoint for 300 iterations; with no effect, from scratch with the same settings (the user's standing decision).
#   bash scripts/plan_std_cap.sh <checkpoint.pt> [more_cycles=6] [scratch_cycles=10]
# Effect (noise-free evaluation) = any pitch at rest standing, or >= MIN_UP pitches still up 1.5 s after the release.
set -u
cd "$(dirname "$0")/.."
CKPT=$1
MORE=${2:-6}
SCRATCH=${3:-10}
MIN_UP=8
TASK=AIB-PitchStand3-v0
IT=$(basename "$CKPT" .pt | sed 's/model_//')
bash scripts/train_cycles.sh "$TASK" "$CKPT" 2 v5 || exit 1
effect=0
for STOP in $((IT + 150)) $((IT + 300)); do
    R="logs/cycle_v5_$STOP.txt"
    REST=$(sed -n 's/.*s after the release: \([0-9]*\)\/[0-9]* (failed:.*/\1/p' "$R" | head -1)
    UP=$(sed -n 's/.*still up 1.5 s after the release: \([0-9]*\)\/.*/\1/p' "$R" | head -1)
    echo "[std cap] $STOP: at rest ${REST:-?}, still up ${UP:-?}"
    { [ -n "$REST" ] && [ "$REST" -ge 1 ]; } && effect=1
    { [ -n "$UP" ] && [ "$UP" -ge $MIN_UP ]; } && effect=1
done
LAST=$(sed -n 's/^\[cycle\] next: //p' "logs/cycle_v5_$((IT + 300)).txt")
if [ $effect = 1 ]; then
    echo "[std cap] effect: continue $MORE more cycles from $LAST"
    bash scripts/train_cycles.sh "$TASK" "$LAST" "$MORE" v5
else
    echo "[std cap] no effect: train $TASK from scratch, $SCRATCH cycles"
    bash scripts/train_cycles.sh "$TASK" none "$SCRATCH" v5scratch
fi
