#!/usr/bin/env bash
# One "train N iterations, then film the policy" cycle (training and rendering do not fit on the 12 GB GPU together).
#   bash scripts/train_video_cycle.sh <task> <checkpoint.pt> <stop_iteration> <run_name> [num_envs]
# Trains from <checkpoint.pt> until model_<stop_iteration>.pt is saved, stops, records
# videos/pitch_iter_<stop_iteration>.mp4 and prints the checkpoint path for the next cycle.
set -u
cd "$(dirname "$0")/.."
TASK=$1
CKPT=$2
STOP=$3
RUN=$4
ENVS=${5:-4096}
PY=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
LOAD_RUN=$(basename "$(dirname "$CKPT")")
LOAD_CKPT=$(basename "$CKPT")
EXP_DIR=$(dirname "$(dirname "$CKPT")")

$PY -u scripts/train.py --task "$TASK" --headless --num_envs "$ENVS" --max_iterations 5000 --run_name "$RUN" \
    --resume True --load_run "$LOAD_RUN" --checkpoint "$LOAD_CKPT" >> logs_pitch_stand.txt 2>&1 &
PID=$!
NEW=""
while kill -0 $PID 2>/dev/null; do
    NEW=$(ls -t "$EXP_DIR"/*_"$RUN"/model_"$STOP".pt 2>/dev/null | head -1)
    [ -n "$NEW" ] && break
    sleep 30
done
sleep 20  # let the checkpoint finish writing
kill $PID 2>/dev/null
wait $PID 2>/dev/null
sleep 10
if [ -z "$NEW" ]; then
    echo "[cycle] training ended before model_$STOP.pt"
    grep -m1 -A8 -E "^Traceback|CUDA error" logs_pitch_stand.txt
    exit 1
fi
echo "[cycle] checkpoint $NEW"
grep -E "pitch/(release_rate|release_kmh|post_release_fail|release_time_err|lead_foot_err_m|arm_launch_err_m|chain_score|drive_err_rad|com_drop_mps|trunk_err_deg)" \
    logs_pitch_stand.txt | tail -10
$PY -u scripts/record_pitch_side.py --headless --pitches 3 --checkpoint "$NEW" --out "videos/pitch_iter_$STOP.mp4" 2>&1 \
    | grep -E "^\[side\]|Traceback"
$PY -u scripts/diag_pitch.py --headless --checkpoint "$NEW" --sections 1,4,5,6 2>&1     | sed -n '/^== diag_pitch/,$p' | grep -v "^\[INFO\]\|Warning\|^$" | tee "logs/diag_iter_$STOP.txt"
echo "[cycle] next: $NEW"
