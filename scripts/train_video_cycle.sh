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
if [ "$CKPT" = none ]; then  # from scratch
    EXP_DIR=logs/rsl_rl/aib1_pitch
    case "$TASK" in *Stage*) EXP_DIR=logs/rsl_rl/aib1_stage;; esac
    RESUME=()
else
    EXP_DIR=$(dirname "$(dirname "$CKPT")")
    RESUME=(--resume True --load_run "$(basename "$(dirname "$CKPT")")" --checkpoint "$(basename "$CKPT")")
fi

$PY -u scripts/train.py --task "$TASK" --headless --num_envs "$ENVS" --max_iterations 5000 --run_name "$RUN" \
    "${RESUME[@]}" ${RESET_NORM:+--reset_normalizer} ${ZERO_HEAD:+--zero_actor_head} ${WARMUP:+--critic_warmup $WARMUP} ${NOISE:+--max_noise_std $NOISE} >> logs_pitch_stand.txt 2>&1 &
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
grep -E "pitch/(release_rate|release_com_vx|release_kmh|post_release_fail|release_time_err|lead_foot_err_m|arm_launch_err_m|chain_score|drive_err_rad|com_drop_mps|trunk_err_deg)" \
    logs_pitch_stand.txt | tail -10
if [[ "$TASK" == *Stage* ]]; then  # stage 1 step by step: short report, key frames to look at, video
    grep -E "stage/" logs_pitch_stand.txt | tail -6
    $PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 64 --checkpoint "$NEW" --video "videos/${RUN}_iter_$STOP.mp4" 2>&1         | sed -n '/^== eval_stage/,$p' | grep -v "^\[INFO\]\|Warning\|^$" | tee "logs/eval_${RUN}_$STOP.txt"
    echo "[cycle] with training action noise:"
    $PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 64 --checkpoint "$NEW" --stochastic 2>&1         | grep -E "falls:|at rest" | sed 's/^/  noisy /'
    echo "[cycle] next: $NEW"
    exit 0
fi
# report and video in one simulator launch (one Isaac Sim start-up instead of two)
CFG=stand_first; case "$TASK" in *Stand2*) CFG=stand2;; *Stand3*) CFG=stand3;; *Stand4*) CFG=stand4;; *Stand5*) CFG=stand5;; *Stand6*) CFG=stand6;; *Stand7*) CFG=stand7;; *Stand8*) CFG=stand8;; *Stand9*) CFG=stand9;; esac
$PY -u scripts/diag_pitch.py --headless --envs 64 --cfg "$CFG" --checkpoint "$NEW" --sections 1,2,4,5,6,8,9 --video "videos/pitch_iter_$STOP.mp4" 2>&1 \
    | sed -n '/^== diag_pitch/,$p' | grep -v "^\[INFO\]\|Warning\|^$" | tee "logs/diag_iter_$STOP.txt"
# the same pitches with the training action noise: a gap to the noise-free run = the mean action is not trained
echo "[cycle] with training action noise:"
$PY -u scripts/diag_pitch.py --headless --envs 64 --cfg "$CFG" --checkpoint "$NEW" --sections 1 --stochastic 2>&1     | sed -n '/^\[1\] OUTCOME/,$p' | grep -E "release:|falls:|at rest|ball progress" | sed 's/^/  noisy /'
echo "[cycle] next: $NEW"
