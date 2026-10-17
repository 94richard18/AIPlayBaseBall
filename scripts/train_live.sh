#!/usr/bin/env bash
# One continuous training process (no kill / resume between reports) + evaluation of each new checkpoint by a second
# process as it appears. Same output lines as train_cycles.sh ("[cycles] <it>: <k> up (<ckpt>)", "goal reached"), so
# train_curriculum.sh can use it (LIVE=1). Every resume cost a value-loss spike (~1100-1400 vs ~100) and a learning
# rate jump; 50-iteration cycles spent a large share of each round in that transient (fk / fm swings 28 -> 10, 16 -> 0).
#   bash scripts/train_live.sh <task> <checkpoint.pt> <max_iterations> <run_name> [every=50] [goal=60]
# Stops at the goal or after <max_iterations>; the last report also films the best checkpoint (key frames).
# The goal needs both evals: identical starts and start joint noise 0.02 rad (user decision 2026-10-17).
set -u
cd "$(dirname "$0")/.."
TASK=$1
CKPT=$2
MAXIT=$3
RUN=$4
EVERY=${5:-50}
GOAL=${6:-60}
PY=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
EXP_DIR=$(dirname "$(dirname "$CKPT")")
IT0=$(basename "$CKPT" .pt | sed 's/model_//')
[ $((IT0 % EVERY)) -ne 0 ] && { echo "[cycles] start $IT0 not on the checkpoint interval $EVERY - stopped"; exit 1; }
echo "$TASK $RUN $IT0 $((MAXIT / EVERY)) $EVERY" > logs/current_chain.txt
MARK=$(mktemp)
SAVE_EVERY=$EVERY $PY -u scripts/train.py --task "$TASK" --headless --num_envs 4096 --max_iterations 5000 --run_name "$RUN" \
    --resume True --load_run "$(basename "$(dirname "$CKPT")")" --checkpoint "$(basename "$CKPT")" \
    ${NOISE:+--max_noise_std $NOISE} ${SEED:+--seed $SEED} >> logs_pitch_stand.txt 2>&1 &
PID=$!
BEST=-1; BESTCK=$CKPT; IT=$IT0; RESULT=1
while [ $IT -lt $((IT0 + MAXIT)) ]; do
    IT=$((IT + EVERY))
    NEW=""
    while [ -z "$NEW" ]; do
        kill -0 $PID 2>/dev/null || { echo "[cycles] training ended before model_$IT.pt"; grep -m1 -A8 -E "^Traceback|CUDA error" logs_pitch_stand.txt; exit 1; }
        NEW=$(find "$EXP_DIR" -path "*_$RUN/model_$IT.pt" -newer "$MARK" 2>/dev/null | head -1)
        [ -z "$NEW" ] && sleep 30
    done
    sleep 10  # let the checkpoint finish writing
    # identical starts (start noise forced off: the Stage1dN tasks train with it on)
    $PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 64 --checkpoint "$NEW" --set start_joint_noise=0.0 2>&1 \
        | sed -n '/^== eval_stage/,$p' | grep -v "^\[INFO\]\|Warning\|^$" > "logs/eval_${RUN}_$IT.txt"
    # no video here: a 1-env render next to training stalled both (fp 3550: one training iteration 1586 s, the render
    # ~50 min); user 2026-10-17 "等70%結束再錄就好" - the best checkpoint is filmed when the task ends
    if [ -n "${LIVE_VIDEO:-}" ] && [ $(((IT - IT0) % 150)) -eq 0 ]; then
        $PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 1 --checkpoint "$NEW" --set start_joint_noise=0.0             --video "videos/${RUN}_iter_$IT.mp4" --video_episodes 1 2>&1 | grep -E "^\[video\]|CUDA error|out of memory" | head -3
    fi
    UP=$(grep -o '^\[goal\] [0-9]*' "logs/eval_${RUN}_$IT.txt" | head -1 | cut -d' ' -f2)
    echo "[cycles] $IT: ${UP:-?} up ($NEW)"
    grep -E "^  (falls:|at rest|thrown)" "logs/eval_${RUN}_$IT.txt" | sed 's/^/    /'
    grep -E "stage/(success_set|fall_rate_set|success_rsi|fall_rate_rsi)" logs_pitch_stand.txt | tail -4 | sed 's/^ */    train /'
    [ -n "$UP" ] && [ "$UP" -gt "$BEST" ] && { BEST=$UP; BESTCK=$NEW; }
    echo "[cycles] best so far $BEST ($BESTCK)"
    if [ -n "$UP" ] && [ "$UP" -ge "$GOAL" ]; then
        # user 2026-10-17 "兩次都要過": a crossing counts only if the eval with independent starts (start joint noise
        # 0.02 rad) passes too - the identical-start eval is 64 copies of one pitch up to solver noise
        $PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 64 --checkpoint "$NEW" --set start_joint_noise=0.02 2>&1             | sed -n '/^== eval_stage/,$p' | grep -v "^\[INFO\]\|Warning\|^$" > "logs/eval_${RUN}_${IT}_noise.txt"
        NUP=$(grep -o '^\[goal\] [0-9]*' "logs/eval_${RUN}_${IT}_noise.txt" | head -1 | cut -d' ' -f2)
        echo "[cycles] $IT with start noise 0.02: ${NUP:-?} up"
        grep -E "^  (falls:|thrown)" "logs/eval_${RUN}_${IT}_noise.txt" | sed 's/^/    /'
        if [ -n "$NUP" ] && [ "$NUP" -ge "$GOAL" ]; then echo "[cycles] goal reached"; BESTCK=$NEW; RESULT=0; break; fi
    fi
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 10
$PY -u scripts/eval_stage.py --headless --task "$TASK" --envs 64 --checkpoint "$BESTCK" --set start_joint_noise=0.0 --video "videos/${RUN}_best.mp4" 2>&1 \
    | grep -E "^\[video\]" | head -3
echo "[cycles] next: $BESTCK"
exit $RESULT
