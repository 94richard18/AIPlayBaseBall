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
# eval_stage.py output goes to a file first, then filtered: piping it straight into sed hung on the
# omni.telemetry.transmitter.exe that eval_stage leaves behind (it inherits the pipe, sed never sees EOF; hn 7500, 2026-10-24)
ev() {  # <out.txt> <eval_stage args...>: summary from "== eval_stage" on, without INFO / warnings
    OUT=$1; shift
    $PY -u scripts/eval_stage.py --headless "$@" > "$OUT.raw" 2>&1 < /dev/null
    sed -n '/^== eval_stage/,$p' "$OUT.raw" | grep -v "^\[INFO\]\|Warning\|^$" > "$OUT"; rm -f "$OUT.raw"
}
evv() {  # <eval_stage args...>: render only, print the [video] lines
    R=$(mktemp)
    $PY -u scripts/eval_stage.py --headless "$@" > "$R" 2>&1 < /dev/null
    grep -E "^\[video\]|CUDA error|out of memory" "$R" | head -3; rm -f "$R"
}
EXP_DIR=$(dirname "$(dirname "$CKPT")")
IT0=$(basename "$CKPT" .pt | sed 's/model_//')
[ $((IT0 % EVERY)) -ne 0 ] && { echo "[cycles] start $IT0 not on the checkpoint interval $EVERY - stopped"; exit 1; }
echo "$TASK $RUN $IT0 $((MAXIT / EVERY)) $EVERY" > logs/current_chain.txt
MARK=$(mktemp)
SAVE_EVERY=$EVERY $PY -u scripts/train.py --task "$TASK" --headless --num_envs ${NUM_ENVS:-4096} --max_iterations 5000 --run_name "$RUN" \
    --resume True --load_run "$(basename "$(dirname "$CKPT")")" --checkpoint "$(basename "$CKPT")" \
    ${NOISE:+--max_noise_std $NOISE} ${MIN_NOISE:+--min_noise_std $MIN_NOISE} ${SEED:+--seed $SEED} ${ARM_TEACHER:+--arm_teacher $ARM_TEACHER} ${WARMUP:+--critic_warmup $WARMUP} ${WARMUP_LR:+--critic_warmup_lr $WARMUP_LR} >> logs_pitch_stand.txt 2>&1 &
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
    # EVAL_EVERY (a multiple of <every>): evaluate only every EVAL_EVERY iterations - two evals (64 + 128 envs) per
    # 25-iteration checkpoint fell ~200 iterations behind the training (gj)
    # logs/eval_every_<run>.txt, if present, overrides it mid-run (a watcher sets it, e.g. curricula: sparse evals first)
    EE=$(cat logs/eval_every_$RUN.txt 2>/dev/null || echo ${EVAL_EVERY:-$EVERY})
    [ $(((IT - IT0) % EE)) -ne 0 ] && continue
    sleep 10  # let the checkpoint finish writing
    # EVAL_WAIT_FREE=1: the GPU is shared (2 Isaac processes at most, 2026-10-20 the batter runs on the other slot) - hold
    # this eval until only our trainer is running
    if [ -n "${EVAL_WAIT_FREE:-}" ]; then
        # count Isaac processes only (python.exe running scripts/*.py other than live_rules.py; 'scripts/' with the slash: an unrelated D:....venvScriptspython.exe streamlit app matched 'scripts' and held the F evals forever, 2026-10-27): a resident live_rules.py
        # watcher blocked every eval of hi from 7400 (2026-10-20)
        while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/' -and \$_.CommandLine -notmatch 'live_rules' }).Count" | tr -d '\r')" -ge 2 ]; do sleep 20; done
    fi
    # identical starts (start noise forced off: the Stage1dN tasks train with it on)
    ev "logs/eval_${RUN}_$IT.txt" --task "$TASK" --envs 64 --checkpoint "$NEW" --set start_joint_noise=0.0 start_noise_ramp_steps=0 ${EVAL_SET}
    # no video here: a 1-env render next to training stalled both (fp 3550: one training iteration 1586 s, the render
    # ~50 min); user 2026-10-17 "等70%結束再錄就好" - the best checkpoint is filmed when the task ends
    if [ -n "${LIVE_VIDEO:-}" ] && [ $(((IT - IT0) % 150)) -eq 0 ]; then
        evv --task "$TASK" --envs 1 --checkpoint "$NEW" --set start_joint_noise=0.0 start_noise_ramp_steps=0 ${EVAL_SET} --video "videos/${RUN}_iter_$IT.mp4" --video_episodes 1
    fi
    UP=$(grep -o '^\[goal\] [0-9]*' "logs/eval_${RUN}_$IT.txt" | head -1 | cut -d' ' -f2)
    echo "[cycles] $IT: ${UP:-?} up ($NEW)"
    grep -E "^  (falls:|at rest|thrown)" "logs/eval_${RUN}_$IT.txt" | sed 's/^/    /'
    grep -E "stage/(success_set|fall_rate_set|success_rsi|fall_rate_rsi)" logs_pitch_stand.txt | tail -4 | sed 's/^ */    train /'
    SCORE=$UP
    if [ -n "${NOISY_EVAL:-}" ]; then
        # NOISY_EVAL=1: the start-noise eval on every checkpoint, best tracked by it (user 2026-10-17: full RL until
        # 10-18 22:00, the bar is >= 30/64 at rest and thrown with start noise 0.02)
        # NOISY_ENVS=128: two draws' worth of independent starts in one launch, the score scaled back to /64 (a single
        # 64-run draw spreads ~+-4: fy r1 3875 drew 20, then 13)
        NE=${NOISY_ENVS:-64}
        ev "logs/eval_${RUN}_${IT}_noise.txt" --task "$TASK" --envs $NE --checkpoint "$NEW" --set start_joint_noise=0.02 start_noise_ramp_steps=0 ${EVAL_SET}
        SCORE=$(grep -o '^\[goal\] [0-9]*' "logs/eval_${RUN}_${IT}_noise.txt" | head -1 | cut -d' ' -f2)
        [ -n "$SCORE" ] && SCORE=$(((SCORE * 64 + NE / 2) / NE))
        echo "[cycles] $IT with start noise 0.02: ${SCORE:-?} up"
        grep -E "^  (falls:|thrown)" "logs/eval_${RUN}_${IT}_noise.txt" | sed 's/^/    /'
    fi
    if [ -f logs/eval_queue.sh ]; then  # side jobs in the eval slot (training runs on, still 2 Isaac processes)
        mv logs/eval_queue.sh logs/eval_queue_running.sh
        bash logs/eval_queue_running.sh >> logs/eval_queue_out.txt 2>&1 < /dev/null
    fi
    [ -n "$SCORE" ] && [ "$SCORE" -gt "$BEST" ] && { BEST=$SCORE; BESTCK=$NEW; }
    echo "[cycles] best so far $BEST ($BESTCK)"
    if [ -n "$UP" ] && [ "$UP" -ge "$GOAL" ]; then
        # user 2026-10-17 "兩次都要過": a crossing counts only if the eval with independent starts (start joint noise
        # 0.02 rad) passes too - the identical-start eval is 64 copies of one pitch up to solver noise
        ev "logs/eval_${RUN}_${IT}_goalcheck.txt" --task "$TASK" --envs 64 --checkpoint "$NEW" --set start_joint_noise=0.02 start_noise_ramp_steps=0 ${EVAL_SET}
        NUP=$(grep -o '^\[goal\] [0-9]*' "logs/eval_${RUN}_${IT}_goalcheck.txt" | head -1 | cut -d' ' -f2)
        echo "[cycles] $IT with start noise 0.02: ${NUP:-?} up"
        grep -E "^  (falls:|thrown)" "logs/eval_${RUN}_${IT}_goalcheck.txt" | sed 's/^/    /'
        if [ -n "$NUP" ] && [ "$NUP" -ge "$GOAL" ]; then echo "[cycles] goal reached"; BESTCK=$NEW; RESULT=0; break; fi
    fi
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 10
evv --task "$TASK" --envs 64 --checkpoint "$BESTCK" --set start_joint_noise=0.0 start_noise_ramp_steps=0 ${EVAL_SET} --video "videos/${RUN}_best.mp4"
echo "[cycles] next: $BESTCK"
exit $RESULT
