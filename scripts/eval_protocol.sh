#!/usr/bin/env bash
# Step-0 protocol for a checkpoint on one or more tasks: identical starts (64 envs) + two noisy draws (128 envs,
# start joint noise 0.02), then one summary line per task - replaces the ad-hoc logs/step0_*.sh scripts.
#   bash scripts/eval_protocol.sh <checkpoint.pt> <tag> <task> [<task> ...]
# Tasks run in parallel (2 Isaac processes at most on the GPU: give at most 2 tasks when nothing else runs).
# Raw output: logs/eval_<tag>_<i>_{step0,step0_noise,step0_noise2}.raw
set -u
cd "$(dirname "$0")/.."
CKPT=$1; TAG=$2; shift 2
PY=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
one() {
    local T=$1 B=$2
    $PY -u scripts/eval_stage.py --headless --task "$T" --envs 64 --checkpoint "$CKPT" --set start_joint_noise=0.0 start_noise_ramp_steps=0 > "${B}_step0.raw" 2>&1
    $PY -u scripts/eval_stage.py --headless --task "$T" --envs 128 --checkpoint "$CKPT" --set start_joint_noise=0.02 start_noise_ramp_steps=0 > "${B}_step0_noise.raw" 2>&1
    $PY -u scripts/eval_stage.py --headless --task "$T" --envs 128 --checkpoint "$CKPT" --set start_joint_noise=0.02 start_noise_ramp_steps=0 > "${B}_step0_noise2.raw" 2>&1
}
i=0
for T in "$@"; do
    i=$((i + 1))
    one "$T" "logs/eval_${TAG}_$i" &
    sleep 40  # stagger the Isaac start-ups
done
wait
i=0
for T in "$@"; do
    i=$((i + 1)); B="logs/eval_${TAG}_$i"
    g() { grep -o '^\[goal\] [0-9]*' "$1" | head -1 | cut -d' ' -f2; }
    s() { grep -o 'release speed median [0-9.]* m/s, heading' "$1" | head -1 | cut -d' ' -f4; }
    echo "[protocol] $T: identical $(g ${B}_step0.raw)/64 at $(s ${B}_step0.raw) m/s;" \
         "noisy $(g ${B}_step0_noise.raw), $(g ${B}_step0_noise2.raw) /128 at $(s ${B}_step0_noise.raw)/$(s ${B}_step0_noise2.raw) m/s"
    grep -m1 -E "^Traceback|CUDA error|out of memory" ${B}_step0*.raw
done
