#!/bin/bash
# Resident stop-rule watcher that is not a python process (train_live.sh's EVAL_WAIT_FREE counts python.exe, so a
# resident live_rules.py held every eval of hi from 7400 on, 2026-10-20). Calls live_rules.py --once whenever a new
# noisy eval file appears; exits with its verdict line, or when the trainer is gone.
#   bash scripts/live_rules_loop.sh <run> <from_it> [extra live_rules args]
RUN=$1; FROM=$2; shift 2
PY=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cd "$(dirname "$0")/.."
LAST=""
while true; do
    CUR=$(ls logs/eval_${RUN}_*_noise.txt 2>/dev/null | xargs -r grep -l '^\[goal\]' 2>/dev/null | sort | tr '\n' ' ')
    if [ "$CUR" != "$LAST" ]; then
        LAST=$CUR
        OUT=$($PY scripts/live_rules.py --run "$RUN" --from_it "$FROM" --once "$@"); RC=$?
        echo "$OUT" | tail -1
        [ $RC -eq 0 ] && exit 0
    fi
    tasklist 2>/dev/null | grep -qi python.exe || { echo "[rules] no python process left - trainer gone"; exit 0; }
    powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'train\.py' -and \$_.CommandLine -match '$RUN' }) { exit 0 } else { exit 1 }" \
        || { $PY scripts/live_rules.py --run "$RUN" --from_it "$FROM" --once --no_kill "$@" | tail -1; echo "[rules] trainer ended"; exit 0; }
    sleep 60
done
