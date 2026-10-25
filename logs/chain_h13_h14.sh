#!/bin/bash
# After Human13 (run_human13_resume.sh) ends: pick its best checkpoint by the Human13 gate (fall <= 37%, diag1024 >= 0.68,
# ONE OUTS >= 17.0%, fair >= 29 m >= 14.8%; lowest fall, within 5 points the higher win) - none passes -> the delivery
# 12500 - and start Human14 from it (decision session 10-25 20:1x). Falsifier also checks fall at 88-112 <= start + 10.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
until grep -q "^[0-9:]* done" logs/run_human13_resume.txt 2>/dev/null; do sleep 30; done
date +"%H:%M Human13 done"
SEL=$($P - <<'PY'
import re, glob, os
best = None
for line in open("logs/run_human13_resume.txt", encoding="utf-8", errors="ignore"):
    m = re.search(r"model_(\d+) diag1024 ([0-9.]+) \| fall (\d+)% \| fair >= 29 m ([0-9.]+)% \| .*batter ([0-9.]+)%", line)
    if not m: continue
    it, hr, fall, fair, win = int(m[1]), float(m[2]), int(m[3]), float(m[4]), float(m[5])
    if fall <= 37 and hr >= 0.68 and win >= 17.0 and fair >= 14.8:
        if best is None or fall < best[2] - 5 or (abs(fall - best[2]) <= 5 and win > best[4]): best = (it, hr, fall, fair, win)
if best:
    it = best[0]
    path = [p for p in glob.glob(f"logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched13*/model_{it}.pt")][0]
    print(os.path.basename(os.path.dirname(path)), f"model_{it}.pt", best[2])
else:
    print("2026-10-25_13-56-51_human_pitched9b model_12500.pt 57")
PY
)
read START_RUN START_CKPT START_FALL <<< "$SEL"
echo "Human14 start: $START_RUN/$START_CKPT (fall at 88-112: $START_FALL%)"
export START_RUN START_CKPT START_FALL
bash logs/run_human14.sh > logs/run_human14.txt 2>&1
cat logs/run_human14.txt
