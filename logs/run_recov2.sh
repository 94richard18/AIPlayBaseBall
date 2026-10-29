#!/bin/bash
# Human25-R2 (planning 14, 10-29 08:5x): model_998 finishes hunched (torso ~74 deg, bat up) and speeds the bat up after
# the hand-off on a few rows. Resume model_998 +300 on AIB-Recover-R2-v0 (same env + bank; + torso upright, posture
# toward the stance from 0.5 s, arm joint-speed penalty, bat speed-rise penalty for 0.3 s).
# 1. model_998 baseline on the s6 bank with the new eval columns; 2. train; 3. eval the last checkpoint -> falsifiers:
#    on the ground < 5%, torso tilt at 2 s med < 30 / q90 < 45 deg, bat speed above its first-step value in 0.2 s none
#    -> logs/RECOV2_TRAIN_END.txt; 4. (if on the ground passes) joint seed 6 recover_rl with it -> logs/recov2_diag_stats.py
#    -> logs/RECOV2_DIAG_END.txt. One Isaac at a time: Isaac 0 twice 60 s apart before every launch; markers by EXIT trap.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
MT=logs/RECOV2_TRAIN_END.txt; MD=logs/RECOV2_DIAG_END.txt; O=logs/recov2
rm -f $MT $MD; STAGE=train; MSG="aborted"
trap 'if [ "$STAGE" = train ]; then echo "RECOV2_TRAIN_END $(date +%H:%M) $MSG" > $MT; cat $MT; echo "RECOV2_DIAG_END $(date +%H:%M) not started ($MSG)" > $MD; else echo "RECOV2_DIAG_END $(date +%H:%M) $MSG" > $MD; fi; cat $MD' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
R0=logs/rsl_rl/aib1_recover/2026-10-29_06-33-05_h25r_b
mkdir -p $O
idle
timeout 900 $P -u scripts/recover_eval.py --headless --checkpoint $R0/model_998.pt --tag m998 > $O/eval_998.txt 2>&1
grep -E '^\[reval\]|^REVAL|^RTORSO' $O/eval_998.txt
grep -q '^RTORSO' $O/eval_998.txt || fail $O/eval_998.txt eval_998
idle
timeout 3600 $P -u scripts/train.py --task AIB-Recover-R2-v0 --headless --num_envs 4096 --max_iterations 300 --run_name h25r2 --resume True --load_run $(basename $R0) --checkpoint model_998.pt > $O/train.txt 2>&1
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r2 | head -1)
echo "$(date +%H:%M) run $RUN: $(ls $RUN | grep -c model_) checkpoints"
grep -q Traceback $O/train.txt && fail $O/train.txt train
CL=$(ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
idle
timeout 900 $P -u scripts/recover_eval.py --headless --checkpoint $RUN/model_$CL.pt --tag m$CL > $O/eval_$CL.txt 2>&1
grep -E '^\[reval\]|^REVAL|^RTORSO' $O/eval_$CL.txt
G=$(grep '^REVAL' $O/eval_$CL.txt | awk '{print $4}'); T=$(grep '^RTORSO' $O/eval_$CL.txt)
[ -n "$G" ] && [ -n "$T" ] || fail $O/eval_$CL.txt eval_$CL
V=$($P -c "
t='$T'.split(); m, q9, r = float(t[3]), float(t[5]), int(t[7])
g = float('$G')
print(f\"ground {g:.3f} {'PASS' if g < 0.05 else 'FAIL'}, torso med {m} q90 {q9} {'PASS' if m < 30 and q9 < 45 else 'FAIL'}, bat rise rows {r} {'PASS' if r == 0 else 'FAIL'}\")")
MSG="model_$CL ($RUN): $V | model_998 $(grep '^RTORSO' $O/eval_998.txt | cut -d' ' -f3-)"
echo "RECOV2_TRAIN_END $(date +%H:%M) $MSG" > $MT; cat $MT
STAGE=diag; MSG="aborted in stage 2"
if ! $P -c "import sys; sys.exit(0 if float('$G') < 0.05 else 1)"; then MSG="skipped: on the ground $G >= 0.05"; exit 0; fi
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done recover_rl --recover_ckpt $RUN/model_$CL.pt --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/recover_rl.csv > $O/recover_rl.txt 2>&1
echo "recover_rl $(date +%H:%M) $(grep -E '^\[joint\] summary|Traceback' $O/recover_rl.txt | head -2 | tr '\n' ' ')"
if grep -q Traceback $O/recover_rl.txt || [ ! -s $O/recover_rl.csv ]; then fail $O/recover_rl.txt recover_rl; fi
$P logs/recov2_diag_stats.py $O/recover_rl.csv | tee $O/stats.txt
MSG="done ($RUN/model_$CL.pt): $(grep '^RD2 RESULT' $O/stats.txt | cut -d' ' -f3-)"
