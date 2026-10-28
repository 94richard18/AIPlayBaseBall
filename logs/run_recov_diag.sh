#!/bin/bash
# Human25-R stage 2 (planning 14, 10-29): after RECOV_TRAIN_END, final = the better of model_299 and the +1000 checkpoint
# on the s6 bank (REVAL ground); joint_smoke seed 6 (falldiag2 settings) with after_done policy (rerun: follow-through
# stats baseline) and recover_rl -> logs/recov_diag_stats.py. Marker logs/RECOV_DIAG_END.txt (EXIT trap). One Isaac.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/RECOV_DIAG_END.txt; O=logs/recov_diag
rm -f $M; MSG="aborted"
trap 'echo "RECOV_DIAG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
until [ -f logs/RECOV_TRAIN_END.txt ]; do sleep 30; done
cat logs/RECOV_TRAIN_END.txt
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r | head -1); RUN2=$(ls -td logs/rsl_rl/aib1_recover/*_h25r_b 2>/dev/null | head -1)
best=""; bg=9
for e in logs/recov/eval_*.txt; do
  [ "$e" = logs/recov/eval_zero.txt ] && continue
  it=$(basename $e .txt | sed 's/eval_//'); g=$(grep '^REVAL' $e | awk '{print $4}'); [ -n "$g" ] || continue
  if $P -c "import sys; sys.exit(0 if float('$g') < float('$bg') else 1)"; then bg=$g; best=$it; fi
done
[ -n "$best" ] || { MSG="no evaluated checkpoint"; exit 1; }
RC=$RUN/model_$best.pt; [ -f $RC ] || RC=$RUN2/model_$best.pt
echo "final recovery checkpoint $RC (s6 bank on the ground $bg)"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
for AD in policy recover_rl; do
  X=""; [ $AD = recover_rl ] && X="--recover_ckpt $RC"
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done $AD $X --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/$AD.csv > $O/$AD.txt 2>&1
  echo "$AD $(date +%H:%M) $(grep -E '^\[joint\] summary|Traceback' $O/$AD.txt | head -2 | tr '\n' ' ')"
  if grep -q Traceback $O/$AD.txt || [ ! -s $O/$AD.csv ]; then grep -A12 Traceback $O/$AD.txt | tail -8; MSG="crash $AD"; exit 1; fi
done
$P logs/recov_diag_stats.py $O/recover_rl.csv $O/policy.csv
MSG="done ($RC)"
