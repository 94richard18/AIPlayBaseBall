#!/bin/bash
# Human25-R2 (c) (planning 14, 10-29 09:3x): after 20's PA1 re-film with model_1297 (a new logs/M3PA*END*.txt), eval the
# earlier R2 checkpoints (1000 / 1100 / 1200) on the s6 bank; the best on the ground with torso tilt q90 < 15 deg, if it
# beats model_1297 (ground 0.018), gets the joint seed 6 recover_rl rerun + logs/recov2_diag_stats.py.
# Marker logs/RECOV2C_END.txt (EXIT trap). One Isaac at a time: Isaac 0 twice 60 s apart before every launch.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/RECOV2C_END.txt; O=logs/recov2c; RUN=logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2
rm -f $M; MSG="aborted"
trap 'echo "RECOV2C_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
touch $O.start 2>/dev/null || { mkdir -p $O; touch $O.start; }
mkdir -p $O
W=0
until [ -n "$(find logs -maxdepth 1 -name 'M3PA*END*.txt' -newer $O.start 2>/dev/null)" ]; do
  sleep 60; W=$((W + 1)); [ $W -ge 240 ] && { MSG="no film marker from 20 within 4 h"; exit 1; }
done
echo "$(date +%H:%M) film marker: $(cat $(find logs -maxdepth 1 -name 'M3PA*END*.txt' -newer $O.start | head -1))"
best=1297; bg=0.018
for c in 1000 1100 1200; do
  idle
  timeout 900 $P -u scripts/recover_eval.py --headless --checkpoint $RUN/model_$c.pt --tag m$c > $O/eval_$c.txt 2>&1
  grep -E '^REVAL|^RTORSO' $O/eval_$c.txt
  G=$(grep '^REVAL' $O/eval_$c.txt | awk '{print $4}'); Q=$(grep '^RTORSO' $O/eval_$c.txt | awk '{print $6}')
  [ -n "$G" ] && [ -n "$Q" ] || fail $O/eval_$c.txt eval_$c
  if $P -c "import sys; sys.exit(0 if float('$G') < float('$bg') and float('$Q') < 15 else 1)"; then best=$c; bg=$G; fi
done
if [ $best = 1297 ]; then MSG="no earlier R2 checkpoint beats model_1297 (ground 0.018) with torso q90 < 15 - keep model_1297"; exit 0; fi
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done recover_rl --recover_ckpt $RUN/model_$best.pt --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/recover_rl_$best.csv > $O/recover_rl_$best.txt 2>&1
if grep -q Traceback $O/recover_rl_$best.txt || [ ! -s $O/recover_rl_$best.csv ]; then fail $O/recover_rl_$best.txt joint_$best; fi
$P logs/recov2_diag_stats.py $O/recover_rl_$best.csv | tee $O/stats_$best.txt
MSG="best model_$best (s6 ground $bg): $(grep '^RD2 RESULT' $O/stats_$best.txt | cut -d' ' -f3-) | model_1297: frozen 64/64, on the ground 2/64, torso +1.5 s med 6.8 q90 11.8, ss above 1/64"
