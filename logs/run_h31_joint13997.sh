#!/bin/bash
# 14 (11-01 11:4x): joint s6 of H31 model_13997 beside the segment-4 trainer (Isaac 1 -> 2 allowed, no film, RAM checked);
# same flags as the film4 hook in run_h31.sh -> logs/h31/joint_s6_13997.csv + logs/H31_JOINT_13997_END.txt. Since FILM_CKPT
# is 13997 the line is also written to logs/H31_JOINT_END.txt so the film4 hook does not rerun it (delete it if FILM_CKPT changes).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/h31; M=logs/H31_JOINT_13997_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H31_JOINT_13997_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done
JC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-01_09-54-34_human_pitched_h31_3/model_13997.pt; JI=13997
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov4/recover_ckpt.txt)
GY=logs/armaim_auth/scripted_gain.json
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $JC --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --phase_horizon 2.6 --csv $O/joint_s6_$JI.csv \
  > $O/joint_s6_$JI.txt 2>&1
if grep -q Traceback $O/joint_s6_$JI.txt || [ ! -f $O/joint_s6_$JI.csv ]; then
  MSG="CRASH $JC: $(grep -A14 Traceback $O/joint_s6_$JI.txt | tail -1)"
else
  MSG="BC=$JC ROWS=$O/joint_s6_$JI.csv (recover $(basename $(dirname $RC))): $($P $SC/h31_joint_row.py $O/joint_s6_$JI.csv)"
  [ "$(tr -d '\r\n' < $O/FILM_CKPT)" = "$JC" ] && echo "H31_JOINT_END $(date +%H:%M) $MSG" > logs/H31_JOINT_END.txt
fi
