#!/bin/bash
# Fall diag (planning 14, 10-29): joint_smoke seeds 6 and 1, 64 envs, bias 0.055, M2 settings, with the read-only
# fall_pelvis_z / fall_tilt_deg columns (3935ea8) into logs/falldiag/. The world-z fall test on the field lowered 0.196 m.
# Gate: Isaac 0 twice 60 s apart before each launch. Marker: logs/FALLDIAG_END.txt via the EXIT trap (also on failure).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/FALLDIAG_END.txt; O=logs/falldiag
rm -f $M; MSG="aborted"
trap 'echo "FALLDIAG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
for s in 6 1; do
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $s --after_done policy --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/fd_s$s.csv > $O/fd_s$s.txt 2>&1
  echo "s$s $(date +%H:%M) $(grep -E '^\[joint\] summary|Traceback' $O/fd_s$s.txt | head -2 | tr '\n' ' ')"
  if grep -q Traceback $O/fd_s$s.txt || [ ! -s $O/fd_s$s.csv ]; then grep -A12 Traceback $O/fd_s$s.txt | tail -8; MSG="crash s$s"; exit 1; fi
done
MSG="done seeds 6 1"
