#!/bin/bash
# Contact-offset measurement (planning 14, user 10-29 "好先做接觸偏移量測"): frozen joint M2 (h9b 12500 @ 1.5x, P3c,
# bias 0.055, falldiag2 settings) seeds 6 and 1, 64 envs, joint_smoke --contact_dump (read-only taps on the impact
# model) -> scripts/contact_offset.py (CPU replay of resolve_impact) -> logs/contact_offset/. Waits for 20's
# logs/M3CLEAN2_END.txt, then Isaac 0 twice 60 s apart before every launch. Marker logs/CONTACT_END.txt (EXIT trap).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/CONTACT_END.txt; O=logs/contact_offset
rm -f $M; MSG="aborted"
trap 'echo "CONTACT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
W=0
until [ -f logs/M3CLEAN2_END.txt ]; do sleep 60; W=$((W + 1)); [ $W -ge 480 ] && { MSG="no logs/M3CLEAN2_END.txt within 8 h"; exit 1; }; done
echo "$(date +%H:%M) gate: $(cat logs/M3CLEAN2_END.txt)"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
for s in 6 1; do
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $s --after_done policy --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/s$s.csv --contact_dump $O/s$s.npz > $O/s$s.txt 2>&1
  echo "s$s $(date +%H:%M) $(grep -E '^\[joint\] summary|^\[joint\] contact dump|Traceback' $O/s$s.txt | head -3 | tr '\n' ' ')"
  if grep -q Traceback $O/s$s.txt || [ ! -s $O/s$s.npz ]; then grep -A12 Traceback $O/s$s.txt | tail -8; MSG="crash s$s"; exit 1; fi
done
$P scripts/contact_offset.py --runs $O/s6 $O/s1 --out $O > $O/analysis.txt 2>&1 || { tail -8 $O/analysis.txt; MSG="analysis crash"; exit 1; }
cat $O/analysis.txt
MSG="done ($O/summary.txt)"
