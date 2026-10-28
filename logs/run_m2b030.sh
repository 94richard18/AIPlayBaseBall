#!/bin/bash
# M2 at swing-start bias 0.030 (planning 14, 10-29): film-quality candidate. Same protocol as logs/m2 (two-pass timing
# logs/m2lite/timing.csv, after_done policy, seeds 1..10 x 64 envs) with joint_smoke --timing_bias 0.030 (default stays 0.055)
# into logs/m2_b030/. After each seed: pitch columns vs logs/m2 (logs/m2b_compare.py). End: m2_analysis 20000 PAs.
# Gate: Isaac 0 twice 60 s apart before each launch. Marker: logs/M2B030_END.txt via the EXIT trap (also on failure).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/M2B030_END.txt; O=logs/m2_b030
rm -f $M; MSG="aborted"
trap 'echo "M2B030_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
for s in $(seq 1 10); do
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $s --after_done policy --timing_csv logs/m2lite/timing.csv --timing_bias 0.030 --csv $O/m2_s$s.csv > $O/m2_s$s.txt 2>&1
  echo "s$s $(date +%H:%M) $(grep -E '^\[joint\] summary|Traceback' $O/m2_s$s.txt | head -2 | tr '\n' ' ')"
  grep -o "bias [0-9.]*" $O/m2_s$s.txt | head -1
  if grep -q Traceback $O/m2_s$s.txt || [ ! -s $O/m2_s$s.csv ]; then
    grep -A12 Traceback $O/m2_s$s.txt | tail -8; MSG="crash s$s"; exit 1
  fi
  $P logs/m2b_compare.py $O | head -1
done
for i in $(seq 60); do [ "$(isaac)" -eq 0 ] && break; sleep 10; done
$P scripts/m2_analysis.py --rows "$O/m2_s*.csv" --at_bats 20000 2>&1 | grep '^\[m2\]'
$P logs/m2b_compare.py $O
MSG="done 10 seeds"
