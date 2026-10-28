#!/bin/bash
# Fall diag 3 (planning 14, 10-29): film-side after-done recovery in BatterSide (joint_smoke --after_done recover: from
# contact / the pitch passing, policy targets -> batting stance over 0.8 s, policy ignored). Seed 6, 64 envs, bias 0.055,
# into logs/falldiag3/; swing rows must equal logs/m2 s6. If it fails the bar (on the ground within 2 s <= 30% on hits and
# misses): the variant recover_low (0.4 s, knees +10 deg) in the same chain. Default after_done stays policy.
# Gate: Isaac 0 twice 60 s apart before each launch. Marker: logs/FALLDIAG3_END.txt via the EXIT trap (also on failure).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/FALLDIAG3_END.txt; O=logs/falldiag3
rm -f $M; MSG="aborted"
trap 'echo "FALLDIAG3_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
for AD in recover recover_low; do
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done $AD --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/$AD.csv > $O/$AD.txt 2>&1
  echo "$AD $(date +%H:%M) $(grep -E '^\[joint\] summary|Traceback' $O/$AD.txt | head -2 | tr '\n' ' ')"
  if grep -q Traceback $O/$AD.txt || [ ! -s $O/$AD.csv ]; then grep -A12 Traceback $O/$AD.txt | tail -8; MSG="crash $AD"; exit 1; fi
  R=$($P logs/recover_stats.py $O/$AD.csv); echo "$R"
  echo "$R" | grep -q "REC PASS" && { MSG="$AD PASS"; exit 0; }
done
MSG="both FAIL"
