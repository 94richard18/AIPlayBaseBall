#!/bin/bash
# Timing vs contact station (planning 14, user 10-30 "新線": measurement only). Delivery 12500 @ 1.5x, diag1024 seed 0
# (the Human26 baseline command: P3c t18, jitter 0, aim 0, --contact_log) at swing-start bias 0.030 / 0.080 / 0.105
# (0.055 = logs/diag_bat_contact_h26_base.json) -> scripts/timing_contact.py -> logs/timing_contact/.
# Isaac 0 twice 60 s apart before every launch; marker logs/TIMING_END.txt (EXIT trap).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/TIMING_END.txt; O=logs/timing_contact
rm -f $M; MSG="aborted"
trap 'echo "TIMING_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
mkdir -p $O
A="0.055:logs/diag_bat_contact_h26_base.json"
for b in 0.030 0.080 0.105; do
  idle
  timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C0 --envs 1024 --seed 0 --tag tc_b$b --timing_noise 0 --timing_bias $b --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_b$b.txt 2>&1
  if grep -q Traceback $O/diag_b$b.txt || [ ! -s logs/diag_bat_contact_tc_b$b.json ]; then grep -A12 Traceback $O/diag_b$b.txt | tail -8; MSG="crash bias $b"; exit 1; fi
  echo "$(date +%H:%M) bias $b: $(grep '^\[bat\] slow in zone' $O/diag_b$b.txt | cut -c1-80)"
  A="$A $b:logs/diag_bat_contact_tc_b$b.json"
done
$P scripts/timing_contact.py --diag $A --out $O > $O/analysis.txt 2>&1 || { tail -8 $O/analysis.txt; MSG="analysis crash"; exit 1; }
cat $O/analysis.txt
MSG="done ($O/summary.txt)"
