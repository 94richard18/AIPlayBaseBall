#!/bin/bash
# Human30d mid-run falsifier (batter 15, 10-31 09:12): training fair_carry peaked 12575-12650 then fell; diag model_12650
# at jitter 0 and 0.03 (gate 300 + carry_cols) -> logs/H30D_12650_END.txt (EXIT trap). Eval only, one Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
D=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-31_08-16-00_human_pitched30d_1; I=12650
R=assets/game/p3c_identical_t18_release.csv; GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
O=logs/h30d; M=logs/H30D_12650_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H30D_12650_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
OUT=""
for j in 0 0.03; do
  T=h30d_${I}_j$j
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag $T --timing_noise $j \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt && fail $O/diag_$T.txt "diag $T"
  [ -f logs/diag_bat_contact_$T.json ] || { MSG="diag $T: no json"; exit 1; }
  G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label "j$j" | head -1)
  C=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_$T.json --label j$j)
  OUT="$OUT | $G | $C"
done
MSG="model_$I$OUT"
