#!/bin/bash
# ArmAim scripted row 5 (planning 14, 10-30): true-dy y offset + height offset toward D = +1 cm (12500 fit
# D = -0.1437 - 0.048 plate_y + 0.195 plate_z; gains logs/armaim_auth/scripted_gain_yz.json, y held), realised on z.
# Launch only on 14's branch call:  bash logs/run_armaim_row5.sh [window_s, default 0.30]  -> logs/ARMAIM_ROW5_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
GJ=logs/armaim_auth/scripted_gain_yz.json
R=assets/game/p3c_identical_t18_release.csv
W=${1:-0.30}
O=logs/armaim; M=logs/ARMAIM_ROW5_END.txt; T=armaim_script_yz
rm -f $M; MSG="aborted"
trap 'echo "ARMAIM_ROW5_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done
timeout 1200 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted $GJ --aim_scripted_true --aim_scripted_z \
  --aim_window $W --envs 1024 --seed 0 --tag $T --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
grep -q Traceback $O/diag_$T.txt && { grep -A14 Traceback $O/diag_$T.txt | tail -10; MSG="crash $T"; exit 1; }
[ -f logs/diag_bat_contact_$T.json ] || { MSG="$T: no json"; exit 1; }
SAT=$(grep "armaim scripted" $O/diag_$T.txt | tail -2 | tr '\n' ' ')
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 1000 --label "scripted y+z true window $W" | tr '\n' ' ')
MSG="$G | $SAT"
