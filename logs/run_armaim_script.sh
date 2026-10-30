#!/bin/bash
# ArmAim scripted-aim falsifier (14's go 10-30 19:51): no training. Offset = gain(side) x |dy| from
# logs/armaim_auth/scripted_gain.json (scripts/armaim_scripted_gain.py), dy = predicted crossing y - nominal target y, in the
# window; the residual checkpoint (smoke) is loaded but ignored. Rows: sign +1 and the sign-flipped control -1.
# diag1024 seed 0 as the 12500 baseline -> scripts/armaim_gate.py --stage 300 -> logs/ARMAIM_SCRIPT_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
GJ=logs/armaim_auth/scripted_gain.json
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim; M=logs/ARMAIM_SCRIPT_END.txt
rm -f $M; MSG="aborted"
trap 'echo "ARMAIM_SCRIPT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
[ -f $RES ] || { MSG="no residual ckpt $RES"; exit 1; }
OUT=""
# v2 (20:35): the first +1 row saturated every pitch (mean |dy| 0.30 m from the percept): rows = true plate y +1 / -1, then the
# percept +1 once more with the |dy - true| diagnostic
for RW in "1 t" "-1 t" "1 p"; do
  set -- $RW; S=$1; K=$2
  T=armaim_script$K$([ $S -eq 1 ] && echo p || echo m); X=$([ $K = t ] && echo --aim_scripted_true)
  idle
  timeout 1200 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted $GJ --aim_scripted_sign $S $X \
    --envs 1024 --seed 0 --tag $T --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt && { grep -A14 Traceback $O/diag_$T.txt | tail -10; MSG="crash $T"; exit 1; }
  [ -f logs/diag_bat_contact_$T.json ] || { MSG="$T: no json"; exit 1; }
  SAT=$(grep "armaim scripted" $O/diag_$T.txt | tail -1)
  G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label "scripted sign $S dy $K" | tr '\n' ' ')
  OUT="$OUT || $G | $SAT"
done
MSG="$OUT"
