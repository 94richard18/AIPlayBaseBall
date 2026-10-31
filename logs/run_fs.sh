#!/bin/bash
# FS (planning 14, 11-01 02:3x): with the same hip / waist motion at both dt, a lower pelvis yaw rate must come from the feet
# yawing on the ground (torsional slip), not from the joints. Measurement only. Gate: logs/FILM2R_END.txt + Isaac 0 + idle().
# Plain 12500, 4 joint clips, real heights, phase 2.6, j0, hard grip, --qdj (now also both feet's yaw rate, the leg-joint term
# from the foot angular-z Jacobian row and the closure residual, r_knee traced):
#  FS4 / FS8 = 400 / 800 Hz
#  if 800 Hz |foot yaw| exceeds 400 Hz by > 1 rad/s in any -88 / -57 / -27 / +3 ms bin (either foot):
#  FS8_tp / FS4_tp = 800 / 400 Hz, batter torsional patch radius 0.02 m (min 0.02; stage readback printed) -> logs/FS_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/fs; mkdir -p $O
M=logs/FS_END.txt; rm -f $M; MSG="aborted"; OUT=""
trap 'echo "FS_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, rest: extra flags
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --qdj \
    "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return 1
  fi
  OUT="$OUT | $TG [$(grep -h '^\[diag\] \(batter torsional\|stage readback\|readback root\)' $O/diag_$TG.txt | tr '\n' ' ')]: $($P $SC/fs_row.py logs/diag_bat_contact_$TG.json logs/trace_$TG.npz 2>&1 | tail -1)"
}
MSG="waiting for FILM2R_END"
until [ -f logs/FILM2R_END.txt ]; do sleep 60; done
MSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0, measurement only"
row FS4 && row FS8 --physics_dt 0.00125 || exit
G=$($P $SC/fs_gate.py logs/trace_FS4.npz logs/trace_FS8.npz); GS=$?
OUT="$OUT | gate: $G"
if [ $GS -eq 0 ]; then
  row FS8_tp --physics_dt 0.00125 --torsion 0.02
  row FS4_tp --torsion 0.02
else
  OUT="$OUT | gate not met: torsional stage skipped"
fi
MSG="done; $MSG"
