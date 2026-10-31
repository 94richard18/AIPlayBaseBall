#!/bin/bash
# DT (planning 14, 11-01 03:1x): which physics dt is converged? Control stays 100 Hz via decimation. Measurement only.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
# Plain 12500, 4 joint clips, real heights, phase 2.6, j0, hard grip, --qdj: DT200 (1/200, dec 2), DT600 (1/600, dec 6),
# DT1600 (1/1600, dec 16), DT3200 (1/3200, dec 32) only if DT1600 took < 15 min -> logs/DT_END.txt
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/dtc; mkdir -p $O
M=logs/DT_END.txt; rm -f $M; MSG="aborted"; OUT=""
trap 'echo "DT_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
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
  OUT="$OUT | $TG [$(grep -h '^\[diag\] physics dt' $O/diag_$TG.txt | tr '\n' ' ')]: $($P $SC/qdf_row.py logs/diag_bat_contact_$TG.json logs/trace_$TG.npz 2>&1 | tail -1)"
}
MSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0, measurement only"
trow() { local s=$(date +%s); row "$@"; OUT="$OUT ($(( ($(date +%s) - s) / 60 )) min)"; LAST=$(( $(date +%s) - s )); }
trow DT200 --physics_dt 0.005
trow DT600 --physics_dt 0.0016666667
trow DT1600 --physics_dt 0.000625
if [ $LAST -lt 900 ]; then trow DT3200 --physics_dt 0.0003125; else OUT="$OUT | DT3200 skipped (DT1600 ${LAST} s)"; fi
MSG="done; $MSG"
