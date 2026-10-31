#!/bin/bash
# QD (planning 14, 11-01 00:4x): desync overlay. Plain 12500, 4 joint clips, real heights, phase 2.6, j0, hard grip, P3c releases.
#  QD_W4 / QD_W8 = 400 / 800 Hz with --qd_trace (per 5 ms bin: |qd|, q, PD target of waist_yaw / l_elbow / l_wrist_*, |v_ss|;
#    per swing: reference phase at contact) + --link_all; overlay png + one-line summary (scratchpad trace_plot.py).
#  QD_W8v2 = 800 Hz --vel_limit_scale 2 + --link_all: does any peak |qd| pass 40 (does the scale reach PhysX)? -> logs/QD_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/qd; mkdir -p $O
M=logs/QD_END.txt; rm -f $M; MSG="aborted"; OUT=""
trap 'echo "QD_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, rest: extra flags
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --link_all \
    "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return 1
  fi
}
MSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0"
row QD_W4 --qd_trace && row QD_W8 --qd_trace --physics_dt 0.00125 && \
  OUT="$OUT | overlay $O/qd_overlay.png: $($P $SC/trace_plot.py logs/trace_QD_W4.npz logs/trace_QD_W8.npz $O/qd_overlay.png 2>&1 | tail -1)"
row QD_W8v2 --physics_dt 0.00125 --vel_limit_scale 2 && \
  OUT="$OUT | QD_W8v2 (800 Hz, vel x2): $($P $SC/nth_row.py logs/diag_bat_contact_QD_W8v2.json)"
MSG="done; $MSG"
