#!/bin/bash
# TC (planning 14, 11-01 01:5x): is the lead hip-yaw limit stop THE dt-dependent element of the 800 Hz bat-speed loss?
# Measurement only (the +-1 rad hip yaw range is the user's design value). Gate: logs/FILM2_END.txt + Isaac 0 + idle().
# Plain 12500, 4 joint clips, real heights, phase 2.6, j0, hard grip, --qdj + --step_trace on every row:
#  HL8_13 / HL4_13 = 800 / 400 Hz, l_hip_yaw + r_hip_yaw limits +-1.3 rad (PhysX + PD-target clamp)
#  TC4_m10 = 400 Hz, PD targets clipped to [lo + 0.10, hi - 0.10] (the hip cannot be driven into its stop)
#  ST4 / ST8 = 400 / 800 Hz baselines for the per-step l_hip_yaw png around the limit hit (-40..+5 ms)
#  TC8_m10 = 800 Hz, same clip (last, cheap extra) -> logs/TC_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/tc; mkdir -p $O
M=logs/TC_END.txt; rm -f $M; MSG="aborted"; OUT=""
trap 'echo "TC_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, rest: extra flags
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --qdj \
    --step_trace "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return 1
  fi
  OUT="$OUT | $TG [$(grep -h '^\[diag\] \(hip yaw\|PD targets\|step_trace\)' $O/diag_$TG.txt | tr '\n' ' ')]: $($P $SC/tc_row.py logs/diag_bat_contact_$TG.json logs/trace_$TG.npz 2>&1 | tail -1)"
}
MSG="waiting for FILM2_END"
until [ -f logs/FILM2_END.txt ]; do sleep 60; done
MSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0, measurement only"
row HL8_13 --physics_dt 0.00125 --hip_yaw_limit 1.3
row HL4_13 --hip_yaw_limit 1.3
row TC4_m10 --target_clip_margin 0.10
row ST4 && row ST8 --physics_dt 0.00125 && \
  OUT="$OUT | step png $O/hip_step.png: $($P $SC/step_plot.py logs/steptrace_ST4.npz logs/steptrace_ST8.npz $O/hip_step.png 2>&1 | tail -1)"
row TC8_m10 --physics_dt 0.00125 --target_clip_margin 0.10
MSG="done; $MSG"
