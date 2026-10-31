#!/bin/bash
# QDF (planning 14, 11-01 01:1x): is the 800 Hz pelvis-yaw braking foot friction under the 800 Hz load profile? Measurement only
# (friction / offsets are not changed in the spec). Plain 12500, 4 joint clips, real heights, phase 2.6, j0, hard grip, --qdj
# (decomposition + vertical force; qd_trace now also carries hips + waist roll / pitch):
#  QDF_b4 / QDF_b8 = 400 / 800 Hz baselines with the hips traced (do the legs command the same yaw at both dt?)
#  QDF_mu05 / QDF_mu025 = 800 Hz, ground + default friction x0.5 / x0.25
#  QDF_off = 800 Hz, batter contact offset 0.005 m / rest offset 0.0 (PhysX contact generation margin; vs the asset default,
#    printed by PhysX only if set; not verified on CPU) -> logs/QDF_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/qdf; mkdir -p $O
M=logs/QDF_END.txt; rm -f $M; MSG="aborted"; OUT=""
trap 'echo "QDF_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
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
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  OUT="$OUT | $TG [$(grep -h '^\[diag\] \(ground\|batter contact\)' $O/diag_$TG.txt | tr '\n' ' ')]: $($P $SC/qdf_row.py logs/diag_bat_contact_$TG.json logs/trace_$TG.npz 2>&1 | tail -1)"
}
MSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0, measurement only"
row QDF_b4
row QDF_b8 --physics_dt 0.00125
row QDF_mu05 --physics_dt 0.00125 --ground_friction_scale 0.5
row QDF_mu025 --physics_dt 0.00125 --ground_friction_scale 0.25
row QDF_off --physics_dt 0.00125 --robot_contact_offset 0.005 0.0
MSG="done; $MSG"
