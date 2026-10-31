#!/bin/bash
# DTNUM (planning 14, 10-31 20:18): zero-training numerics rows for the 800 vs 400 Hz swing-speed gap. After ADOPT_FILM_END
# (newer than this chain's start) + idle; plain 12500 (no saim), the 4 joint clips, real heights, phase 2.6, j0, P3c releases:
#  V1 = 800 Hz --solver_iters 4 1   (the 400 Hz solver work per second: closed-chain grip / iteration story)
#  V2 = 800 Hz --vel_limit_scale 2  (joint velocity limits binding)
#  V3 = 400 Hz --solver_iters 16 4  (reverse test: if ss drops toward 16, the iteration story is confirmed from the other side)
# vs P8_12500 (800 Hz, ss 16.6) / P4_12500 (400 Hz, ss 21.3). Each row: in-zone hit, exit / launch med, fair, ss_max (hit / miss),
# dz, torque saturation of shoulders / wrists / waist (--sat_log top 10) -> logs/DTNUM_END.txt (EXIT trap). One Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/dtnum; mkdir -p $O
M=logs/DTNUM_END.txt; rm -f $M; MSG="aborted"; OUT=""
touch $O/chain.stamp
trap 'echo "DTNUM_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, rest: extra flags
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --sat_log \
    "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  OUT="$OUT | $TG: $($P $SC/dtnum_row.py logs/diag_bat_contact_$TG.json)"
}
until [ -f logs/ADOPT_FILM_END.txt ] && [ logs/ADOPT_FILM_END.txt -nt $O/chain.stamp ]; do sleep 60; done
MSG="plain 12500, 4 joint clips, real heights, phase 2.6, j0 (P8 800 Hz ss 16.6 / P4 400 Hz ss 21.3)"
row DTV1_800_it41 --physics_dt 0.00125 --solver_iters 4 1
row DTV2_800_vel2 --physics_dt 0.00125 --vel_limit_scale 2
row DTV3_400_it164 --solver_iters 16 4
