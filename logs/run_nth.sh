#!/bin/bash
# NTH (planning 14, 10-31 20:49): does the 400-vs-800 Hz ss_max gap disappear without the closed-chain grip? After H30C800_END
# (newer than this chain's start) + idle; plain 12500 (no saim), the 4 joint clips, real heights, phase 2.6, j0, P3c releases.
#  W4 / W8 = two hands at 400 / 800 Hz (the P4 / P8 shape again, now with per-joint peak |qd| via --link_all)
#  R4 / R8 = top hand released (--release_top_hand: diag-made USD copy, top_hand_grip disabled) at 400 / 800 Hz
# Gap test: (W4 - W8) vs (R4 - R8) in ss_max; absolute one-handed speeds will be lower. -> logs/NTH_END.txt (EXIT trap).
# TAUM (14, 21:08, pre-approved): if the gap survives without the top hand (W4 - W8 >= 2 and R4 - R8 >= half of it), two hands at
# 400 / 800 Hz again with --tau_log (PhysX projected joint forces / effort limit: peak, mean, share >= 0.98, vs the Isaac Lab
# estimate over the same window) -> logs/TAUM_END.txt; else TAUM_END "skipped".
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/nth; mkdir -p $O
M=logs/NTH_END.txt; rm -f $M logs/TAUM_END.txt; MSG="aborted"; OUT=""; TMSG="aborted (NTH did not finish)"; TOUT=""
touch $O/chain.stamp
trap 'echo "NTH_END $(date +%H:%M) $MSG$OUT" > $M; cat $M; echo "TAUM_END $(date +%H:%M) $TMSG$TOUT" > logs/TAUM_END.txt; cat logs/TAUM_END.txt' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, rest: extra flags
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --sat_log \
    --link_all "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  OUT="$OUT | $TG: $($P $SC/nth_row.py logs/diag_bat_contact_$TG.json)"
}
ss() { $P -c "import json, numpy as np; P = {k: np.array(v, dtype=float) for k, v in json.load(open('logs/diag_bat_contact_$1.json'))['per_pitch'].items()}; z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.5) & (P['plate_z'] <= 1.1); print(round(float(np.nanmedian(P['ss_speed_max'][z])), 2))" 2>/dev/null || echo nan; }
trow() {  # $1 tag, rest: extra flags (TAUM)
  local TG=$1; shift
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0     --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --tau_log     "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    TOUT="$TOUT | $TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  TOUT="$TOUT | $TG: $($P $SC/taum_row.py logs/diag_bat_contact_$TG.json)"
}
until [ -f logs/H30C800_END.txt ] && [ logs/H30C800_END.txt -nt $O/chain.stamp ]; do sleep 60; done
MSG="plain 12500, 4 joint clips, real heights, phase 2.6, j0; peak |qd| rad/s / sat % last 0.1 s, in-zone swings"
row NTH_W4
row NTH_W8 --physics_dt 0.00125
row NTH_R4 --release_top_hand
row NTH_R8 --release_top_hand --physics_dt 0.00125
MSG="done; $MSG"
GW=$(ss NTH_W4) G8=$(ss NTH_W8) GR=$(ss NTH_R4) GR8=$(ss NTH_R8)
SURV=$($P -c "w = $GW - $G8; r = $GR - $GR8; print(int(w >= 2 and r >= 0.5 * w))" 2>/dev/null || echo 0)
OUT="$OUT | gap W4-W8 $GW-$G8, R4-R8 $GR-$GR8, survives $SURV"
[ "$SURV" = 1 ] || { TMSG="skipped (gap does not survive without the top hand, or rows missing)"; exit 0; }
TMSG="plain 12500, 4 joint clips, phase 2.6, j0; per joint: measured |tau|/limit peak / mean / % steps >= 0.98 vs estimate %"
trow TAUM_W4m
trow TAUM_W8m --physics_dt 0.00125
TMSG="done; $TMSG"
