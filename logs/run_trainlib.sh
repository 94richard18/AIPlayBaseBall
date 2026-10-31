#!/bin/bash
# (f) falsifier (planning 14, 10-31 17:12): is the training-vs-diag gap of model_13797 (training EMA hit 0.82, fair_carry
# 19.6, launch 12.5 vs diag j0 0.749 / 35.1 / 45.8) the pitch library or the phase input? After S7_13298_END (newer than
# this chain's start), three jitter-0 diag rows of 13797 + row 9 saim (one Isaac at a time):
#  A trainlib    = the training pitch library (Human15+: P3c-shaped releases, mix_frac 1, 55-75 km/h, zone |y| <= 0.2525,
#                  z 0.50-1.10), phase input t / 9.0 s as every diag / game so far;
#  B trainlib_ph = A + phase_horizon 2.6 (the training episode: the phase input as in training);
#  C eval_ph     = the eval releases (as the h30f rows) + phase_horizon 2.6.
# Each: armaim_gate +300 line, carry_cols, diag_fair_carry -> logs/TRAINLIB_DIAG_END.txt (EXIT trap). logs/h30f/GO stays held.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
CK=$(ls -td $B/*human_pitched30c_long2 | head -1)/model_13797.pt
TL="--mix_release_csv $R --mix_frac 1.0 --mix_speed 55 75 --mix_zone_y 0.2525 --mix_zone_z 0.50 1.10"
O=logs/trainlib; mkdir -p $O
M=logs/TRAINLIB_DIAG_END.txt; rm -f $M; MSG="aborted"
touch $O/chain.stamp
trap 'echo "TRAINLIB_DIAG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
until [ -f logs/S7_13298_END.txt ] && [ logs/S7_13298_END.txt -nt $O/chain.stamp ]; do sleep 60; done
OUT=""
for row in "trainlib|$TL" "trainlib_ph|$TL --phase_horizon 2.6" "eval_ph|--release_csv $R --phase_horizon 2.6"; do
  TG=h30cL_13797_${row%%|*}; X=${row#*|}
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $CK $SAIM --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 $X --contact_log > $O/diag_$TG.txt 2>&1
  if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
    OUT="$OUT | ${row%%|*}: $($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_$TG.json --label ${row%%|*}) ; $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label ${row%%|*} | head -1 | cut -c1-200) ; $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label ${row%%|*})"
  else OUT="$OUT | ${row%%|*}: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; fi
  MSG="13797 j0 (training EMA: hit 0.82, fair_carry 19.6, launch 12.5; diag eval j0: 0.749 / 35.1 / 45.8)$OUT"
done
