#!/bin/bash
# CHAIN + GRIP (planning 14, 10-31 22:23). After H30C800B_END (newer than this chain's start) + idle; the 4 joint clips, real
# heights, phase 2.6, j0, P3c releases. One chain, two markers:
#  CHAIN_END: plain 12500, hard grip, --chain_log (hits: each joint group's share of the sweet-spot velocity at contact) and
#    --grip_log (every in-zone swing: r_hand grip point vs bat grip point gap, relative speed; the bat is welded to l_hand, so the
#    loop's only slack is at the top hand) at 400 Hz (CH_W4) and 800 Hz (CH_W8)
#  GRIP_END: compliant top-hand grip (--grip_soft K C: USD copy, spherical loop disabled, D6 loop with translation springs; hard
#    asset untouched, default off) at 800 Hz, two stiffnesses:
#      K 80000 / C 400: about the hard loop's effective stiffness at 400 Hz (hand-scale mass 0.5 kg / dt^2), critically damped
#      K 20000 / C 200: four times softer, critically damped
#    for plain 12500 (G8_*) and 13298 + saim (S8_*); target ss_max back near 21 with hit ~0.48.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
B=logs/rsl_rl/aib1_tee_batting_mimic
H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
H1=$(ls -td $B/*human_pitched30c_long1 | head -1)/model_13298.pt
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/grip; mkdir -p $O
rm -f logs/CHAIN_END.txt logs/GRIP_END.txt
CMSG="aborted"; COUT=""; GMSG="aborted (CHAIN did not finish)"; GOUT=""
touch $O/chain.stamp
trap 'echo "CHAIN_END $(date +%H:%M) $CMSG$COUT" > logs/CHAIN_END.txt; cat logs/CHAIN_END.txt; echo "GRIP_END $(date +%H:%M) $GMSG$GOUT" > logs/GRIP_END.txt; cat logs/GRIP_END.txt' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, $2 ckpt, rest: extra flags -> prints the row
  local TG=$1 CK=$2; shift 2
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $CK --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log --grip_log \
    "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    echo "$TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  echo "$TG: $($P $SC/grip_row.py logs/diag_bat_contact_$TG.json)"
}
until [ -f logs/H30C800B_END.txt ] && [ logs/H30C800B_END.txt -nt $O/chain.stamp ]; do sleep 60; done
CMSG="plain 12500, hard grip, 4 joint clips, phase 2.6, j0"
COUT="$COUT | $(row CH_W4 $H0 --chain_log)"
COUT="$COUT | $(row CH_W8 $H0 --chain_log --physics_dt 0.00125)"
CMSG="done; $CMSG"
GMSG="800 Hz, compliant top-hand grip (K N/m / C N s/m), 4 joint clips, phase 2.6, j0"
for KC in "80000 400" "20000 200"; do
  set -- $KC
  GOUT="$GOUT | $(row G8_k$1 $H0 --physics_dt 0.00125 --grip_soft $1 $2)"
done
for KC in "80000 400" "20000 200"; do
  set -- $KC
  GOUT="$GOUT | $(row S8_k$1 $H1 $SAIM --physics_dt 0.00125 --grip_soft $1 $2)"
done
GMSG="done; $GMSG"
