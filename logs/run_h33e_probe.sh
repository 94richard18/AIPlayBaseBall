#!/bin/bash
# H33E probe (planning 14, 11-03 00:xx): is H33D's dodged miss a pre-hit `lost`? 150 its from 12500 on H33D seg-1 settings (da L1 /
# wait torso 0, AIB_LOST_IS_MISS 0) with the episode-end shares logged; read at +10 (the EMA warm-up stands in for +0) / +75 / +150.
# Swap = end_lost up >= 0.10 AND end_passed down >= 0.10 from +10 to +150 -> exec logs/run_h33e.sh (H33E, no wait for 14); else stop.
# Marker logs/H33E_PROBE_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_FORM_S AIB_FORM_ON_HITS AIB_LOST_IS_MISS
export AIB_SS_CA_HIT_ONLY=1 AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=14.7 AIB_SS_CA_HI=24.7 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=15.5 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_OUTFIELD_W=10 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=35 AIB_LAUNCH_HI_SPAN=15 \
  AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 AIB_LAUNCH_LO_SPAN=15 AIB_DA_L1_W=0 AIB_WAIT_TORSO_W=0 AIB_FALL_AFTER_HIT_W=10.0
TASK=AIB-PitchedBatting-Human33-v0; O=logs/h33
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
M=logs/H33E_PROBE_END.txt; rm -f $M; MSG="aborted"; GO=0
trap 'echo "H33E_PROBE_END $(date +%H:%M) $MSG" > $M; cat $M; [ $GO -eq 1 ] && exec bash logs/run_h33e.sh > logs/h33e_run.out 2>&1' EXIT
MSG="waiting for idle"; idle; MSG="training"
AIB_PHYSICS_DT=0.00125 timeout 3600 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 150 \
  --run_name h33e_probe --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_eprobe.txt 2>&1
grep -q Traceback $O/train_eprobe.txt && { MSG="CRASH $(grep -A14 Traceback $O/train_eprobe.txt | tail -1)"; exit 1; }
grep -aq "bat/end_lost" $O/train_eprobe.txt || { MSG="no bat/end_lost stat"; exit 1; }
E=$($P logs/h33_ends.py $O/train_eprobe.txt 12510 12575 12650)
L0=$(echo "$E" | sed -n 1p | sed -n 's/.* lost \([0-9.]*\) .*/\1/p'); L1=$(echo "$E" | sed -n 3p | sed -n 's/.* lost \([0-9.]*\) .*/\1/p')
P0=$(echo "$E" | sed -n 1p | sed -n 's/.* passed \([0-9.]*\) .*/\1/p'); P1=$(echo "$E" | sed -n 3p | sed -n 's/.* passed \([0-9.]*\) .*/\1/p')
SW=$(awk -v a="$L0" -v b="$L1" -v c="$P0" -v d="$P1" 'BEGIN{print (b - a >= 0.10 && c - d >= 0.10) ? "SWAP" : "NO SWAP"}')
MSG="$SW (lost $L0 -> $L1, passed $P0 -> $P1) | $(echo "$E" | tr '\n' '|') curve $($P logs/h33_curve.py $O/train_eprobe.txt 50)"
[ "$SW" = SWAP ] && { MSG="$MSG -> H33E launched"; GO=1; } || MSG="$MSG -> stopped (untaxed-termination theory not shown)"
