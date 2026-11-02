#!/bin/bash
# H32 joint s7 for 15993 + R6b (planning 14, 11-02 08:2x): second seed of the recover rule on the 15993 candidate (s6 = the chain's
# joint_s6_15993: after-done ground 3/64, tilt ok 56). Same joint_smoke flags as the chain's s6 row, seed 7, 6.5 s, beside segment 4
# (Isaac <= 1, RAM >= 20 GB, no film). Pass = s6 + s7 after-done ground <= 4/128 and tilt ok >= 52 each. Marker logs/H32_J7_15993_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_WAIT_TORSO_W AIB_FALL_AFTER_HIT_W
O=logs/h32; BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-02_04-02-58_human_pitched_h32_3/model_15993.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(tr -d '\r\n' < logs/recov6/recover_ckpt.txt)
M=logs/H32_J7_15993_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H32_J7_15993_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 7 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s7_15993.csv > $O/joint_s7_15993.txt 2>&1
[ -f $O/joint_s7_15993.csv ] && ! grep -q Traceback $O/joint_s7_15993.txt || { MSG="FAILED: joint crash $(grep -A14 Traceback $O/joint_s7_15993.txt | tail -1)"; exit 1; }
MSG="15993 + $(basename $(dirname $RC)) $(basename $RC .pt) s7:$($P logs/film_pick.py $O/joint_s7_15993.csv $O/film_pick_scratch.txt | head -1) | s6:$($P logs/film_pick.py $O/joint_s6_15993.csv $O/film_pick_scratch.txt | head -1)"
