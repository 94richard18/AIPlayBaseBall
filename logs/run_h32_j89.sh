#!/bin/bash
# H32 joint s8 + s9 for 15993 + R6b (planning 14, 11-02 12:47): measurement only, no decision rides on it (pair stays 15993 + R6b);
# tightens the recover number for the user report ("after-done ground rate over 256 PAs" = s6..s9). Same joint_smoke flags as the
# s6 / s7 rows, 6.5 s, each launch at Isaac <= 1, RAM >= 20 GB, no film. Marker logs/H32_J89_15993_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_WAIT_TORSO_W AIB_FALL_AFTER_HIT_W
O=logs/h32; BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-02_04-02-58_human_pitched_h32_3/model_15993.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(tr -d '\r\n' < logs/recov6/recover_ckpt.txt)
M=logs/H32_J89_15993_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H32_J89_15993_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
for SD in 8 9; do
  MSG="FAILED: waiting for idle before s$SD"
  until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $SD --after_done recover_rl     --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5     --csv $O/joint_s${SD}_15993.csv > $O/joint_s${SD}_15993.txt 2>&1
  [ -f $O/joint_s${SD}_15993.csv ] && ! grep -q Traceback $O/joint_s${SD}_15993.txt || { MSG="FAILED: joint s$SD crash $(grep -A14 Traceback $O/joint_s${SD}_15993.txt | tail -1)"; exit 1; }
done
MSG="15993 + $(basename $(dirname $RC)) $(basename $RC .pt) | $($P logs/h32_jsum.py 6 7 8 9)"
