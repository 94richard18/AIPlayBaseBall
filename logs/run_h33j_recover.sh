#!/bin/bash
# run_h33j_recover.sh (14, 11-03 20:5x): the recover rule on h33j_1/model_13997 + R6b, as for 15993 + R6b on 11-02 (run_h32_j7.sh /
# run_h32_j89.sh): joint_smoke 64 envs per seed, same flags, seeds 6 (exists: logs/h33/joint_s6_13997j.csv) + 7, then 8 + 9.
# Rule: s6 + s7 after-done ground <= 4/128 and tilt ok >= 52 each; the 4-seed total is reported like 15993's (8/256).
# Each launch at Isaac <= 1, RAM >= 20 GB, no film. Marker logs/H33J_RECOVER_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_WAIT_TORSO_W AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT
O=logs/h33; BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-03_14-16-55_human_pitched_h33j_1/model_13997.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(tr -d '\r\n' < logs/recov6/recover_ckpt.txt)
M=logs/H33J_RECOVER_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33J_RECOVER_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
for SD in 7 8 9; do
  MSG="FAILED: waiting for idle before s$SD"
  until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $SD --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s${SD}_13997j.csv > $O/joint_s${SD}_13997j.txt 2>&1
  [ -f $O/joint_s${SD}_13997j.csv ] && ! grep -q Traceback $O/joint_s${SD}_13997j.txt || { MSG="FAILED: joint s$SD crash $(grep -A14 Traceback $O/joint_s${SD}_13997j.txt | tail -1)"; exit 1; }
done
R2=$($P logs/jsum.py "$O/joint_s{}_13997j.csv" 6 7 | sed -n 's/.*TOTAL 2 seeds: after-done ground \([0-9]*\)\/.*/\1/p')
T2=$($P logs/jsum.py "$O/joint_s{}_13997j.csv" 6 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p') T3=$($P logs/jsum.py "$O/joint_s{}_13997j.csv" 7 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p')
V=$(awk -v g="$R2" -v a="$T2" -v b="$T3" 'BEGIN{print (g ~ /^[0-9]+$/ && g + 0 <= 4 && a + 0 >= 52 && b + 0 >= 52) ? "PASS" : "FAIL"}')
MSG="recover rule $V (s6+s7 after-done ground $R2/128 <= 4, tilt ok s6 $T2 / s7 $T3 >= 52) | 13997j + $(basename $(dirname $RC)) $(basename $RC .pt) | $($P logs/jsum.py "$O/joint_s{}_13997j.csv" 6 7 8 9) | 15993 ref: 8/256, tilt 56.5"
