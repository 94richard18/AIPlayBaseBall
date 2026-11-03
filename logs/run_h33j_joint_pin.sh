#!/bin/bash
# run_h33j_joint_pin.sh (14, 11-03 23:0x): joint s6 of h33j_1/model_13997 + R6b with AIB_PRESTART_PIN=1 for film17 (07:30 GO),
# after EMA_CHAIN_END. csv logs/h33/joint_s6_13997j_pin.csv, marker logs/H33J_JOINT_PIN_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_ACT_EMA
O=logs/h33; M=logs/H33J_JOINT_PIN_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33J_JOINT_PIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-03_14-16-55_human_pitched_h33j_1/model_13997.pt
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
MSG="waiting for EMA_CHAIN_END"; until [ -f logs/EMA_CHAIN_END.txt ] && [ logs/EMA_CHAIN_END.txt -nt logs/run_h33j_joint_pin.sh ]; do sleep 60; done
MSG="joint s6 pin"; until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
AIB_PRESTART_PIN=1 timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_13997j_pin.csv > $O/joint_s6_13997j_pin.txt 2>&1
if [ -f $O/joint_s6_13997j_pin.csv ] && ! grep -q Traceback $O/joint_s6_13997j_pin.txt; then
  MSG="h33j_1/model_13997 + R6b, AIB_PRESTART_PIN=1: $($P logs/film_pick.py $O/joint_s6_13997j_pin.csv $O/film_pick_13997j_pin.txt | head -1) | $($P logs/jsum.py "$O/joint_s6_13997j_pin.csv" 6 | cut -d'|' -f1) | unpinned ref: $($P logs/jsum.py "$O/joint_s6_13997j.csv" 6 | cut -d'|' -f1) | csv $O/joint_s6_13997j_pin.csv"
else MSG="joint crash (see $O/joint_s6_13997j_pin.txt)"; exit 1; fi
