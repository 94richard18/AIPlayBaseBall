#!/bin/bash
# run_pin_cmp.sh (14, 11-03 21:xx): AIB_PRESTART_PIN=1 strict compare - joint s6 15993 + R6b (64 envs, the Pareto chain's flags)
# with the pin vs logs/h33/joint_s6_15993_h32.csv (pin off). Marker logs/PIN_CMP_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE AIB_TEMPO_SCRIPT
O=logs/h33; M=logs/PIN_CMP_END.txt; rm -f $M; MSG="aborted"
trap 'echo "PIN_CMP_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-02_04-02-58_human_pitched_h32_3/model_15993.pt
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
until [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
AIB_PRESTART_PIN=1 timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_15993_pin.csv > $O/joint_s6_15993_pin.txt 2>&1
[ -f $O/joint_s6_15993_pin.csv ] && ! grep -q Traceback $O/joint_s6_15993_pin.txt || { MSG="joint crash $(grep -A14 Traceback $O/joint_s6_15993_pin.txt | tail -1)"; exit 1; }
MSG="$($P logs/csv_cmp.py $O/joint_s6_15993_h32.csv $O/joint_s6_15993_pin.csv)"
