#!/bin/bash
# run_take_precheck.sh (14, 11-06 GO): all-take joint run (64 envs, every env --take_mode 10) with the delivery pair
# (16300 + R6b, AIB_ACT_EMA=0.5), pin ON (AIB_PRESTART_PIN=1) vs pin OFF; scripts/take_precheck.py = joint_smoke.py + --take_npz
# (generated, joint_smoke.py untouched). Waits for logs/FILM22_END.txt + Isaac 0. Marker logs/TAKE_PRECHECK_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=. AIB_ACT_EMA=0.5
O=logs/takepre; mkdir -p $O
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-04_19-15-28_human_pitched_h32e/model_16300.pt
M=logs/TAKE_PRECHECK_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'printf "TAKE_PRECHECK_END %s %s\n%s\n" "$(date +%H:%M)" "$MSG" "$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until { [ -f logs/FILM22_END.txt ] || [ -f $O/GO_NOW ]; } && [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
for f in $PC $RC $BC; do [ -f $f ] || { MSG="no $f"; exit 1; }; done
{ echo "env,swing"; for e in $(seq 0 63); do echo "$e,10"; done; } > $O/take10_all.csv
MSG="waiting for FILM22_END + Isaac 0"
for V in on off; do
  [ -f logs/h33/STOP ] && { MSG="stopped by STOP before pin $V"; exit 0; }
  idle
  MSG="pin $V run"
  if [ $V = on ]; then export AIB_PRESTART_PIN=1; else unset AIB_PRESTART_PIN; fi
  timeout 1500 $P -u scripts/take_precheck.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --seconds 6.5 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 \
    --swing_csv $O/take10_all.csv --csv $O/take10_pin$V.csv --take_npz $O/take10_pin$V.npz > $O/take10_pin$V.txt 2>&1
  [ -f $O/take10_pin$V.csv ] && [ -f $O/take10_pin$V.npz ] && ! grep -q Traceback $O/take10_pin$V.txt \
    || { MSG="pin $V crash: $(grep -A14 Traceback $O/take10_pin$V.txt | tail -1)"; exit 1; }
  ROWS="$ROWS
$($P logs/take_precheck_sum.py "pin $V" $O/take10_pin$V.csv $O/take10_pin$V.npz)"
done
MSG="done (16300 + R6b, EMA 0.5, all 64 envs take mode 10, seed 6)"
