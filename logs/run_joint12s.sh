#!/bin/bash
# run_joint12s.sh (14, 11-06 GO + addendum): joint s6 and s7 of the delivery pair at --seconds 12 (6.5 s rows stay the film
# reference) to see what the hit / NaN-exit rows are (pop-ups still airborne at 6.5 s?). After TAKE_PRECHECK_END, Isaac 0,
# no film, before 13:30. Files logs/h33/joint_s{6,7}_16300_h32e_pin_ema05_12s.csv; marker logs/JOINT_S6_12S_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=. AIB_ACT_EMA=0.5 AIB_PRESTART_PIN=1
O=logs/h33
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-04_19-15-28_human_pitched_h32e/model_16300.pt
M=logs/JOINT_S6_12S_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'printf "JOINT_S6_12S_END %s %s\n%s\n" "$(date +%H:%M)" "$MSG" "$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
MSG="waiting for TAKE_PRECHECK_END"; until [ -f logs/TAKE_PRECHECK_END.txt ]; do sleep 30; done
for S in 6 7; do
  until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); [ $((10#$H)) -lt 1330 ] || { MSG="s$S not launched: $H is past 13:30"; exit 0; }
  [ -f $O/STOP ] && { MSG="stopped by STOP before s$S"; exit 0; }
  F=$O/joint_s${S}_16300_h32e_pin_ema05_12s; MSG="joint s$S 12 s"
  timeout 1800 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $S --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 12 \
    --csv $F.csv > $F.txt 2>&1
  [ -f $F.csv ] && ! grep -q Traceback $F.txt || { MSG="s$S crash: $(grep -A14 Traceback $F.txt | tail -1)"; exit 1; }
  ROWS="$ROWS
s$S ($F.csv): $($P logs/joint12s_cmp.py $O/joint_s${S}_16300_h32e_pin_ema05.csv $F.csv)"
done
MSG="done: $O/joint_s6_16300_h32e_pin_ema05_12s.csv + $O/joint_s7_16300_h32e_pin_ema05_12s.csv"
