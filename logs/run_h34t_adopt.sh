#!/bin/bash
# run_h34t_adopt.sh (14, 11-06 04:5x GO (a)): adoption chain for h34ta/model_16449 (alpha 0.35 + pin0 trained in the loop; EMA 0.35,
# PRESTART_PIN, PIN_UNTIL_RELEASE 0.00, R6b recover): joint s6 64 envs (before-done falls <= 4, OF >= 15) -> game-scene tables (bars
# 16.93 / 16.76, OF >= 3%, AND both columns >= 16300 - 1 pt = 26.01 / 31.93; hit rate from the diag h34t_16449) -> joint s7 + recover
# rule over s6 + s7 (after-done ground <= 4 / 128, tilt ok >= 52 each). Markers H34T_16449_JOINT_END / _TABLES_END / _RECOVER_END, chain
# end H34T_ADOPT_END. film22 GO 14:25: no step launches from 13:30 until FILM22_END exists.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W AIB_WAIT_WRIST_W AIB_WAIT_WRIST_RAMP_STEPS
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.35 AIB_PRESTART_PIN=1 AIB_PIN_UNTIL_RELEASE=0.00
B=logs/rsl_rl/aib1_tee_batting_mimic; O=logs/h33; I=16449; T=h34t_16449; S=${I}_h34t
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=$(ls -td $B/*_human_pitched_h34ta | head -1)/model_$I.pt
M=logs/H34T_ADOPT_END.txt; rm -f $M logs/H34T_16449_JOINT_END.txt logs/H34T_16449_TABLES_END.txt logs/H34T_16449_RECOVER_END.txt
MSG="aborted"
trap 'echo "H34T_ADOPT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
f22() { H=$(date +%H%M); [ $((10#$H)) -lt 1330 ] || [ -f logs/FILM22_END.txt ]; }
idle() { until f22 && [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm && f22; do sleep 30; done; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\).*/\1/p"; }
joint() {  # $1 seed, $2 csv stem
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/$2.csv > $O/$2.txt 2>&1
  [ -f $O/$2.csv ] && ! grep -q Traceback $O/$2.txt
}
[ -f $BC ] || { MSG="no $BC"; exit 1; }
[ -f logs/diag_bat_contact_$T.json ] || { MSG="no diag json $T"; exit 1; }
[ -f $O/STOP ] && { MSG="stopped by STOP"; exit 0; }
MSG="joint s6"; idle
joint 6 joint_s6_$S || { MSG="joint s6 crash: $(grep -A14 Traceback $O/joint_s6_$S.txt | tail -1)"; exit 1; }
JG=$($P logs/ema_gate.py h32e_16300 $T $O/joint_s6_16300_h32e_pin_ema05.csv $O/joint_s6_$S.csv | sed -n 's/.*| joint ref \(.*\)$/\1/p')
FB=$(fld fbd " ${JG#*joint test}"); OF=$(fld of " ${JG#*joint test}")
echo "H34T_16449_JOINT_END $(date +%H:%M) joint s6 (16449 EMA 0.35 + PRESTART_PIN + PIN_UNTIL_RELEASE 0.00, R6b) ref (16300 film19 setup) $JG" > logs/H34T_16449_JOINT_END.txt
echo "$(date +%H:%M) H34T 16449 joint s6: ref $JG" >> $O/progress.txt
awk -v f="$FB" -v o="$OF" 'BEGIN{n = "^[0-9.]+$"; exit !(f ~ n && o ~ n && f + 0 <= 4 && o + 0 >= 15)}' \
  || { MSG="H34T 16449 joint s6 FAIL (before-done '$FB' <= 4, OF '$OF' >= 15): ref $JG"; exit 0; }
[ -f $O/STOP ] && { MSG="stopped by STOP before tables"; exit 0; }
MSG="tables"; idle
bash logs/run_h33_tables.sh $BC $S $T > logs/h33_tables_$S.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_${S}_END.txt)
Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% .*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
echo "H34T_16449_TABLES_END $(date +%H:%M) $TM | 16300 ref 27.01 / 32.93 (OF 20.25 / 26.42), bar 26.01 / 31.93" > logs/H34T_16449_TABLES_END.txt
echo "$(date +%H:%M) H34T 16449 tables: $TM" >> $O/progress.txt
awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 26.01 && b + 0 >= 31.93 && oa + 0 >= 3 && ob + 0 >= 3)}' \
  || { MSG="H34T 16449 tables below 16300 - 1 pt (26.01 / 31.93) or OF < 3 ($TM)"; exit 0; }
[ -f $O/STOP ] && { MSG="stopped by STOP before joint s7"; exit 0; }
MSG="joint s7"; idle
joint 7 joint_s7_$S || { MSG="joint s7 crash: $(grep -A14 Traceback $O/joint_s7_$S.txt | tail -1)"; exit 1; }
R2=$($P logs/jsum.py "$O/joint_s{}_$S.csv" 6 7 | sed -n 's/.*TOTAL 2 seeds: after-done ground \([0-9]*\)\/.*/\1/p')
T2=$($P logs/jsum.py "$O/joint_s{}_$S.csv" 6 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p'); T3=$($P logs/jsum.py "$O/joint_s{}_$S.csv" 7 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p')
V=$(awk -v g="$R2" -v a="$T2" -v b="$T3" 'BEGIN{print (g ~ /^[0-9]+$/ && g + 0 <= 4 && a + 0 >= 52 && b + 0 >= 52) ? "PASS" : "FAIL"}')
echo "H34T_16449_RECOVER_END $(date +%H:%M) recover rule $V (s6+s7 after-done ground $R2/128 <= 4, tilt ok $T2 / $T3 >= 52; 16300 ref 2/128, 62 / 62)" > logs/H34T_16449_RECOVER_END.txt
[ $V = PASS ] || { MSG="H34T 16449 recover rule FAIL ($R2/128, tilt $T2 / $T3)"; exit 0; }
MSG="PASS: h34ta/model_16449 (EMA 0.35 + PRESTART_PIN + PIN_UNTIL_RELEASE 0.00, R6b) passes joint s6 ($JG), tables ($TM) and recover ($R2/128) -> new delivery candidate, film22 eye gate"
