#!/bin/bash
# run_h32e_post.sh (11-04 21:0x): H32E model_16300 (newest ckpt of the run killed at 16393) diag with EMA 0.5 on = hit 0.915, exit 30.2,
# pose 0.276, wait_torso_hp 2.95 (-19% vs 3.64; partial, > 2.55). It beats 15993 on hit / exit / pose, so run the rest of the
# H32E gate list on it (pinned joint s6 -> tables -> pinned s7 + recover rule), unchanged from run_h32e.sh. No Isaac 05:15-12:00.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.5
TASK=AIB-PitchedBatting-Human32-v0
B=logs/rsl_rl/aib1_tee_batting_mimic; LR=2026-11-02_04-02-58_human_pitched_h32_3; LC=model_15993.pt; I0=15993; T1=logs/h33/train_h32e.txt
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
O=logs/h33; RN=human_pitched_h32e; TF=$O/train_h32e.txt
M=logs/H32E_POST_END.txt; rm -f $M logs/H32E_TABLES_END.txt logs/H32E_RECOVER_END.txt
MSG="aborted"
trap 'echo "H32E_POST_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); if [ $((10#$H)) -ge 515 ] && [ $((10#$H)) -lt 1200 ]; then MSG="no Isaac launch at $H (film19 window 05:15-12:00): stopped before $MSG"; exit 0; fi; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
killrn() { powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\).*/\1/p"; }
joint() {  # $1 seed, $2 csv stem (pinned)
  AIB_PRESTART_PIN=1 timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/$2.csv > $O/$2.txt 2>&1
  [ -f $O/$2.csv ] && ! grep -q Traceback $O/$2.txt
}
D=$(ls -td $B/*$RN | head -1); I=16300; BC=$D/model_$I.pt; T=h32e_$I
[ -f $BC ] && [ -f logs/diag_bat_contact_$T.json ] || { MSG="no $BC or diag json"; exit 1; }
[ -f $O/STOP ] && { MSG="stopped by STOP"; exit 0; }
MSG="joint s6 pinned"; idle
joint 6 joint_s6_${I}_h32e_pin_ema05 || { MSG="joint s6 crash"; exit 1; }
JG=$($P logs/ema_gate.py form_15993 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_${I}_h32e_pin_ema05.csv | sed -n 's/.*| joint test \(.*\)$/\1/p')
FB=$(fld fbd " $JG"); OF=$(fld of " $JG")
echo "$(date +%H:%M) H32E joint s6 pinned: $JG" >> $O/progress.txt
awk -v f="$FB" -v o="$OF" 'BEGIN{n = "^[0-9.]+$"; exit !(f ~ n && o ~ n && f + 0 <= 4 && o + 0 >= 15)}' \
  || { MSG="H32E pinned joint s6 FAIL (before-done '$FB' <= 4, OF '$OF' >= 15): $JG"; exit 0; }
MSG="tables"; idle
bash logs/run_h33_tables.sh $BC $I $T > logs/h33_tables_$I.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_${I}_END.txt)
Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% .*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
echo "H32E_TABLES_END $(date +%H:%M) $TM | 15993 ref 23.25 / 27.69" > logs/H32E_TABLES_END.txt
awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 16.93 && b + 0 >= 16.76 && oa + 0 >= 3 && ob + 0 >= 3)}' \
  || { MSG="H32E tables below the bars ($TM)"; exit 0; }
MSG="joint s7 pinned"; idle
joint 7 joint_s7_${I}_h32e_pin_ema05 || { MSG="joint s7 crash"; exit 1; }
R2=$($P logs/jsum.py "$O/joint_s{}_${I}_h32e_pin_ema05.csv" 6 7 | sed -n 's/.*TOTAL 2 seeds: after-done ground \([0-9]*\)\/.*/\1/p')
T2=$($P logs/jsum.py "$O/joint_s{}_${I}_h32e_pin_ema05.csv" 6 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p'); T3=$($P logs/jsum.py "$O/joint_s{}_${I}_h32e_pin_ema05.csv" 7 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p')
V=$(awk -v g="$R2" -v a="$T2" -v b="$T3" 'BEGIN{print (g ~ /^[0-9]+$/ && g + 0 <= 4 && a + 0 >= 52 && b + 0 >= 52) ? "PASS" : "FAIL"}')
echo "H32E_RECOVER_END $(date +%H:%M) recover rule $V (pinned s6+s7 after-done ground $R2/128 <= 4, tilt ok $T2 / $T3 >= 52)" > logs/H32E_RECOVER_END.txt
[ $V = PASS ] || { MSG="H32E recover rule FAIL ($R2/128, tilt $T2 / $T3)"; exit 0; }
MSG="PASS: H32E model_$I (killed run, newest ckpt) passes every gate (tables $TM; recover $R2/128) -> candidate to replace 15993"
