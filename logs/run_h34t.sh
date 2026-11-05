#!/bin/bash
# run_h34t.sh (14, 11-05 GO 2, step 2 after H34Q = no inference alpha gives the gain): H34T = h32e/model_16300 fine-tuned 300 its with
# AIB_PIN_UNTIL_RELEASE=0.00 IN the loop (training can relearn the pre-release adjustment) and the action EMA deepened to alpha 0.35
# (AIB_ACT_EMA, trained in), everything else = H32E (reward unchanged, no new term). Segments (Isaac max 1, so no diag beside the
# trainer): +150 -> deterministic diag 16449 -> resume +100 -> diag 16549 -> resume +50 -> end diag 16599. Mid-run stop if hit < 0.88,
# exit < 29.0 or launch < +14 (h34p_gate fields); end gate = h34p_gate PASS + h34q_gain GAIN (app torso / legs / feet <= 16300 - 10%)
# -> pinned joint s6 -> tables (bars + both columns >= 16300 - 1.0 pt: 26.01 / 31.93) -> s7 + recover rule. All runs with alpha 0.35
# + pin0. Launch only after FILM21_END + Isaac 0.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W AIB_WAIT_WRIST_W AIB_WAIT_WRIST_RAMP_STEPS
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.35 AIB_PIN_UNTIL_RELEASE=0.00
TASK=AIB-PitchedBatting-Human32-v0
B=logs/rsl_rl/aib1_tee_batting_mimic; LR0=$(basename $(ls -td $B/*_human_pitched_h32e | head -1)); I0=16300
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
O=logs/h33
M=logs/H34T_END.txt; rm -f $M logs/H34T_*_DIAG_END.txt logs/H34T_*_JOINT_END.txt logs/H34T_*_TABLES_END.txt logs/H34T_*_RECOVER_END.txt
MSG="aborted"; PK=0
trap 'echo "H34T_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ -f logs/FILM21_END.txt ] && [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
killrn() { powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\).*/\1/p"; }
pose() { echo "$1" | sed -n 's/.*| pose \([0-9.]*\)\/.*/\1/p'; }
joint() {  # $1 seed, $2 csv stem
  AIB_PRESTART_PIN=1 timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/$2.csv > $O/$2.txt 2>&1
  [ -f $O/$2.csv ] && ! grep -q Traceback $O/$2.txt
}
diag() {  # $1 ckpt, $2 tag
  timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --percept_filter --envs 1024 --seed 0 --tag $2 \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log --jit_groups > $O/diag_$2.txt 2>&1
  ! grep -q Traceback $O/diag_$2.txt && [ -f logs/diag_bat_contact_$2.json ]
}
train() {  # $1 load run, $2 ckpt, $3 its, $4 run name -> trains in the foreground with the RAM watchdog
  local TF=$O/train_$4.txt
  AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations $3 --run_name $4 \
    --resume True --load_run $1 --checkpoint $2 > $TF 2>&1 &
  local TP=$!
  while kill -0 $TP 2>/dev/null; do
    G=$(priv $4); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then killrn $4; MSG="RAM watchdog: $4 private ${G} GB > 30, killed"; exit 1; fi
    sleep 30
  done
  grep -q Traceback $TF && { MSG="$4 train crash: $(grep -A14 Traceback $TF | tail -1)"; exit 1; }
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="$4: no AIB_PHYSICS_DT line"; exit 1; }
}
[ -f $O/STOP ] && { MSG="stopped by STOP before H34T"; exit 0; }
MSG="waiting for FILM21_END, then smoke"; idle
AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 2 --run_name h34t_smoke \
  --resume True --load_run $LR0 --checkpoint model_$I0.pt > $O/smoke_h34t.txt 2>&1
grep -q Traceback $O/smoke_h34t.txt && { MSG="smoke crash: $(grep -A14 Traceback $O/smoke_h34t.txt | tail -1)"; exit 1; }
SD=$(ls -td $B/*h34t_smoke 2>/dev/null | head -1)
YD=$($P logs/yaml_diff.py $B/$LR0/params/env.yaml $SD/params/env.yaml act_ema pin_until_release_s num_envs w_wait_wrist wait_wrist_ramp_steps)
grep -q "act_ema: 0.35" $SD/params/env.yaml || YD="DIFF act_ema not 0.35 in env.yaml | $YD"
grep -q "pin_until_release_s: 0.0" $SD/params/env.yaml || YD="DIFF pin_until_release_s not 0.0 | $YD"
echo "$(date +%H:%M) H34T smoke (16300, EMA 0.35, pin0): $YD" >> $O/progress.txt
case "$YD" in OK*) ;; *) MSG="env.yaml check failed: $(echo "$YD" | cut -c1-500)"; exit 1;; esac
LR=$LR0; LC=model_$I0.pt
for SEG in a:150 b:100 c:50; do
  RN=human_pitched_h34t${SEG%%:*}; N=${SEG#*:}
  [ -f $O/STOP ] && { MSG="stopped by STOP before segment $RN"; exit 0; }
  MSG="H34T segment $RN ($N its from $LR/$LC)"; idle
  train $LR $LC $N $RN
  D=$(ls -td $B/*_$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] && [ -f $D/model_$I.pt ] || { MSG="no checkpoint after $RN"; exit 1; }
  LR=$(basename $D); LC=model_$I.pt; BC=$D/$LC; T=h34t_$I
  MSG="diag $T"; idle
  diag $BC $T || { MSG="diag $T crash: $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $T $O/form_$T.csv | tail -1)
  GT=$($P logs/h34p_gate.py jg_16300 $T "$(pose "$FR")"); GN=$($P logs/h34q_gain.py jg_16300 $T)
  ROW="model_$I ($RN, alpha 0.35 + pin0 in the loop): $GT | $GN | $($P logs/h34p_ho.py h34p_base $T 000) | FORM $FR | private peak ${PK} GB"
  echo "H34T_${I}_DIAG_END $(date +%H:%M) $ROW" > logs/H34T_${I}_DIAG_END.txt; echo "$(date +%H:%M) H34T $ROW" >> $O/progress.txt
  F1=$(echo "${GT%% |*}" | sed -n 's/^FAIL //p')
  case ",$F1," in *,hit,*|*,exit,*|*,launch,*) MSG="H34T FALSIFIER at model_$I (hit / exit / launch): $ROW -> stopped, 16300 stays"; exit 0;; esac
done
[ "$I" -ge $((I0 + 299)) ] || { MSG="H34T ended at model_$I before +300"; exit 1; }
[ "${GT%% *}" = PASS ] && [ "${GN%% *}" = GAIN ] || { MSG="H34T model_$I end gate FAIL (${GT%% |*}; ${GN%% (*}), 16300 stays: $ROW"; exit 0; }
MSG="joint s6"; idle
joint 6 joint_s6_${I}_h34t || { MSG="joint s6 crash"; exit 1; }
JG=$($P logs/ema_gate.py h32e_16300 $T $O/joint_s6_16300_h32e_pin_ema05.csv $O/joint_s6_${I}_h34t.csv | sed -n 's/.*| joint ref \(.*\)$/\1/p')
FB=$(fld fbd " ${JG#*joint test}"); OF=$(fld of " ${JG#*joint test}")
echo "H34T_${I}_JOINT_END $(date +%H:%M) ref (16300) $JG" > logs/H34T_${I}_JOINT_END.txt; echo "$(date +%H:%M) H34T joint s6: ref $JG" >> $O/progress.txt
awk -v f="$FB" -v o="$OF" 'BEGIN{n = "^[0-9.]+$"; exit !(f ~ n && o ~ n && f + 0 <= 4 && o + 0 >= 15)}' \
  || { MSG="H34T joint s6 FAIL (before-done '$FB' <= 4, OF '$OF' >= 15): ref $JG"; exit 0; }
MSG="tables"; idle
bash logs/run_h33_tables.sh $BC ${I}_h34t $T > logs/h33_tables_${I}_h34t.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_${I}_h34t_END.txt)
Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% .*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
echo "H34T_${I}_TABLES_END $(date +%H:%M) $TM | 16300 ref 27.01 / 32.93, bar - 1.0 pt = 26.01 / 31.93" > logs/H34T_${I}_TABLES_END.txt
awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 26.01 && b + 0 >= 31.93 && oa + 0 >= 3 && ob + 0 >= 3)}' \
  || { MSG="H34T tables below 16300 - 1.0 pt ($TM)"; exit 0; }
MSG="joint s7"; idle
joint 7 joint_s7_${I}_h34t || { MSG="joint s7 crash"; exit 1; }
R2=$($P logs/jsum.py "$O/joint_s{}_${I}_h34t.csv" 6 7 | sed -n 's/.*TOTAL 2 seeds: after-done ground \([0-9]*\)\/.*/\1/p')
T2=$($P logs/jsum.py "$O/joint_s{}_${I}_h34t.csv" 6 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p'); T3=$($P logs/jsum.py "$O/joint_s{}_${I}_h34t.csv" 7 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p')
V=$(awk -v g="$R2" -v a="$T2" -v b="$T3" 'BEGIN{print (g ~ /^[0-9]+$/ && g + 0 <= 4 && a + 0 >= 52 && b + 0 >= 52) ? "PASS" : "FAIL"}')
echo "H34T_${I}_RECOVER_END $(date +%H:%M) recover rule $V (s6+s7 after-done ground $R2/128 <= 4, tilt ok $T2 / $T3 >= 52)" > logs/H34T_${I}_RECOVER_END.txt
[ $V = PASS ] || { MSG="H34T recover rule FAIL ($R2/128, tilt $T2 / $T3)"; exit 0; }
MSG="PASS: H34T model_$I (alpha 0.35 + pin0 trained in) passes every gate (tables $TM; recover $R2/128) -> candidate to replace 16300"
