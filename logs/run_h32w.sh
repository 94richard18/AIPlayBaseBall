#!/bin/bash
# run_h32w.sh (14, 11-05 06:xx GO; user: hands / feet tremble; per-group jitter: wrists the largest + least EMA-reduced, wait RMS 11.49):
# H32W = h32e/model_16300 + 300 its, AIB_ACT_EMA 0.5 in the loop (as H32E), everything else = H32E, plus ONE term: L1 on the 5 Hz
# high-passed wrist joint velocity (6 wrist joints) before the release only (AIB_WAIT_WRIST_W 0.033 = ~8 / ep at the start: rt_wt_unit
# ~62 / ep x wrist / torso RMS 11.49 / 2.95), 100-it linear ramp. Falsifiers at +150 / +250 (50-it means vs this run's first 50 its):
# ss_ca < 23.3, hit < start - 0.08, fall share > 1.5x and > +0.01. End: EMA-on j0 1024 diag with --jit_groups vs 16300: hit >= 0.90,
# exit >= 29.5, pose <= 0.30, wait wrists <= 8.0, no other group > 5% worse than jg_16300 in any window -> adoption chain (pinned s6,
# tables, pinned s6+s7 recover). No Isaac launch after 16:00, trainer killed at 15:55 (film20 16:25).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.5 AIB_WAIT_WRIST_W=0.033 AIB_WAIT_WRIST_RAMP_STEPS=3200
TASK=AIB-PitchedBatting-Human32-v0
B=logs/rsl_rl/aib1_tee_batting_mimic; LR=$(basename $(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*_human_pitched_h32e | head -1)); LC=model_16300.pt; I0=16300
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
O=logs/h33; RN=human_pitched_h32w; TF=$O/train_h32w.txt
M=logs/H32W_END.txt; rm -f $M logs/H32W_150_END.txt logs/H32W_250_END.txt
MSG="aborted"
trap 'echo "H32W_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); if [ $((10#$H)) -ge 1600 ]; then MSG="no Isaac launch at $H (film20 16:25): stopped before $MSG"; exit 0; fi; }
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
[ -f $O/STOP ] && { MSG="stopped by STOP before H32W"; exit 0; }
MSG="smoke"; idle
AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 2 --run_name h32w_smoke \
  --resume True --load_run $LR --checkpoint $LC > $O/smoke_h32w.txt 2>&1
grep -q Traceback $O/smoke_h32w.txt && { MSG="smoke crash: $(grep -A14 Traceback $O/smoke_h32w.txt | tail -1)"; exit 1; }
SD=$(ls -td $B/*h32w_smoke 2>/dev/null | head -1)
YD=$($P logs/yaml_diff.py $B/$LR/params/env.yaml $SD/params/env.yaml w_wait_wrist wait_wrist_ramp_steps act_ema num_envs act_ema_commit_s act_ema_ramp_s form_hinge_w lost_ends_after_hit)
grep -q "act_ema: 0.5" $SD/params/env.yaml || YD="DIFF act_ema not 0.5 in env.yaml | $YD"
grep -q "w_wait_wrist: 0.033" $SD/params/env.yaml || YD="DIFF w_wait_wrist not 0.033 | $YD"
grep -q "wait_wrist_ramp_steps: 3200" $SD/params/env.yaml || YD="DIFF wait_wrist_ramp_steps not 3200 | $YD"
grep -aq "rt_wwr_unit" $O/smoke_h32w.txt || YD="DIFF no rt_wwr_unit in the smoke log | $YD"
echo "$(date +%H:%M) H32W smoke (16300, EMA 0.5, wait wrist 0.033 ramp 3200): $YD" >> $O/progress.txt
case "$YD" in OK*) ;; *) MSG="env.yaml check failed: $(echo "$YD" | cut -c1-500)"; exit 1;; esac
MSG="H32W training"; idle
AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name $RN \
  --resume True --load_run $LR --checkpoint $LC > $TF 2>&1 &
TP=$!; NC=0; PK=0
while kill -0 $TP 2>/dev/null; do
  H=$(date +%H%M); if [ $((10#$H)) -ge 1555 ]; then killrn $RN; MSG="clock: H32W trainer killed at $H (film20 16:25)"; exit 0; fi
  for CI in $((I0 + 150)) $((I0 + 250)); do
    if [ $NC -lt $([ $CI -eq $((I0 + 150)) ] && echo 1 || echo 2) ] && grep -aq "Learning iteration $CI/" $TF; then
      NC=$((NC + 1)); CK=$($P logs/h32q_check.py $TF $I0 $((I0 + 50)) $TF $((CI - 50)) $CI 23.3)
      WU=$($P logs/rt_mean.py $TF $((CI - 50)) $CI rt_wwr_unit rt_wait_wrist rt_wt_unit)
      K=$((CI - I0)); echo "H32W_${K}_END $(date +%H:%M) check at $CI: $CK | $WU" > logs/H32W_${K}_END.txt; echo "$(date +%H:%M) H32W check $CI: $CK | $WU" >> $O/progress.txt
      case "$CK" in KILL*) killrn $RN; MSG="H32W FALSIFIER at $CI (50-it means, ss floor 23.3): $CK | $WU -> trainer killed"; exit 0;; esac
    fi
  done
  G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then killrn $RN; MSG="RAM watchdog: H32W trainer private ${G} GB > 30, killed"; exit 1; fi
  sleep 30
done
grep -q Traceback $TF && { MSG="train crash: $(grep -A14 Traceback $TF | tail -1)"; exit 1; }
grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="no AIB_PHYSICS_DT line"; exit 1; }
D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] && [ "$I" -ge $((I0 + 299)) ] || { MSG="H32W ended at model_$I before +300"; exit 1; }
BC=$D/model_$I.pt; T=h32w_$I
CV="$($P logs/h32q_check.py $TF $I0 $((I0 + 50)) $TF $((I - 50)) $I 23.3 | cut -d'|' -f2-) | $($P logs/rt_mean.py $TF $((I - 50)) $I rt_wwr_unit rt_wait_wrist rt_wt_unit)"
MSG="diag"; idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $BC --percept_filter --envs 1024 --seed 0 --tag $T \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
  --physics_dt 0.00125 --contact_log --form_log --jit_groups > $O/diag_$T.txt 2>&1
! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ] || { MSG="diag crash"; exit 1; }
G=$($P logs/ema_gate.py form_15993 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_15993_h32_pin.csv | cut -d'|' -f2-3)
G2=$($P logs/ema_gate.py h32e_16300 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_15993_h32_pin.csv | cut -d'|' -f2-3)
FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I $O/form_$I.csv | tail -1)
WG=$($P logs/h32w_gate.py jg_16300 $T)
TE=$(echo "$G" | sed -n 's/.* test: \(.*\)$/\1/p')
HT=$(fld hit " $TE"); EX=$(fld exit " $TE"); WT=$(fld wait_torso_hp " $TE"); PO=$(echo "$FR" | sed -n 's/.*| pose \([0-9.]*\)\/.*/\1/p')
ROW="model_$I (H32W 16300 +300, EMA 0.5, wait wrist 0.033) | WRIST GATE $WG | diag j0 1024 vs 16300:$G2 | vs 15993:$G | FORM $FR | train +50 vs end: $CV | private peak ${PK} GB"
echo "$(date +%H:%M) H32W +300: $ROW" >> $O/progress.txt
OK3=$(awk -v h="$HT" -v e="$EX" -v p="$PO" 'BEGIN{n = "^[0-9.]+$"; print (h ~ n && e ~ n && p ~ n && h + 0 >= 0.90 && e + 0 >= 29.5 && p + 0 <= 0.30) ? 1 : 0}')
[ "$OK3" = 1 ] && [ "${WG%% *}" = PASS ] || { MSG="H32W model_$I FAIL (hit '$HT' >= 0.90, exit '$EX' >= 29.5, pose '$PO' <= 0.30, wrist gate ${WG%% *}; wait_torso_hp '$WT'), 16300 stays: $ROW"; exit 0; }
MSG="joint s6 pinned"; idle
joint 6 joint_s6_${I}_h32w_pin_ema05 || { MSG="joint s6 crash"; exit 1; }
JG=$($P logs/ema_gate.py form_15993 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_${I}_h32w_pin_ema05.csv | sed -n 's/.*| joint test \(.*\)$/\1/p')
FB=$(fld fbd " $JG"); OF=$(fld of " $JG")
echo "$(date +%H:%M) H32W joint s6 pinned: $JG" >> $O/progress.txt; echo "H32W_${I}_JOINT_END $(date +%H:%M) $JG" > logs/H32W_${I}_JOINT_END.txt
awk -v f="$FB" -v o="$OF" 'BEGIN{n = "^[0-9.]+$"; exit !(f ~ n && o ~ n && f + 0 <= 4 && o + 0 >= 15)}' \
  || { MSG="H32W pinned joint s6 FAIL (before-done '$FB' <= 4, OF '$OF' >= 15): $JG"; exit 0; }
MSG="tables"; idle
bash logs/run_h33_tables.sh $BC $I $T > logs/h33_tables_$I.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_${I}_END.txt)
Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% .*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
echo "H32W_${I}_TABLES_END $(date +%H:%M) $TM | 15993 ref 23.25 / 27.69" > logs/H32W_${I}_TABLES_END.txt
awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 16.93 && b + 0 >= 16.76 && oa + 0 >= 3 && ob + 0 >= 3)}' \
  || { MSG="H32W tables below the bars ($TM)"; exit 0; }
MSG="joint s7 pinned"; idle
joint 7 joint_s7_${I}_h32w_pin_ema05 || { MSG="joint s7 crash"; exit 1; }
R2=$($P logs/jsum.py "$O/joint_s{}_${I}_h32w_pin_ema05.csv" 6 7 | sed -n 's/.*TOTAL 2 seeds: after-done ground \([0-9]*\)\/.*/\1/p')
T2=$($P logs/jsum.py "$O/joint_s{}_${I}_h32w_pin_ema05.csv" 6 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p'); T3=$($P logs/jsum.py "$O/joint_s{}_${I}_h32w_pin_ema05.csv" 7 | sed -n 's/.*tilt ok \([0-9]*\) .*/\1/p')
V=$(awk -v g="$R2" -v a="$T2" -v b="$T3" 'BEGIN{print (g ~ /^[0-9]+$/ && g + 0 <= 4 && a + 0 >= 52 && b + 0 >= 52) ? "PASS" : "FAIL"}')
echo "H32W_${I}_RECOVER_END $(date +%H:%M) recover rule $V (pinned s6+s7 after-done ground $R2/128 <= 4, tilt ok $T2 / $T3 >= 52)" > logs/H32W_${I}_RECOVER_END.txt
[ $V = PASS ] || { MSG="H32W recover rule FAIL ($R2/128, tilt $T2 / $T3)"; exit 0; }
MSG="PASS: H32W model_$I passes every gate (tables $TM; recover $R2/128; $ROW) -> candidate to replace 16300"
