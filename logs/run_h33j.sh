#!/bin/bash
# run_h33j.sh (14, 11-03 12:4x): H33J = timing-robust form-gated segment.
# Mechanism (13997 tables, 11-03 12:1x): at jitter 0.03 contacts fall only 4%, but the fair launch median drops +17.6 -> +8.5
# and the outfield-carry share 0.219 -> 0.100. The lift contact is timing-fragile; 15993's exit-speed outfield survives.
# Lever: training swing-start noise AIB_TIMING_NOISE=0.03 -> cfg.timing_noise_s, U(+-0.03), not ramped = the tables' jitter
# column. The pre-release t_arrive leak is kept; tempo absorbs the offset.
# Frame = H33H (hinge f, floors, lift, lost = miss, ramp hit-only, WDA / WWT).
# Start: H33H's final model if H33H_END says PASS, else h33g_1/model_13498.
# +1000 runs as two 500-it trainer runs, so the +500 diag never runs beside a trainer.
# Falsifiers:
#   Baseline (14, 13:0x): start ckpt at jitter 0.03, 256 envs -> L0 = bins all-fair launch median (H33J_BASE_END).
#   +250: train hit < 0.50 or train launch (bat/launch_angle, EMA mean over all hits) < -3 -> kill.
#   +500: diag at --timing_noise 0.03 (256 envs) needs hit >= 0.80 and bins all-fair launch median >= L0 + 4, else stop.
# Gates at +1000:
#   (i) jitter-0 diag: pose <= 0.215 and key <= 0.113.
#   (ii) jitter-0.03 diag: hit >= 0.85, bins all-fair launch median >= max(L0 + 6, 16), diag fair carry median >= 22, pop-ups <= 0.20.
#   Both pass -> game-scene tables (jitter 0.03 primary).
# Launch gates = h33_bins all-fair median; carry = h32_form.py's in-zone fair median (the diag's row).
# Markers H33J_250_END / H33J_500_END / H33J_1000_END, final H33J_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_DA_L1_W AIB_WAIT_TORSO_W AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W
export AIB_LIFT_W=5 AIB_LIFT_REF=5 AIB_LIFT_SPAN=15 AIB_LAUNCH_LO_ALL=1 AIB_FORM_S=0.2162 AIB_FORM_HINGE_HI=0.245 AIB_FORM_HINGE_W=0.045 AIB_LOST_IS_MISS=1 AIB_SS_CA_HIT_ONLY=1 AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=14.7 AIB_SS_CA_HI=24.7 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=15.5 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_OUTFIELD_W=10 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=35 AIB_LAUNCH_HI_SPAN=15 \
  AIB_LAUNCH_LO_W=10 AIB_LAUNCH_LO_REF=5 AIB_LAUNCH_LO_SPAN=15 AIB_FALL_AFTER_HIT_W=10.0
NSEG=1; WDA=0.0259; WWT=0.00360
TASK=AIB-PitchedBatting-Human33-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
O=logs/h33; mkdir -p $O
M=logs/H33J_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H33J_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
lastv() { grep -a "$1:" $2 | tail -1 | awk '{print $NF}'; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\) .*/\1/p"; }
HJ5=0.80; CJ10=22  # 14 (11-03 13:0x): LJ5 = L0 + 4, LJ10 = max(L0 + 6, 16), L0 = the start ckpt's jitter-0.03 bins all-fair launch median
REW="H33J = H33H frame (form hinge f = clamp((0.245 - pose)/0.045, 0, 1), lift 5 x f, floor 10/5/15 all hits, lost = miss) + training timing noise U(+-0.03) s"
if grep -q "PASS" logs/H33H_END.txt 2>/dev/null; then
  D=$(ls -td $B/*human_pitched_h33h_1 | head -1); LR=$(basename $D); LC=model_$(last $D).pt
else LR=2026-11-03_06-50-43_human_pitched_h33g_1; LC=model_13498.pt; fi
echo "$(date +%H:%M) H33J start $LR/$LC (H33H: $(cut -c1-120 logs/H33H_END.txt 2>/dev/null))" >> $O/progress.txt
PK=0
killrn() { powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }
diag() {  # $1 ckpt, $2 tag, $3 envs, $4 timing noise
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --percept_filter --envs $3 --seed 0 --tag $2 \
    --timing_noise $4 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log > $O/diag_$2.txt 2>&1
  ! grep -q Traceback $O/diag_$2.txt && [ -f logs/diag_bat_contact_$2.json ]
}
binl() { $P logs/h33_bins.py $1 2>/dev/null | grep "all fair" | head -1 | sed -n 's/.*launch med *\([-+0-9.]*\) .*//p'; }  # all-fair launch median
form() { $P logs/h32_form.py logs/diag_bat_contact_$1.json $2 $O/form_$2$3.csv | tail -1; }
fv() { echo "$2" | sed -n "s/.*| $1 \([-+0-9.na]*\)[ \/].*/\1/p"; }  # one field of the h32_form row
TF1=$O/train_j1.txt; TF2=$O/train_j2.txt
MSG="H33J baseline diag"; idle  # baseline row (14): start ckpt at jitter 0.03, 256 envs -> L0
S0=${LC#model_}; S0=${S0%.pt}; TB=h33j_base_${S0}_j03s
diag $B/$LR/$LC $TB 256 0.03 || { MSG="H33J baseline diag crash or no json"; exit 1; }
ERB=$($P logs/e10_row.py logs/diag_bat_contact_$TB.json); FRB=$(form $TB $S0 _j03s); L0=$(binl $TB)
awk -v l="$L0" 'BEGIN{exit !(l ~ /^[-+]?[0-9.]+$/)}' || { MSG="H33J baseline: no bins launch median ('$L0')"; exit 1; }
LJ5=$(awk -v l="$L0" 'BEGIN{printf "%.1f", l + 4}'); LJ10=$(awk -v l="$L0" 'BEGIN{x = l + 6; printf "%.1f", (x > 16 ? x : 16)}')
echo "H33J_BASE_END $(date +%H:%M) $LR/$LC @jitter 0.03 (256 envs): L0 (bins all-fair launch median) $L0 -> LJ5 $LJ5, LJ10 $LJ10; hit $(fld hit "$ERB") diag fair carry $(fv carry "$FRB"); $ERB; FORM $FRB" > logs/H33J_BASE_END.txt
cat logs/H33J_BASE_END.txt >> $O/progress.txt
for ph in 1 2; do
  [ -f $O/STOP ] && { MSG="stopped by 14 before H33J part $ph"; exit 0; }
  MSG="H33J part $ph"; idle
  RN=human_pitched_h33j_$ph; TF=$O/train_j$ph.txt
  AIB_TIMING_NOISE=0.03 AIB_DA_L1_W=$WDA AIB_WAIT_TORSO_W=$WWT AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
    --resume True --load_run $LR --checkpoint $LC > $TF 2>&1 &
  TP=$!; I0=${LC#model_}; I0=${I0%.pt}; EF=$((ph - 1))
  while kill -0 $TP 2>/dev/null; do
    if [ $EF -eq 0 ] && grep -aq "Learning iteration $((I0 + 250))/" $TF; then
      EF=1; CV=$($P logs/h33_curve.py $TF 50 $((I0 + 250)))
      HV=$(echo "$CV" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p'); LV=$(echo "$CV" | sed -n 's/.* la \([-0-9.]*\)$/\1/p')
      echo "$(date +%H:%M) H33J +250 check: $CV" >> $O/progress.txt; echo "H33J_250_END $(date +%H:%M) $CV" > logs/H33J_250_END.txt
      if ! awk -v h="$HV" -v l="$LV" 'BEGIN{exit !(h ~ /^[0-9.]+$/ && l ~ /^-?[0-9.]+$/ && h + 0 >= 0.50 && l + 0 >= -3)}'; then
        killrn $RN; MSG="H33J FALSIFIER at +250 (train hit '$HV' >= 0.50, train launch '$LV' >= -3): $CV -> trainer killed, stopped"; exit 0
      fi
    fi
    G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then killrn $RN; MSG="RAM watchdog: H33J part $ph trainer private ${G} GB > 30, killed"; exit 1; fi
    sleep 30
  done
  grep -q Traceback $TF && fail $TF train_j$ph
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="H33J part $ph: no AIB_PHYSICS_DT line"; exit 1; }
  D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="H33J part $ph: no checkpoint"; exit 1; }
  LR=$(basename $D); LC=model_$I.pt
  [ $ph -eq 2 ] && break
  MSG="H33J +500 diag"; idle  # +500 falsifier: jitter-0.03 diag row, 256 envs
  T=h33j_${I}_j03s; diag $D/model_$I.pt $T 256 0.03 || { MSG="H33J +500 diag crash or no json"; exit 1; }
  ER=$($P logs/e10_row.py logs/diag_bat_contact_$T.json); FR=$(form $T $I _j03s)
  HT=$(fld hit "$ER"); LA=$(binl $T)
  echo "H33J_500_END $(date +%H:%M) model_$I @jitter 0.03 (256 envs): $ER; bins all-fair launch median $LA (gate >= $LJ5); FORM $FR; train $($P logs/h33_curve.py $TF 50 $I)" > logs/H33J_500_END.txt
  cat logs/H33J_500_END.txt >> $O/progress.txt
  awk -v h="$HT" -v l="$LA" -v hm=$HJ5 -v lm=$LJ5 'BEGIN{exit !(h ~ /^[0-9.]+$/ && l ~ /^[-+]?[0-9.]+$/ && h + 0 >= hm && l + 0 >= lm)}' \
    || { MSG="H33J FALSIFIER at +500 (jitter-0.03 hit '$HT' >= $HJ5, bins all-fair launch median '$LA' >= $LJ5 = L0 $L0 + 4) -> stopped"; exit 0; }
done
MSG="H33J +1000 eval"; idle
T0=h33j_$I; diag $D/model_$I.pt $T0 1024 0 || { MSG="H33J jitter-0 diag crash or no json"; exit 1; }
idle
T3=h33j_${I}_j03; diag $D/model_$I.pt $T3 1024 0.03 || { MSG="H33J jitter-0.03 diag crash or no json"; exit 1; }
ER0=$($P logs/e10_row.py logs/diag_bat_contact_$T0.json); FR0=$(form $T0 $I ""); PU0=$($P logs/popup_share.py logs/diag_bat_contact_$T0.json)
ER3=$($P logs/e10_row.py logs/diag_bat_contact_$T3.json); FR3=$(form $T3 $I _j03); PU3=$($P logs/popup_share.py logs/diag_bat_contact_$T3.json)
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_$I.csv > $O/joint_s6_$I.txt 2>&1
if [ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt; then JS=$($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1); else JS=crashed; fi
ROW="+1000 model_$I | JITTER 0: $ER0; pop-ups $PU0; FORM $FR0 | JITTER 0.03: $ER3; pop-ups $PU3; bins all-fair launch median $(binl $T3) (L0 $L0); FORM $FR3 | joint s6 (R6b): $JS | episode ends $(for x in hit fall passed lost timeout other; do echo -n "$x $(lastv bat/end_$x $TF2) "; done)| $($P logs/h33_curve.py $TF1 100) | $($P logs/h33_curve.py $TF2 100) | $REW | private peak ${PK} GB"
echo "H33J_1000_END $(date +%H:%M) $ROW" > logs/H33J_1000_END.txt; echo "$(date +%H:%M) $ROW" >> $O/progress.txt; ROWS=" $ROW"
PO=$(fv pose "$FR0"); KE=$(fv key "$FR0")
awk -v p="$PO" -v q="$KE" 'BEGIN{n = "^[0-9.]+$"; exit !(p ~ n && q ~ n && p + 0 <= 0.215 && q + 0 <= 0.113)}' \
  || { MSG="H33J FORM gate FAILED (jitter-0 pose '$PO' <= 0.215, key '$KE' <= 0.113) -> stopped"; exit 0; }
HT=$(fld hit "$ER3"); LA=$(binl $T3); CA=$(fv carry "$FR3"); PUV=$(echo "$PU3" | awk '{print $1}')
awk -v h="$HT" -v l="$LA" -v c="$CA" -v u="$PUV" -v lm=$LJ10 -v cm=$CJ10 'BEGIN{n = "^[-+]?[0-9.]+$"; exit !(h ~ n && l ~ n && c ~ n && u ~ n && h + 0 >= 0.85 && l + 0 >= lm && c + 0 >= cm && u + 0 <= 0.20)}' \
  || { MSG="H33J jitter-0.03 rule FAILED (hit '$HT' >= 0.85, bins all-fair launch median '$LA' >= $LJ10 = max(L0 $L0 + 6, 16), diag fair carry median '$CA' >= $CJ10, pop-ups '$PUV' <= 0.20) -> stopped"; exit 0; }
MSG="H33J gates PASSED -> tables"; bash logs/run_h33_tables.sh $D/model_$I.pt $I $T0 > logs/h33_tables_$I.out 2>&1
MSG="PASS: H33J gates (+1000) | $(cut -c1-300 logs/H33_TABLES_${I}_END.txt)"
