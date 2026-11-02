#!/bin/bash
# H33 (planning 14, 11-02 18:3x; user option B - the batter's form looked awkward: swing pose_err 0.32 rad at 15993 vs 0.16 at
# 12500): FORM FIRST. AIB-PitchedBatting-Human33-v0 = Human32 (plain PitchedBattingEnv, percept_filter, no saim) with every positive
# distance reward (ss ramp, fair exit hinge, carry 15/35, outfield 10, spray) x f = exp(-(pose_swing / 0.2402)^2) (f = 0.5 at
# 0.20 rad; pose_swing = the episode's mean unweighted joint RMS error, clip start -> contact); launch ceiling 5 / 35 / 15, grounder
# floor 5 / 0 / 15, miss, fall, da L1 0.2, wait torso L1 0.03 stay additive. From plain 12500 (h9b) at 1.0x, 800 Hz, 2048 envs,
# 500-it segments, NSEG 4 (the user's budget: +2000 its). Ramp / hinge re-based on 12500 @1.0x as H32 was on 14496 (ss_ca LO = the
# start ss_contact median 14.7, HI +10; exit hinge ref = the start in-zone exit median 15.5). Smoke 64 envs 20 its (mimic/form_f
# must appear) -> logs/H33_SMOKE_END.txt. Per segment: STOP / HOLD (logs/h33/HOLD, k > 1) at loop top, idle() + nofilm before each
# Isaac launch, RAM watchdog 30 GB; eval = game-scene diag (--form_log) + bins + joint s6 with R6b -> logs/H33_<its>_END.txt.
# Gates, else STOP: swing pose_err mean <= 0.20 and key_err mean <= 0.11 (HARD - a form trip with fair carry < 15 m = the frame is
# wrong), hit >= 0.40 (seg 1) / >= previous - 0.02, exit >= previous - 0.3, da <= 0.60, in-zone fair launch median 5..30,
# pop-ups <= 0.20. Fell by end not gated. Final marker logs/H33_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_FORM_S
export AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=14.7 AIB_SS_CA_HI=24.7 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=15.5 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_OUTFIELD_W=10 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=35 AIB_LAUNCH_HI_SPAN=15 \
  AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 AIB_LAUNCH_LO_SPAN=15 AIB_DA_L1_W=0.2 AIB_WAIT_TORSO_W=0.03 AIB_FALL_AFTER_HIT_W=10.0
NSEG=${NSEG:-4}
TASK=AIB-PitchedBatting-Human33-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
O=logs/h33; mkdir -p $O
M=logs/H33_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H33_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
lastv() { grep -a "$1:" $2 | tail -1 | awk '{print $NF}'; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\) .*/\1/p"; }
REW="form x exp(-(pose/0.2402)^2) on ramp ${AIB_SS_CA_LO}-${AIB_SS_CA_HI} win ${AIB_SS_CA_WIN} / exit hinge ${AIB_EXIT_HINGE_W} ref ${AIB_EXIT_HINGE_REF} cap ${AIB_EXIT_HINGE_CAP} / carry ${AIB_CARRY_REF}/${AIB_CARRY_SPAN} / outfield ${AIB_OUTFIELD_W}; additive: launch ceiling ${AIB_LAUNCH_HI_W}/${AIB_LAUNCH_HI_REF}/${AIB_LAUNCH_HI_SPAN}, grounder floor ${AIB_LAUNCH_LO_W}/${AIB_LAUNCH_LO_REF}/${AIB_LAUNCH_LO_SPAN}, da L1 ${AIB_DA_L1_W}, wait torso L1 ${AIB_WAIT_TORSO_W}"
LR=2026-10-25_13-56-51_human_pitched9b; LC=model_12500.pt; PK=0; HPREV=""; EPREV=""
MSG="reward smoke"; idle
AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 20 --run_name h33_smoke --resume True --load_run $LR --checkpoint $LC > $O/smoke.txt 2>&1
if grep -q Traceback $O/smoke.txt; then SM="CRASH $(grep -A14 Traceback $O/smoke.txt | tail -1)"
elif ! grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/smoke.txt; then SM="no AIB_PHYSICS_DT line"
elif FF=$(lastv mimic/form_f $O/smoke.txt); ! awk -v f="$FF" 'BEGIN{exit !(f ~ /^[0-9.]+$/ && f + 0 > 0 && f + 0 <= 1)}'; then SM="mimic/form_f '$FF' not in (0, 1]"
elif ! grep -q "bat/outfield" $O/smoke.txt; then SM="no bat/outfield stat"
else SM="ok: form_f $FF pose_swing $(lastv mimic/form_pose_swing $O/smoke.txt); $($P logs/h33_curve.py $O/smoke.txt 10 | tail -1) ; $(grep -o 'Iteration time: [0-9.]*s' $O/smoke.txt | tail -1)"; fi
echo "H33_SMOKE_END $(date +%H:%M) $SM ($REW)" > logs/H33_SMOKE_END.txt
case "$SM" in ok*) ;; *) MSG="smoke failed: $SM"; exit 1;; esac
for k in $(seq 1 $NSEG); do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  while [ $k -gt 1 ] && [ -f $O/HOLD ]; do MSG="on HOLD before segment $k"; sleep 60; done
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  MSG="segment $k"; idle
  RN=human_pitched_h33_$k
  AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
    --resume True --load_run $LR --checkpoint $LC > $O/train_$k.txt 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: segment $k trainer private ${G} GB > 30, killed"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $O/train_$k.txt && fail $O/train_$k.txt train_$k
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/train_$k.txt || { MSG="segment $k: no AIB_PHYSICS_DT line"; exit 1; }
  D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="segment $k: no checkpoint"; exit 1; }
  T=h33_$I; MSG="segment $k eval"
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ] && { MSG="segment $k: diag crash or no json $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  ER=$($P logs/e10_row.py logs/diag_bat_contact_$T.json)
  FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I $O/form_$I.csv | tail -1)
  PU=$($P logs/popup_share.py logs/diag_bat_contact_$T.json)
  BN=$($P logs/h33_bins.py $T 2>&1 | tail -7 | tr -s ' ' | tr '\n' '/')
  ROW="+$((k * 500)) model_$I (1.0x, 800 Hz, 4 clips, real height, no saim): $ER; pop-ups $PU; FORM $FR; bins $BN"
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s6_$I.csv > $O/joint_s6_$I.txt 2>&1
  if [ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt; then
    ROW="$ROW ; joint s6 (R6b 3091):$($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1)"
  else ROW="$ROW ; joint s6 crashed"; fi
  ROW="$ROW ; $($P logs/h33_curve.py $O/train_$k.txt 100) ; $REW ; $(grep -o 'Iteration time: [0-9.]*s' $O/train_$k.txt | tail -1), private peak ${PK} GB"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H33_$((k * 500))_END $(date +%H:%M) $ROW" > logs/H33_$((k * 500))_END.txt
  ROWS="$ROWS $ROW ||"
  HT=$(fld hit "$ER"); EX=$(fld exit "$ER"); DA=$(fld da "$ER")
  PO=$(echo "$FR" | sed -n 's/.*| pose \([0-9.]*\)\/.*/\1/p'); KE=$(echo "$FR" | sed -n 's/.*| key \([0-9.]*\)\/.*/\1/p')
  FC=$(echo "$FR" | sed -n 's/.*| carry \([0-9.na]*\) .*/\1/p')
  LZ=$($P -c "import json, numpy as np; P = {k: np.array(v, float) for k, v in json.load(open('logs/diag_bat_contact_$T.json'))['per_pitch'].items()}; z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10) & (P['hit'] > 0) & (np.abs(P['spray_contact']) <= 45) & np.isfinite(P['launch_contact']); print(f'{np.median(P[\"launch_contact\"][z]):.1f}')")
  PUV=$(echo "$PU" | awk '{print $1}')
  awk -v p="$PO" -v q="$KE" 'BEGIN{n = "^[0-9.]+$"; exit !(p ~ n && q ~ n && p + 0 <= 0.20 && q + 0 <= 0.11)}' \
    || { FW=$(awk -v c="$FC" 'BEGIN{print (c ~ /^[0-9.]+$/ && c + 0 >= 15) ? "" : " - fair carry under 15 m: the frame is wrong (14 fail condition)"}')
         MSG="segment-$k FORM gate FAILED (pose_err '$PO' <= 0.20, key_err '$KE' <= 0.11; fair carry '$FC')$FW -> stopped"; exit 0; }
  if [ $k -eq 1 ]; then HMIN=0.40; EMIN=0; else HMIN=$(awk -v h="$HPREV" 'BEGIN{print h - 0.02}'); EMIN=$(awk -v e="$EPREV" 'BEGIN{print e - 0.3}'); fi
  awk -v h="$HT" -v e="$EX" -v a="$DA" -v l="$LZ" -v u="$PUV" -v hm="$HMIN" -v em="$EMIN" \
    'BEGIN{n = "^-?[0-9.]+$"; exit !(h ~ n && e ~ n && a ~ n && l ~ n && u ~ n && h + 0 >= hm && e + 0 >= em && a + 0 <= 0.60 && l + 0 >= 5 && l + 0 <= 30 && u + 0 <= 0.20)}' \
    || { MSG="segment-$k rule FAILED (hit '$HT' >= $HMIN, exit '$EX' >= $EMIN, da '$DA' <= 0.60, in-zone fair launch median '$LZ' in 5..30, pop-ups '$PUV' <= 0.20) -> stopped"; exit 0; }
  HPREV=$HT; EPREV=$EX
  LR=$(basename $D); LC=model_$I.pt
done
MSG="PASS: all segment-$NSEG gates (+$((NSEG * 500)))"
