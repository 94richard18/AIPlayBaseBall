#!/bin/bash
# H32 (planning 14, 11-01 20:xx; user: batter strength back to human level + no axis-alignment signal, it made the body shake
# in the swing): resume H31 model_14496 at AIB_EFFORT_SCALE 1.0 on AIB-PitchedBatting-Human32-v0 = Human31 rewards on the
# plain PitchedBattingEnv (no saim in the loop, percept_filter on), 800 Hz, 2048 envs, 500-it segments, SAVE_EVERY 100.
# Reward bounds come in from the caller env (AIB_SS_CA_LO / HI / WIN, AIB_EXIT_HINGE_W / REF / CAP, AIB_CARRY_REF / SPAN,
# AIB_LAUNCH_HI_W ...) - set from the E10 1.0x diag. 64-env 20-it smoke -> logs/H32_SMOKE_END.txt, then waits for logs/h32/GO.
# Per segment: STOP before, HOLD between segments (waits while logs/h32/HOLD exists), RAM watchdog 30 GB; eval = diag 4 clips
# (percept_filter, no saim, + jitter) and joint s6 (recover RC, 6.5 s) -> logs/H32_<its>_END.txt; final logs/H32_END.txt.
# Rules (14, 20:5x), else STOP: segment 1 hit >= 0.40, exit >= EXREF - 1, da < 0.57; segment 2 exit >= EXREF, da < 0.50; both:
# wait-phase (start -> release) da / torso HP not above the 14496 1.0x reference (logs/H32_REF.txt) + 5 %. The swing-window
# torso HP is reported, not gated (confounded by the swing). AIB_DA_L1_W 0.1 (all steps) from the caller.
# run_h32_s3.sh (14, 11-02 01:1x): copy of run_h32.sh for segment 3+, launched with K0=3 LR0/LC0 = the seg-2 ckpt if it passed,
# else 14995. Seg-3 rewards from the caller (5e23b74): AIB_CARRY_REF 15 / SPAN 35, AIB_OUTFIELD_W 10 (fair + past the ONE OUTS
# infield, hit step, flight prediction), AIB_LAUNCH_HI_W 5 / 40 / 15, the rest as seg 2. The reward smoke (64 envs, 20 its,
# bat/outfield must appear) runs after HOLD lifts, at Isaac 0 + nofilm -> logs/H32S3_SMOKE_END.txt. Rules k >= 3, else STOP:
# hit >= 0.75, exit >= EX2 - 1, da < 0.50, wait terms <= 1.05 x REF, fell by end <= 0.60; joint outfield >= 10/64 and fair
# carry median > 25 m reported (adoption proxy), the ONE OUTS table decides.
# run_h32_s4.sh (14, 11-02 06:5x, approved): segment 4 only, from seg-3 model_15993 (it failed wait_torso_hp 3.80 > 3.586 and
# fell by end 0.68 > 0.60) = seg-3 rewards + standing term (the episode outlives the follow-through, AIB_FOLLOW_THROUGH_S / EPISODE_S
# on the TRAIN lines only - the diag / joint evals stay as before - with a reduced fall-after-hit penalty AIB_FALL_AFTER_HIT_W) +
# wait-phase torso L1 (AIB_WAIT_TORSO_W). Small weights: the 10-27 standing designs lowered falls by weakening the swing. Gates =
# the seg-3 six + exit >= 27.5 + diag fair carry median >= 30 m (a fail there = standing weights too high). Waits for
# logs/FILM7_END.txt (nofilm). Own smoke -> logs/H32S4B_SMOKE_END.txt (mimic/wait_torso_hp must appear); final marker
# logs/H32S4_END.txt (logs/H32_END.txt keeps the seg-3 stop text).
# run_h32_s4b.sh (14, 11-02 10:0x, approved): seg 4 failed (16492) because the FT option's lost_after_hit_is_fall counted every
# hit as a fall (fall_rate == hit_rate) and lost still ended the episode -> seg 4b from 15993: AIB_FT_LOST_IS_FALL=0 +
# AIB_FT_LOST_ENDS=0 (after the hit only a real pelvis / tilt fall ends the 1.5 s follow-through), fall after hit 3, wait torso
# 0.03, launch ceiling ref 35 (w 5, span 15). Smoke gate: bat/fall_rate must come apart from bat/hit_rate (> 0.05). Gates = seg 4
# + in-zone fair launch median <= 30 + pop-ups <= 0.20. Waits for logs/FILM8_END.txt newer than logs/FILM8_GO.txt (14's GPU order:
# R6c -> film8 -> seg 4b). Names: run human_pitched_h32_4b, train_4b.txt, diag tag h32b_<I>, joint_s6b_<I>, H32_2000B_END,
# smoke H32S4B_SMOKE_END, final logs/H32S4B_END.txt.
cd /d/MyProject/AIPlayBaseBall
: ${EX2:?}
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EFFORT_PROBE
: ${EXREF:?}
NSEG=${NSEG:-6}
TASK=AIB-PitchedBatting-Human32-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
O=logs/h32; mkdir -p $O
: ${FT:?} ${EPS:?} ${AIB_WAIT_TORSO_W:?} ${AIB_FALL_AFTER_HIT_W:?}
M=logs/H32S4B_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H32S4B_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
REW="outfield ${AIB_OUTFIELD_W:-0.0}, ramp ${AIB_SS_CA_LO:-12.0}-${AIB_SS_CA_HI:-22.0} win ${AIB_SS_CA_WIN:-0.05}, exit hinge ${AIB_EXIT_HINGE_W:-0.0} ref ${AIB_EXIT_HINGE_REF:-16.0} cap ${AIB_EXIT_HINGE_CAP:-14.0}, carry ${AIB_CARRY_REF:-10.0}/${AIB_CARRY_SPAN:-30.0}, launch ceiling ${AIB_LAUNCH_HI_W:-0.0}, grounder floor ${AIB_LAUNCH_LO_W:-0.0}, da L1 ${AIB_DA_L1_W:-0.0}, wait torso L1 $AIB_WAIT_TORSO_W, follow-through ${FT} s / episode ${EPS} s, fall after hit $AIB_FALL_AFTER_HIT_W"
LR=${LR0:-2026-11-01_11-42-57_human_pitched_h31_4}; LC=${LC0:-model_14496.pt}; PK=0
lastv() { grep -a "$1:" $2 | tail -1 | awk '{print $NF}'; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\) .*/\1/p"; }
# wait-phase reference (14, 20:5x): 14496 + percept_filter at 1.0x with the wait-window jitter (diag da9bf98 + wait terms)
MSG="wait-phase reference diag"; idle
RT=e10_14496_pf_w
[ -f logs/diag_bat_contact_$RT.json ] || timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $B/$LR/$LC --percept_filter \
  --envs 1024 --seed 0 --tag $RT --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height \
  --clips $C4 --physics_dt 0.00125 --contact_log > $O/diag_$RT.txt 2>&1
REF=$($P logs/e10_row.py logs/diag_bat_contact_$RT.json); WDREF=$(fld wait_da "$REF"); WTREF=$(fld wait_torso_hp "$REF")
[ -n "$WDREF" ] && [ -n "$WTREF" ] || { MSG="reference diag failed: $(grep -A14 Traceback $O/diag_$RT.txt | tail -1)"; exit 1; }
echo "H32_REF $(date +%H:%M) $RT: $REF" > logs/H32_REF.txt
MSG="smoke done, waiting for logs/h32/GO"
until [ -f $O/GO ]; do [ -f $O/STOP ] && { MSG="stopped before segment 1"; exit 0; }; sleep 60; done
RC=${RC0:-$(tr -d '\r\n' < logs/recov6/recover_ckpt.txt)}  # 14, 11-02 00:2x: R6 model_2792 adopted for the seg-2+ joint rows
for k in $(seq ${K0:-1} $NSEG); do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  while [ $k -gt 1 ] && [ -f $O/HOLD ]; do MSG="on HOLD before segment $k"; sleep 60; done
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  MSG="segment $k: waiting for logs/FILM8_END.txt newer than FILM8_GO"
  until [ -f logs/FILM8_GO.txt ] && [ -f logs/FILM8_END.txt ] && [ logs/FILM8_END.txt -nt logs/FILM8_GO.txt ]; do sleep 60; done
  if [ $k -ge 4 ] && ! grep -q '^H32S4B_SMOKE_END.* ok' logs/H32S4B_SMOKE_END.txt 2>/dev/null; then
    MSG="segment-$k reward smoke"; idle
    AIB_FT_LOST_IS_FALL=0 AIB_FT_LOST_ENDS=0 AIB_FOLLOW_THROUGH_S=$FT AIB_EPISODE_S=$EPS AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 20 --run_name h32s4b_smoke --resume True --load_run $LR --checkpoint $LC > $O/smoke_s4b.txt 2>&1
    if grep -q Traceback $O/smoke_s4b.txt; then SM="CRASH $(grep -A14 Traceback $O/smoke_s4b.txt | tail -1)"
    elif ! grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/smoke_s4b.txt; then SM="no AIB_PHYSICS_DT line"
    elif ! grep -q "bat/outfield" $O/smoke_s4b.txt; then SM="no bat/outfield stat"
    elif ! grep -q "mimic/wait_torso_hp" $O/smoke_s4b.txt; then SM="no mimic/wait_torso_hp stat"
    elif FH=$(lastv bat/fall_rate $O/smoke_s4b.txt); HH=$(lastv bat/hit_rate $O/smoke_s4b.txt); ! awk -v f="$FH" -v h="$HH" 'BEGIN{d = f - h; if (d < 0) d = -d; exit !(f != "" && h != "" && d > 0.05)}'; then SM="fall_rate $FH not apart from hit_rate $HH"
    else SM="ok: fall_rate $FH vs hit_rate $HH; $($P $SC/h31_curve.py $O/smoke_s4b.txt 10) ; $(grep -o 'Iteration time: [0-9.]*s' $O/smoke_s4b.txt | tail -1)"; fi
    echo "H32S4B_SMOKE_END $(date +%H:%M) $SM ; wait torso $(grep -a mimic/wait_torso_hp $O/smoke_s4b.txt | tail -1 | tr -s ' ') ($REW)" > logs/H32S4B_SMOKE_END.txt
    case "$SM" in ok*) ;; *) MSG="segment-$k smoke failed: $SM"; exit 1;; esac
  fi
  MSG="segment $k"; idle
  RN=human_pitched_h32_${k}b
  AIB_FT_LOST_IS_FALL=0 AIB_FT_LOST_ENDS=0 AIB_FOLLOW_THROUGH_S=$FT AIB_EPISODE_S=$EPS AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
    --resume True --load_run $LR --checkpoint $LC > $O/train_${k}b.txt 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: segment $k trainer private ${G} GB > 30, killed"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $O/train_${k}b.txt && fail $O/train_${k}b.txt train_${k}b
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/train_${k}b.txt || { MSG="segment $k: no AIB_PHYSICS_DT line"; exit 1; }
  D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="segment $k: no checkpoint"; exit 1; }
  T=h32b_$I; MSG="segment $k eval"
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log > $O/diag_$T.txt 2>&1
  if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
    ER=$($P logs/e10_row.py logs/diag_bat_contact_$T.json)
    ROW="+$((k * 500)) model_$I (1.0x, 800 Hz, 4 clips, no saim): $ER; pop-ups $($P logs/popup_share.py logs/diag_bat_contact_$T.json); $($P $SC/h31_carry.py logs/diag_bat_contact_$T.json 2>&1 | tail -1)"
  else ER=""; ROW="+$((k * 500)) model_$I: diag crash or no json"; fi
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s6b_$I.csv > $O/joint_s6b_$I.txt 2>&1
  if [ -f $O/joint_s6b_$I.csv ] && ! grep -q Traceback $O/joint_s6b_$I.txt; then
    ROW="$ROW ; joint s6 ($(basename $(dirname $RC)) $(basename $RC .pt)):$($P logs/film_pick.py $O/joint_s6b_$I.csv $O/film_pick_scratch.txt | head -1)"
  else ROW="$ROW ; joint s6 crashed"; fi
  ROW="$ROW ; $($P $SC/h31_curve.py $O/train_${k}b.txt 100) ; $REW ; $(grep -o 'Iteration time: [0-9.]*s' $O/train_${k}b.txt | tail -1), private peak ${PK} GB"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H32_$((k * 500))B_END $(date +%H:%M) $ROW" > logs/H32_$((k * 500))B_END.txt
  ROWS="$ROWS $ROW ||"
  if [ $k -le 2 ]; then  # segment-1 / -2 rules (wait-phase terms: not above the 14496 reference + 5 %)
    HT=$(fld hit "$ER"); EX=$(fld exit "$ER"); DA=$(fld da "$ER"); WD=$(fld wait_da "$ER"); WT=$(fld wait_torso_hp "$ER")
    if [ $k -eq 1 ]; then EMIN=$(awk -v e="$EXREF" 'BEGIN{print e - 1}'); DMAX=0.57; else EMIN=$EXREF; DMAX=0.50; fi
    awk -v h="$HT" -v e="$EX" -v a="$DA" -v wd="$WD" -v wt="$WT" -v em="$EMIN" -v dm="$DMAX" -v rd="$WDREF" -v rt="$WTREF" -v k=$k \
      'BEGIN{n = "^[0-9.]+$"; exit !(h ~ n && e ~ n && a ~ n && wd ~ n && wt ~ n && (k > 1 || h + 0 >= 0.40) && e + 0 >= em && a + 0 < dm && wd + 0 <= 1.05 * rd && wt + 0 <= 1.05 * rt)}' \
      || { MSG="segment-$k rule FAILED (hit '$HT'$([ $k -eq 1 ] && echo ' >= 0.40'), exit '$EX' >= $EMIN, da '$DA' < $DMAX, wait da '$WD' <= 1.05 x $WDREF, wait torso '$WT' <= 1.05 x $WTREF) -> stopped"; exit 0; }
  else  # segment 3+ rules (14, 11-02 01:1x)
    HT=$(fld hit "$ER"); EX=$(fld exit "$ER"); DA=$(fld da "$ER"); WD=$(fld wait_da "$ER"); WT=$(fld wait_torso_hp "$ER"); FE=$(fld "fell by end" "$ER")
    EMIN=$(awk -v e="$EX2" 'BEGIN{print e - 1}'); CM=$(echo "$ROW" | sed -n 's/.*carry fair med \([0-9.]*\) m.*/\1/p'); PU=$(echo "$ROW" | sed -n 's/.*pop-ups \([0-9.]*\) .*/\1/p')
    LZ=$($P -c "import json, numpy as np; P = {k: np.array(v, float) for k, v in json.load(open('logs/diag_bat_contact_$T.json'))['per_pitch'].items()}; z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10) & (P['hit'] > 0) & (np.abs(P['spray_contact']) <= 45) & np.isfinite(P['launch_contact']); print(f'{np.median(P[\"launch_contact\"][z]):.1f}')")
    awk -v h="$HT" -v e="$EX" -v a="$DA" -v wd="$WD" -v wt="$WT" -v f="$FE" -v em="$EMIN" -v rd="$WDREF" -v rt="$WTREF"       'BEGIN{n = "^[0-9.]+$"; exit !(h ~ n && e ~ n && a ~ n && wd ~ n && wt ~ n && f ~ n && h + 0 >= 0.75 && e + 0 >= em && a + 0 < 0.50 && wd + 0 <= 1.05 * rd && wt + 0 <= 1.05 * rt && f + 0 <= 0.60)}'       || { MSG="segment-$k rule FAILED (hit '$HT' >= 0.75, exit '$EX' >= $EMIN, da '$DA' < 0.50, wait da '$WD' <= 1.05 x $WDREF, wait torso '$WT' <= 1.05 x $WTREF, fell by end '$FE' <= 0.60) -> stopped"; exit 0; }
    [ $k -ge 4 ] && { awk -v l="$LZ" -v p="$PU" 'BEGIN{n = "^-?[0-9.]+$"; exit !(l ~ n && p ~ n && l + 0 <= 30 && p + 0 <= 0.20)}'       || { MSG="segment-$k launch rule FAILED (in-zone fair launch median '$LZ' <= 30, pop-ups '$PU' <= 0.20: the ceiling did not hold) -> stopped"; exit 0; }; }
    [ $k -ge 4 ] && { awk -v e="$EX" -v c="$CM" 'BEGIN{n = "^[0-9.]+$"; exit !(e ~ n && c ~ n && e + 0 >= 27.5 && c + 0 >= 30)}'       || { MSG="segment-$k swing rule FAILED (exit '$EX' >= 27.5, fair carry median '$CM' >= 30: standing weights too high) -> stopped"; exit 0; }; }
  fi
  LR=$(basename $D); LC=model_$I.pt
done
MSG="PASS: all segment-$NSEG gates (+$((NSEG * 500)))"
