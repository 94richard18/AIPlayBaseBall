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
# run_h33r.sh (14, 11-02 21:xx, retry under today's autonomy): H33 seg 1 (12999) failed because the ss ramp paid misses (hit 0.47
# -> 0.03, pose 0.25, carry 1.5 m) -> AIB_SS_CA_HIT_ONLY=1 (ramp x hit, still x f), all else unchanged; again from 12500; NSEG 3
# (the retry = segment 2 of the 4 approved). Early falsifier in every segment: at +250 its train bat/hit_rate < 0.30 or
# mimic/form_pose_swing > 0.22 -> trainer killed, stop. Starts only after logs/FILM11_END.txt exists (+ idle). Names: run
# human_pitched_h33r_<k>, train_r<k>.txt, smoke_r.txt, diag tag h33r_<I>; markers logs/H33R_SMOKE_END.txt, logs/H33R_<its>_END.txt,
# final logs/H33R_END.txt.
# run_h33d.sh (14, 11-02 23:2x, approved under the 11-06 extension): probes P1 / P2 showed da L1 0.2 costing ~27 per episode vs
# < 1 for all batting terms (the H33R drop) -> H33D = H33R frame with the smoothness penalties scaled to the batting terms: seg 1
# da L1 0 / wait torso 0; seg 2 weights from seg 1's end-of-segment term EMAs: w_da = min(0.02, 0.25 B / da_unit), w_wt =
# min(0.003, 0.02 B / wt_unit), B = positive batting terms earned per episode after f = rt_ramp + rt_hit - rt_launch_pen (rt_hit
# already holds exit hinge + carry + outfield + spray, net of the launch pens); seg 3-4: the same rule (caps 0.05 / 0.0075) re-applied
# only when the segment's diag da > 0.60, else the weights are kept. da gate: seg 1 reported only; seg 2-3 a failing da raises the
# weights (no stop); seg 4 (last) da <= 0.60 hard. +250 falsifier: train hit < 0.20 or pose_swing > 0.22. NSEG 4; HOLD before seg 4
# for film12 (14 creates logs/h33/HOLD after H33D_1000_END). Names human_pitched_h33d_<k>, train_d<k>.txt, diag tag h33d_<I>,
# markers logs/H33D_SMOKE_END.txt, logs/H33D_<its>_END.txt, final logs/H33D_END.txt.
# run_h33e.sh (14, 11-03 00:xx): H33D seg 1 dropped the ball while the miss penalty vanished (-3.1 -> -0.02 per episode,
# episodes 138 -> 125 steps, no falls) -> pre-hit `lost` dodges w_miss. H33E = H33D + AIB_LOST_IS_MISS=1 (a pre-hit lost charged
# w_miss), launched by run_h33e_probe.sh only if its 150-it probe shows the termination swap. Every row carries the episode-end
# shares (bat/end_hit / fall / passed / lost / timeout / other). Names human_pitched_h33e_<k>, train_e<k>.txt, diag tag h33e_<I>,
# markers logs/H33F_SMOKE_END.txt, logs/H33E_<its>_END.txt, final logs/H33E_END.txt.
# run_h33f.sh (14, 11-03 04:3x): H33E seg 2 (13498, grounder floor 15/5/15 fair-only) tripped FORM (pose 0.212) while fair fell
# 0.78 -> 0.55 and the in-zone launch median went +0.3 -> -0.9: the policy fouled the low balls off to dodge the floor. H33F from
# 12999: (a) floor 15/5/15 on every hit, fair or foul (AIB_LAUNCH_LO_ALL=1), (b) form_s 0.2162 (f 0.5 at 0.18 rad), (c) gate fair
# >= previous - 0.05. da L1 0.0234 / wait torso 0.00339 (H33E seg 2's NEXT rule), re-ruled (caps 0.05 / 0.0075) only when diag
# da > 0.60. Gates as H33E seg 2: pose <= 0.20 / key <= 0.11 hard, hit >= prev - 0.02, exit >= prev - 0.3, in-zone fair launch
# median >= prev + 3, pop-ups <= 0.20, da soft. +250 falsifier: train hit < 0.60 or pose_swing > 0.21. NSEG 2 after a 20-it smoke
# (new rt_launch_lo term). Starts after film12 (idle() waits on FILM12_END). Rows add fair / foul share of hits, in-zone launch
# median and the per-term sums. Names human_pitched_h33f_<k>, train_f<k>.txt, tag h33f_<I>, markers H33F_SMOKE_END / H33F_<its>_END / H33F_END.
# run_h33f2.sh (14, 11-03 05:4x): H33F seg 1 hit its +250 falsifier at train pose 0.213 (12999 itself trains at 0.20-0.21) with the
# floor now paid (-2.49/ep) and train fair carry 17.9 m. k = 0: diag + form + joint s6 on h33f_1/model_13200 -> H33F_13200_END;
# relaunch only if diag pose <= 0.20, key <= 0.11 and hit >= 0.887, then 2 segments from 13200 in the same frame with the
# falsifier at train pose > 0.225 or hit < 0.60 (gates' previous = the 13200 eval). Names human_pitched_h33f2_<k>, train_g<k>.txt,
# markers H33F2_500/1000_END, final H33F2_END.
# run_h33g.sh (14, 11-03 06:0x): the H33F frame with the floor at w 10 (all hits) + lift bonus 5 x clamp((launch - 5)/15, 0, 1)
# on every hit inside the x f hit rewards (lift pays only through good form; f on the floor itself would reward bad form).
# Start ckpt and the gates' previous values from run_h33g_chain.sh (G_LR G_LC G_H G_E G_L G_F). 2 segments, gates as H33F,
# falsifier train pose > 0.225 or hit < 0.60. Names human_pitched_h33g_<k>, train_h<k>.txt, tag h33g_<I>, markers H33G_500/1000_END, H33G_END.
# run_h33g2.sh (14, 11-03 09:0x): H33G segment 2 from h33g_1/model_13498 after film13 (run_h33g2_chain.sh), same frame,
# da 0.0259 / wt 0.00360 (seg 1's NEXT rule). Gates (14, form line 0.20 -> 0.215 pending the user's film13 look): pose <= 0.215
# and key <= 0.113 hard, hit >= 0.94, fair >= 0.79, exit >= 22.9, in-zone fair launch median >= 17, pop-ups <= 0.20; falsifier
# train pose > 0.235 or hit < 0.60. Marker H33G_1000_END, final H33G2_END.
# run_h33h.sh (14, 11-03 11:5x): H33G seg 2 (13997) carried 35.3 m but form drifted 0.215 -> 0.223 (the Gaussian is past its
# inflection, d f / d pose ~ -3.3 / rad, form pressure ~ 0.22 / ep vs batting ~ 30 / ep). H33H = the H33G frame with f a linear
# hinge on the same pose input (swing-window mean unweighted joint RMS, clip start -> contact): f = clamp((0.245 - pose) / 0.045,
# 0, 1) (AIB_FORM_HINGE_HI / _W), everywhere f is used (ramp, hit rewards, lift). From h33g_2/model_13997, 1 segment of +1000,
# after FILM14_END (run_h33h_chain.sh). Falsifiers: +250 train pose > 0.235 or hit < 0.60; +500 train pose not below 0.22.
# Gates: pose <= 0.215 AND key <= 0.113 (hard), hit >= 0.90, fair >= 0.80, fair carry median >= 28 m, pop-ups <= 0.20.
# Markers H33H_250_END / H33H_500_END / H33H_1000_END, final H33H_END.
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
M=logs/H33H_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H33H_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
lastv() { grep -a "$1:" $2 | tail -1 | awk '{print $NF}'; }
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\) .*/\1/p"; }
REW="H33H form HINGE f = clamp((0.245 - pose)/0.045, 0, 1) (1 at 0.200, 0 at 0.245) replacing the Gaussian; H33G lift 5 x clamp((launch - 5)/15) x f on all hits, floor 10/5/15 on ALL hits (fair or foul), f 0.5 at 0.18 (s 0.2162), pre-hit lost = miss, ramp HIT ONLY, form x exp(-(pose/0.2402)^2) on ramp ${AIB_SS_CA_LO}-${AIB_SS_CA_HI} win ${AIB_SS_CA_WIN} / exit hinge ${AIB_EXIT_HINGE_W} ref ${AIB_EXIT_HINGE_REF} cap ${AIB_EXIT_HINGE_CAP} / carry ${AIB_CARRY_REF}/${AIB_CARRY_SPAN} / outfield ${AIB_OUTFIELD_W}; additive: launch ceiling ${AIB_LAUNCH_HI_W}/${AIB_LAUNCH_HI_REF}/${AIB_LAUNCH_HI_SPAN}, grounder floor ${AIB_LAUNCH_LO_W}/${AIB_LAUNCH_LO_REF}/${AIB_LAUNCH_LO_SPAN}, da L1 / wait torso L1 per segment (rule)"
LR=2026-11-03_09-46-59_human_pitched_h33g_2; LC=model_13997.pt; PK=0
for k in 1; do
  TF=$O/train_hh$k.txt
  MK=H33H_1000_END
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  while [ $k -gt 1 ] && [ -f $O/HOLD ]; do MSG="on HOLD before segment $k"; sleep 60; done
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  MSG="segment $k"; idle
  RN=human_pitched_h33h_$k
  AIB_DA_L1_W=$WDA AIB_WAIT_TORSO_W=$WWT AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 1000 --run_name $RN \
    --resume True --load_run $LR --checkpoint $LC > $TF 2>&1 &
  TP=$!; I0=${LC#model_}; I0=${I0%.pt}; EF=0; EF5=0
  while kill -0 $TP 2>/dev/null; do
    if [ $EF -eq 0 ] && grep -aq "Learning iteration $((I0 + 250))/" $TF; then  # early falsifier (14): +250 its
      EF=1; CV=$($P logs/h33_curve.py $TF 50 $((I0 + 250)))
      HV=$(echo "$CV" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p'); PV=$(echo "$CV" | sed -n 's/.* pose \([0-9.]*\) .*/\1/p')
      echo "$(date +%H:%M) segment $k +250 check: $CV" >> $O/progress.txt; echo "H33H_250_END $(date +%H:%M) $CV" > logs/H33H_250_END.txt
      if ! awk -v h="$HV" -v p="$PV" 'BEGIN{n = "^[0-9.]+$"; exit !(h ~ n && p ~ n && h + 0 >= 0.60 && p + 0 <= 0.235)}'; then
        powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
        MSG="segment-$k EARLY FALSIFIER at +250 (train hit '$HV' >= 0.60, pose_swing '$PV' <= 0.235): $CV -> trainer killed, stopped"; exit 0
      fi
    fi
    if [ $EF5 -eq 0 ] && grep -aq "Learning iteration $((I0 + 500))/" $TF; then  # second falsifier (14): +500 its, form must move
      EF5=1; CV=$($P logs/h33_curve.py $TF 50 $((I0 + 500)))
      PV=$(echo "$CV" | sed -n 's/.* pose \([0-9.]*\) .*//p')
      echo "$(date +%H:%M) segment $k +500 check: $CV" >> $O/progress.txt; echo "H33H_500_END $(date +%H:%M) $CV" > logs/H33H_500_END.txt
      if ! awk -v p="$PV" 'BEGIN{exit !(p ~ /^[0-9.]+$/ && p + 0 < 0.22)}'; then
        powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
        MSG="segment-$k FALSIFIER at +500 (train pose_swing '$PV' < 0.22: form did not move, the lever is wrong): $CV -> trainer killed, stopped"; exit 0
      fi
    fi
    G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: segment $k trainer private ${G} GB > 30, killed"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $TF && fail $TF train_f$k
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="segment $k: no AIB_PHYSICS_DT line"; exit 1; }
  D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="segment $k: no checkpoint"; exit 1; }
  T=h33h_$I; MSG="segment $k eval"
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ] && { MSG="segment $k: diag crash or no json $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  ER=$($P logs/e10_row.py logs/diag_bat_contact_$T.json)
  FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I $O/form_$I.csv | tail -1)
  PU=$($P logs/popup_share.py logs/diag_bat_contact_$T.json)
  BN=$($P logs/h33_bins.py $T 2>&1 | tail -7 | tr -s ' ' | tr '\n' '/')
  ROW="+1000 model_$I (1.0x, 800 Hz, 4 clips, real height, no saim): $ER; pop-ups $PU; FORM $FR; bins $BN"
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s6_$I.csv > $O/joint_s6_$I.txt 2>&1
  if [ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt; then
    ROW="$ROW ; joint s6 (R6b 3091):$($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1)"
  else ROW="$ROW ; joint s6 crashed"; fi
  ROW="$ROW ; episode ends $(for x in hit fall passed lost timeout other; do echo -n "$x $(grep -a "bat/end_$x:" $TF | tail -1 | awk '{print $NF}') "; done) ; weights this segment da L1 $WDA wait torso $WWT ; $($P logs/h33_curve.py $TF 100) ; $REW ; $(grep -o 'Iteration time: [0-9.]*s' $TF | tail -1), private peak ${PK} GB"
  LZ=$($P -c "import json, numpy as np; P = {k: np.array(v, float) for k, v in json.load(open('logs/diag_bat_contact_$T.json'))['per_pitch'].items()}; z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10) & (P['hit'] > 0) & (np.abs(P['spray_contact']) <= 45) & np.isfinite(P['launch_contact']); print(f'{np.median(P[\"launch_contact\"][z]):.1f}')")
  FA=$(fld fair "$ER")
  ROW="$ROW ; H33H fair $FA foul share of hits $(awk -v f="$FA" 'BEGIN{printf "%.2f", 1 - f}') in-zone fair launch median $LZ ; term sums $(for x in ramp hit lift launch_lo launch_pen carry exit_hinge outfield miss da_l1 wait_torso track; do echo -n "$x $(lastv bat/rt_$x $TF) "; done)"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
    # next segment's smoothness weights (14's rule, H33E seg 2+): only if this segment's diag da > 0.60
  if [ $k -lt $NSEG ] && awk -v a="$(fld da "$ER")" 'BEGIN{exit !(a + 0 > 0.60)}'; then
    CD=0.05; CW=0.0075
    read WDA WWT WNOTE <<< "$($P -c "
import re
t = open('$TF', encoding='utf-8', errors='replace').read()
g = lambda k: float(re.findall(r'bat/' + k + r':\s*([-0-9.e]+)', t)[-1])
B = g('rt_ramp') + g('rt_hit') - g('rt_launch_pen'); du, wu = g('rt_da_unit'), g('rt_wt_unit')
wd = min($CD, 0.25 * max(B, 0.0) / max(du, 1e-6)); ww = min($CW, 0.02 * max(B, 0.0) / max(wu, 1e-6))
print(f'{wd:.4f} {ww:.5f} B={B:.3f},da_unit={du:.2f},wt_unit={wu:.2f}')")"
    awk -v d="$WDA" -v w="$WWT" 'BEGIN{exit !(d ~ /^[0-9.]+$/ && w ~ /^[0-9.]+$/)}' || { MSG="segment $k: weight rule failed ('$WDA' '$WWT')"; exit 1; }
    ROW="$ROW ; NEXT weights (rule, caps $CD / $CW): da L1 $WDA wait torso $WWT from $WNOTE"
  fi
  echo "$MK $(date +%H:%M) $ROW" > logs/$MK.txt
  ROWS="$ROWS $ROW ||"
  HT=$(fld hit "$ER"); EX=$(fld exit "$ER"); DA=$(fld da "$ER")
  PO=$(echo "$FR" | sed -n 's/.*| pose \([0-9.]*\)\/.*/\1/p'); KE=$(echo "$FR" | sed -n 's/.*| key \([0-9.]*\)\/.*/\1/p')
  FC=$(echo "$FR" | sed -n 's/.*| carry \([0-9.na]*\) .*/\1/p')
  PUV=$(echo "$PU" | awk '{print $1}')
  awk -v p="$PO" -v q="$KE" 'BEGIN{n = "^[0-9.]+$"; exit !(p ~ n && q ~ n && p + 0 <= 0.215 && q + 0 <= 0.113)}' \
    || { MSG="segment-$k FORM gate FAILED (pose_err '$PO' <= 0.215, key_err '$KE' <= 0.113; fair carry '$FC') -> stopped"; exit 0; }
  awk -v h="$HT" -v fa="$FA" -v c="$FC" -v u="$PUV" 'BEGIN{n = "^[0-9.]+$"; exit !(h ~ n && fa ~ n && c ~ n && u ~ n && h + 0 >= 0.90 && fa + 0 >= 0.80 && c + 0 >= 28 && u + 0 <= 0.20)}' \n    || { MSG="segment-$k rule FAILED (hit '$HT' >= 0.90, fair '$FA' >= 0.80, fair carry median '$FC' >= 28, pop-ups '$PUV' <= 0.20; exit '$EX' da '$DA' in-zone launch '$LZ' reported) -> stopped"; exit 0; }
  HPREV=$HT; EPREV=$EX; LPREV=$LZ; FPREV=$FA
  LR=$(basename $D); LC=model_$I.pt
done
MSG="PASS: all H33H gates (+1000)"
