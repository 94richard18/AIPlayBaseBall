#!/bin/bash
# run_h33j_t.sh (14, 11-03 14:3x, rulings b + c): the H33J baseline row showed the diag at jitter 0.03 RAISING launch (+26.5,
# pop-ups 0.33) while the tables at jitter 0.03 lower it, so the diag launch gates were dropped. run_h33j.sh stops after part 1
# (logs/h33/STOP, tagged "h33j-part2-skip"). This script judges the +500 model on the TABLE metric (the adoption metric):
#   1. jitter-0 diag (1024 envs, form) + the jitter-0.03 game-scene table set (3 tables) + contact-quality bins of those tables
#      (all-fair launch median, OF-carry share, grounder share) -> marker H33J_T500_END.
#   2. Part 2 (+500 more, same frame, U(+-0.03) training noise) only if j0.03 column >= 15.8, diag pose <= 0.215, key <= 0.113
#      and diag hit >= 0.85; else H33J ends ("jitter training did not move the fragile column at +500").
#   3. +1000: pose / key hard (jitter-0 diag), then both table columns vs the bars 16.93 / 16.76 with OF >= 3% each.
# Part 2 falsifier at +250 as part 1 (train hit < 0.50 or la < -3). Markers H33J_T500_END, H33J_1000_END, H33J_END;
# chain marker logs/H33JT_CHAIN_END.txt (the tempo sweep waits on it).
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
trap 'echo "H33J_END $(date +%H:%M) $MSG |$ROWS" > $M; echo "H33JT_CHAIN_END $(date +%H:%M) $MSG" > logs/H33JT_CHAIN_END.txt; cat $M' EXIT
rm -f logs/H33JT_CHAIN_END.txt
killrn() { powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }
diag() {  # $1 ckpt, $2 tag, $3 envs, $4 timing noise
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --percept_filter --envs $3 --seed 0 --tag $2 \
    --timing_noise $4 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log > $O/diag_$2.txt 2>&1
  ! grep -q Traceback $O/diag_$2.txt && [ -f logs/diag_bat_contact_$2.json ]
}
form() { $P logs/h32_form.py logs/diag_bat_contact_$1.json $2 $O/form_$2.csv | tail -1; }
fv() { echo "$2" | sed -n "s/.*| $1 \([-+0-9.na]*\)[ \/].*/\1/p"; }  # one field of the h32_form row
tq() { $P -c "
import json, glob, numpy as np
S = [s for f in sorted(glob.glob('$1')) for s in json.load(open(f))['swings']]
c = np.array([s['carry_m'] for s in S]); l = np.array([s['launch_deg'] for s in S]); sp = np.array([s['spray_deg'] for s in S]); fa = np.abs(sp) <= 45
print(f'contacts/table {len(S) / max(len(glob.glob(\"$1\")), 1):.0f} fair {fa.mean():.2f} all-fair launch med {np.median(l[fa]):+.1f} grounder(<5) {(l[fa] < 5).mean():.2f} fair carry med {np.median(c[fa]):.1f} OF-carry share {(fa & (c >= 38.9)).mean():.3f}')"; }
evalj0() {  # $1 dir, $2 iter -> ER FR PO KE HT (jitter-0 diag, 1024 envs)
  T0=h33j_$2; MSG="H33J jitter-0 diag $2"; idle
  diag $1/model_$2.pt $T0 1024 0 || { MSG="H33J jitter-0 diag $2 crash or no json"; exit 1; }
  ER=$($P logs/e10_row.py logs/diag_bat_contact_$T0.json); FR=$(form $T0 $2); PO=$(fv pose "$FR"); KE=$(fv key "$FR"); HT=$(fld hit "$ER")
}
MSG="waiting for H33J_CHAIN_END (part 1 + its +500 diag, then the STOP exit)"
until [ -f logs/H33J_CHAIN_END.txt ] && [ logs/H33J_CHAIN_END.txt -nt logs/run_h33j_t.sh ]; do sleep 60; done
D=$(ls -td $B/*human_pitched_h33j_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="no h33j_1 checkpoint"; exit 1; }
I0=13498; [ "$I" -ge $((I0 + 499)) ] || { MSG="h33j_1 ended at model_$I before +500 (part 1 falsifier or crash): $(cut -c1-300 logs/H33J_END.txt)"; exit 0; }
evalj0 $D $I
idle; MSG="H33J +500 j0.03 tables"; bash logs/run_h33_tables_j03.sh $D/model_$I.pt $I $T0 > logs/h33_tables_j03_$I.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_J03_${I}_END.txt); Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% .*/\1/p')
QB=$(tq "logs/h33/tables_game/t3_${I}_j0.03_?.json")
ROW="+500 model_$I | JITTER-0.03 TABLES $TM | contact quality j0.03: $QB | jitter-0 diag: $ER; FORM $FR | 13498 j0.03 ref: 14.61% (14.4-14.8), all-fair launch +10.6, OF-carry 0.096, grounder 0.43"
echo "H33J_T500_END $(date +%H:%M) $ROW" > logs/H33J_T500_END.txt; echo "$(date +%H:%M) $ROW" >> $O/progress.txt; ROWS=" $ROW ||"
awk -v w="$Wa" -v p="$PO" -v q="$KE" -v h="$HT" 'BEGIN{n = "^[0-9.]+$"; exit !(w ~ n && p ~ n && q ~ n && h ~ n && w + 0 >= 15.8 && p + 0 <= 0.215 && q + 0 <= 0.113 && h + 0 >= 0.85)}' \
  || { MSG="H33J ends at +500: jitter training did not move the fragile column (j0.03 '$Wa' >= 15.8, pose '$PO' <= 0.215, key '$KE' <= 0.113, hit '$HT' >= 0.85)"; exit 0; }
grep -q "h33j-part2-skip" $O/STOP 2>/dev/null && rm -f $O/STOP  # only the STOP 14's ruling b created
[ -f $O/STOP ] && { MSG="stopped by STOP before H33J part 2"; exit 0; }
LR=$(basename $D); LC=model_$I.pt; RN=human_pitched_h33j_2; TF=$O/train_j2.txt; MSG="H33J part 2"; idle
AIB_TIMING_NOISE=0.03 AIB_DA_L1_W=$WDA AIB_WAIT_TORSO_W=$WWT AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
  --resume True --load_run $LR --checkpoint $LC > $TF 2>&1 &
TP=$!; EF=0; PK=0
while kill -0 $TP 2>/dev/null; do
  if [ $EF -eq 0 ] && grep -aq "Learning iteration $((I + 250))/" $TF; then
    EF=1; CV=$($P logs/h33_curve.py $TF 50 $((I + 250)))
    HV=$(echo "$CV" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p'); LV=$(echo "$CV" | sed -n 's/.* la \([-0-9.]*\)$/\1/p')
    echo "$(date +%H:%M) H33J part 2 +250 check: $CV" >> $O/progress.txt
    if ! awk -v h="$HV" -v l="$LV" 'BEGIN{exit !(h ~ /^[0-9.]+$/ && l ~ /^-?[0-9.]+$/ && h + 0 >= 0.50 && l + 0 >= -3)}'; then
      killrn $RN; MSG="H33J part 2 FALSIFIER at +250 (train hit '$HV' >= 0.50, la '$LV' >= -3): $CV -> trainer killed, stopped"; exit 0
    fi
  fi
  G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then killrn $RN; MSG="RAM watchdog: H33J part 2 trainer private ${G} GB > 30, killed"; exit 1; fi
  sleep 30
done
grep -q Traceback $TF && fail $TF train_j2
grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="H33J part 2: no AIB_PHYSICS_DT line"; exit 1; }
D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="H33J part 2: no checkpoint"; exit 1; }
evalj0 $D $I
PU=$($P logs/popup_share.py logs/diag_bat_contact_$T0.json)
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_$I.csv > $O/joint_s6_$I.txt 2>&1
if [ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt; then JS=$($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1); else JS=crashed; fi
ROW="+1000 model_$I | jitter-0 diag: $ER; pop-ups $PU; FORM $FR | joint s6 (R6b): $JS | episode ends $(for x in hit fall passed lost timeout other; do echo -n "$x $(lastv bat/end_$x $TF) "; done)| $($P logs/h33_curve.py $TF 100) | private peak ${PK} GB"
echo "H33J_1000_END $(date +%H:%M) $ROW" > logs/H33J_1000_END.txt; echo "$(date +%H:%M) $ROW" >> $O/progress.txt; ROWS="$ROWS $ROW ||"
awk -v p="$PO" -v q="$KE" 'BEGIN{n = "^[0-9.]+$"; exit !(p ~ n && q ~ n && p + 0 <= 0.215 && q + 0 <= 0.113)}' \
  || { MSG="H33J +1000 FORM gate FAILED (pose '$PO' <= 0.215, key '$KE' <= 0.113) -> stopped"; exit 0; }
idle; MSG="H33J +1000 tables"; bash logs/run_h33_tables.sh $D/model_$I.pt $I $T0 > logs/h33_tables_$I.out 2>&1
TM=$(cut -c1-300 logs/H33_TABLES_${I}_END.txt)
Wa=$(echo "$TM" | sed -n 's/.*jitter 0.03: \([0-9.]*\)% \/ OF \([0-9.]*\)%.*/\1/p'); Oa=$(echo "$TM" | sed -n 's/.*jitter 0.03: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
Wb=$(echo "$TM" | sed -n 's/.*jitter 0: \([0-9.]*\)% .*/\1/p'); Ob=$(echo "$TM" | sed -n 's/.*jitter 0: [0-9.]*% \/ OF \([0-9.]*\)%.*/\1/p')
ROWS="$ROWS TABLES $TM | contact quality j0.03: $(tq "logs/h33/tables_game/t3_${I}_j0.03_?.json") ||"
awk -v a="$Wa" -v b="$Wb" -v oa="$Oa" -v ob="$Ob" 'BEGIN{n = "^[0-9.]+$"; exit !(a ~ n && b ~ n && oa ~ n && ob ~ n && a + 0 >= 16.93 && b + 0 >= 16.76 && oa + 0 >= 3 && ob + 0 >= 3)}' \
  || { MSG="H33J +1000 tables below the bars (j0.03 '$Wa' >= 16.93, j0 '$Wb' >= 16.76, OF '$Oa' / '$Ob' >= 3)"; exit 0; }
MSG="PASS: H33J +1000 form + both table bars ($TM)"
