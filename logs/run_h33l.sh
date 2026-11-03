#!/bin/bash
# run_h33l.sh (14, 11-03 16:5x): H33L-probe. The tempo sweep showed untrained policies cannot follow a faster clip (ss falls,
# contact late) and the top joints sit at 0.93-0.99 torque_sat at every tempo. One trained probe decides whether "a trained
# policy could follow a faster clip" is open: 500 its from h33g_1/model_13498 (hit 0.852 at x1.15 untrained) with the reference
# played x1.15 (AIB_TEMPO_SCRIPT), frame = H33H minus lift (AIB_LIFT_W=0 = the H33F frame: ramp hit-only x f, floor on all hits,
# exit hinge, carry, lost = miss), f = the H33H linear hinge (simplest: same env as H33H), WDA / WWT as H33H, timing noise 0.
# Falsifier +250: train hit < 0.60 -> kill. Verdict +500 (diag j0, 1024 envs, --tempo_script 1.15, pose vs the sped-up clip):
# ss_contact >= 22.5 AND pose <= 0.215 AND hit >= 0.85 AND fair launch median >= +5 -> "CAN follow 1.15" (H33L becomes a line);
# else closed. Top-3 torque_sat reported. Runs after FILM15_END. Markers H33L_250_END / H33L_500_END / H33L_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_DA_L1_W AIB_WAIT_TORSO_W AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT
export AIB_LIFT_W=0 AIB_LIFT_REF=5 AIB_LIFT_SPAN=15 AIB_LAUNCH_LO_ALL=1 AIB_FORM_S=0.2162 AIB_FORM_HINGE_HI=0.245 AIB_FORM_HINGE_W=0.045 AIB_LOST_IS_MISS=1 AIB_SS_CA_HIT_ONLY=1 AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=14.7 AIB_SS_CA_HI=24.7 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=15.5 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_OUTFIELD_W=10 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=35 AIB_LAUNCH_HI_SPAN=15 \
  AIB_LAUNCH_LO_W=10 AIB_LAUNCH_LO_REF=5 AIB_LAUNCH_LO_SPAN=15 AIB_FALL_AFTER_HIT_W=10.0
WDA=0.0259; WWT=0.00360; TS=1.15
TASK=AIB-PitchedBatting-Human33-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
O=logs/h33; M=logs/H33L_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33L_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
killrn() { powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"; }
fv() { echo "$2" | sed -n "s/.*$1 \([-+0-9.]*\) .*/\1/p" | head -1; }
MSG="waiting for FILM15_END"; until [ -f logs/FILM15_END.txt ] && [ logs/FILM15_END.txt -nt logs/run_h33l.sh ]; do sleep 60; done
[ -f $O/STOP ] && { MSG="stopped by STOP before H33L"; exit 0; }
MSG="H33L probe training"; idle
RN=human_pitched_h33l_1; TF=$O/train_l1.txt; I0=13498; PK=0; EF=0
AIB_TEMPO_SCRIPT=$TS AIB_DA_L1_W=$WDA AIB_WAIT_TORSO_W=$WWT AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
  --resume True --load_run 2026-11-03_06-50-43_human_pitched_h33g_1 --checkpoint model_13498.pt > $TF 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  if [ $EF -eq 0 ] && grep -aq "Learning iteration $((I0 + 250))/" $TF; then
    EF=1; CV=$($P logs/h33_curve.py $TF 50 $((I0 + 250))); HV=$(echo "$CV" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p')
    echo "$(date +%H:%M) H33L +250 check: $CV" >> $O/progress.txt; echo "H33L_250_END $(date +%H:%M) $CV" > logs/H33L_250_END.txt
    if ! awk -v h="$HV" 'BEGIN{exit !(h ~ /^[0-9.]+$/ && h + 0 >= 0.60)}'; then
      killrn $RN; MSG="H33L FALSIFIER at +250 (train hit '$HV' >= 0.60): $CV -> trainer killed; trained-follow question closed"; exit 0
    fi
  fi
  G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then killrn $RN; MSG="RAM watchdog: H33L trainer private ${G} GB > 30, killed"; exit 1; fi
  sleep 30
done
grep -q Traceback $TF && { MSG="H33L training crash $(grep -A14 Traceback $TF | tail -1)"; exit 1; }
grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $TF || { MSG="H33L: no AIB_PHYSICS_DT line"; exit 1; }
D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="H33L: no checkpoint"; exit 1; }
MSG="H33L +500 diag"; idle
T=h33l_${I}_t115
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
  --physics_dt 0.00125 --contact_log --form_log --sat_log --tempo_script $TS > $O/diag_$T.txt 2>&1
if grep -q Traceback $O/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ]; then MSG="H33L diag crash or no json"; exit 1; fi
TR=$($P logs/tempo_row.py $T); SAT=$(grep "torque_sat" $O/diag_$T.txt | tail -1 | sed 's/.*top 10): //' | cut -c1-120)
SS=$(fv ss_contact "$TR"); PO=$(fv pose "$TR"); HT=$(fv hit "$TR"); LA=$(fv "fair launch" "$TR")
ROW="+500 model_$I @tempo $TS (1024 envs, j0): $TR | torque_sat top: $SAT | sweep untrained 13498 x1.15: ss 21.0 hit 0.852 launch +4.0 pose 0.215 | train $($P logs/h33_curve.py $TF 50 $I) | private peak ${PK} GB"
echo "H33L_500_END $(date +%H:%M) $ROW" > logs/H33L_500_END.txt; echo "$(date +%H:%M) H33L $ROW" >> $O/progress.txt
if awk -v s="$SS" -v p="$PO" -v h="$HT" -v l="$LA" 'BEGIN{n = "^[-+]?[0-9.]+$"; exit !(s ~ n && p ~ n && h ~ n && l ~ n && s + 0 >= 22.5 && p + 0 <= 0.215 && h + 0 >= 0.85 && l + 0 >= 5)}'; then
  MSG="CAN follow 1.15: a trained policy follows the faster clip (ss '$SS' >= 22.5, pose '$PO' <= 0.215, hit '$HT' >= 0.85, fair launch '$LA' >= +5) | $ROW"
else
  MSG="CLOSED: trained probe does not follow 1.15 (ss '$SS' >= 22.5, pose '$PO' <= 0.215, hit '$HT' >= 0.85, fair launch '$LA' >= +5) | $ROW"
fi
