#!/bin/bash
# run_h34q.sh (14, 11-05 20:2x GO, step 2 option 6): no training. After H34P_ADOPT_END: inference-only deeper action EMA on 16300 +
# AIB_PIN_UNTIL_RELEASE=0.00, alpha 0.40 and 0.35 (trained with 0.5), deterministic --jit_groups j0 1024 diags. Gates vs jg_16300 as H34P
# (logs/h34p_gate.py) + the gain (logs/h34q_gain.py: app torso / legs / feet each <= -10%, else "no gain"). Marker logs/H34Q_END.txt:
# the 16300 + pin0 row (h34p_000) + one row per alpha. No launch from 02:35 (film21 03:25).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W AIB_WAIT_WRIST_W AIB_WAIT_WRIST_RAMP_STEPS AIB_ACT_EMA
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_PRESTART_PIN=1 AIB_PIN_UNTIL_RELEASE=0.00
B=logs/rsl_rl/aib1_tee_batting_mimic; O=logs/h33
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
BC=$(ls -td $B/*_human_pitched_h32e | head -1)/model_16300.pt
M=logs/H34Q_END.txt; rm -f $M; MSG="aborted"
pose() { echo "$1" | sed -n 's/.*| pose \([0-9.]*\)\/.*/\1/p'; }
FR0=$($P logs/h32_form.py logs/diag_bat_contact_h34p_000.json h34p_000 $O/form_h34p_000.csv | tail -1)
ROWS="16300 + pin0, alpha 0.5 (h34p_000): $($P logs/h34p_gate.py jg_16300 h34p_000 $(pose "$FR0")) | $($P logs/h34q_gain.py jg_16300 h34p_000)"
trap 'printf "H34Q_END %s %s\n%s\n" "$(date +%H:%M)" "$MSG" "$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); if [ $((10#$H)) -ge 235 ] && [ $((10#$H)) -lt 1200 ]; then MSG="no Isaac launch at $H (film21 03:25): stopped before $MSG"; exit 0; fi; }
[ -f $BC ] || { MSG="no $BC"; exit 1; }
MSG="waiting for H34P_ADOPT_END"; until [ -f logs/H34P_ADOPT_END.txt ]; do sleep 60; done
WIN=""
for A in 0.40 0.35; do
  [ -f $O/STOP ] && { MSG="stopped by STOP before alpha $A"; exit 0; }
  T=h34q_${A/./}; MSG="diag $T"; idle
  AIB_ACT_EMA=$A timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $BC --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log --jit_groups > $O/diag_$T.txt 2>&1
  ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ] || { MSG="diag $T crash: $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $T $O/form_$T.csv | tail -1)
  GT=$($P logs/h34p_gate.py jg_16300 $T "$(pose "$FR")"); GN=$($P logs/h34q_gain.py jg_16300 $T)
  ROW="alpha $A (inference, pin0): $GT | $GN | $($P logs/h34p_ho.py h34p_base $T 000) | FORM $FR"
  ROWS="$ROWS
$ROW"; echo "$(date +%H:%M) H34Q $ROW" >> $O/progress.txt
  [ -z "$WIN" ] && [ "${GT%% *}" = PASS ] && [ "${GN%% *}" = GAIN ] && WIN=$A
done
MSG="done; alpha passing with the gain: ${WIN:-none}"
