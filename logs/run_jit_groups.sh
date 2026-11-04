#!/bin/bash
# run_jit_groups.sh (14, 11-05 05:5x; user: hands / feet tremble, the pelvis / torso pose CSV cannot separate film19 from film18):
# --jit_groups jitter-0 1024-env diag of 15993 (EMA off, its own protocol) and 16300 (AIB_ACT_EMA 0.5), one row each in
# logs/JITTER_GROUPS_END.txt (logs/jit_groups.py). No training. GPU free until 16:20 (14); no film running (nofilm).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W AIB_ACT_EMA
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2
B=logs/rsl_rl/aib1_tee_batting_mimic; O=logs/h33
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
M=logs/JITTER_GROUPS_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "JITTER_GROUPS_END $(date +%H:%M) $MSG$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); [ $((10#$H)) -lt 1600 ] || { MSG="not started at $H (GPU free until 16:20)"; exit 0; }; }
diag() {  # $1 checkpoint, $2 tag, $3 label, $4 AIB_ACT_EMA (empty = off)
  MSG="diag $2"; idle
  if [ -n "$4" ]; then export AIB_ACT_EMA=$4; else unset AIB_ACT_EMA; fi
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --percept_filter --envs 1024 --seed 0 --tag $2 \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log --jit_groups > $O/diag_$2.txt 2>&1
  ! grep -q Traceback $O/diag_$2.txt && [ -f logs/diag_bat_contact_$2.json ] || { MSG="diag $2 crash: $(grep -A14 Traceback $O/diag_$2.txt | tail -1)"; exit 1; }
  ROWS="$ROWS || $($P logs/jit_groups.py $2 "$3")"
}
[ -f $O/STOP ] && { MSG="stopped by STOP"; exit 0; }
diag $B/2026-11-02_04-02-58_human_pitched_h32_3/model_15993.pt jg_15993 "15993 EMA off" ""
diag $(ls -td $B/*_human_pitched_h32e | head -1)/model_16300.pt jg_16300 "16300 EMA 0.5" 0.5
MSG="done (rad/s RMS of the 5 Hz high-passed joint velocity; all = start -> contact, wait = start -> release, app = release -> contact)"
echo "$(date +%H:%M) jitter groups:$ROWS" >> $O/progress.txt
