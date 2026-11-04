#!/bin/bash
# run_h32w_diags.sh (14, 11-05 07:3x, closure of H32W): --jit_groups EMA-on j0 1024 diags of h32w/model_16400 and model_16500 (the 16599 settings),
# rows vs 16300 in logs/H32W_<I>_DIAG_END.txt: does contact quality fall monotonically with the wrist term or collapse late.
# No training. No launch after 16:00 (film20 16:25).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.5
B=logs/rsl_rl/aib1_tee_batting_mimic; O=logs/h33
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
MSG="aborted"; M=logs/H32W_DIAGS_CHAIN_END.txt
trap 'echo "H32W_DIAGS_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); [ $((10#$H)) -lt 1600 ] || { MSG="not started at $H (GPU free until 16:20)"; exit 0; }; }
D=$(ls -td $B/*_human_pitched_h32w | head -1)
for I in 16400 16500; do
  T=h32w_$I; MK=logs/H32W_${I}_DIAG_END.txt; rm -f $MK
  MSG="diag $T"; idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log --jit_groups > $O/diag_$T.txt 2>&1
  ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ] || { MSG="diag $T crash: $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  G2=$($P logs/ema_gate.py h32e_16300 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_15993_h32_pin.csv | cut -d'|' -f2-3)
  FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I $O/form_$I.csv | tail -1)
  WG=$($P logs/h32w_gate.py jg_16300 $T)
  ROW="model_$I (H32W, EMA 0.5) | diag j0 1024 vs 16300:$G2 | FORM $FR | JITTER vs jg_16300: $WG"
  echo "H32W_${I}_DIAG_END $(date +%H:%M) $ROW" > $MK; echo "$(date +%H:%M) H32W $I diag: $ROW" >> $O/progress.txt
done
MSG="done: H32W_16400_DIAG_END + H32W_16500_DIAG_END"
