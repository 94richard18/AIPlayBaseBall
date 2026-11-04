#!/bin/bash
# run_h32e_diag.sh (11-04 20:5x): H32E was killed at 16393 by the ss_ca falsifier (50-it ss 23.83 < 24.3) while train hit rose
# (0.799 vs 0.692) and the wait torso unit fell 15%. Same cheap check as H32P / H32Q: jitter-0 1024-env diag of the newest saved
# checkpoint (model_16300) with AIB_ACT_EMA=0.5 on, vs form_15993. Writes logs/H32E_16300_DIAG_END.txt. No Isaac 05:15-12:00.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE \
  AIB_FALL_AFTER_HIT_W AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_WAIT_TORSO_W AIB_WAIT_TORSO_RAMP_STEPS AIB_PRESTART_PIN AIB_LIFT_W AIB_LIFT_REF AIB_LIFT_SPAN AIB_LAUNCH_LO_ALL \
  AIB_FORM_S AIB_FORM_HINGE_HI AIB_FORM_HINGE_W AIB_LOST_IS_MISS AIB_SS_CA_HIT_ONLY AIB_DA_PRE_W
export AIB_SS_CA_LO=24.1 AIB_SS_CA_HI=34.1 AIB_SS_CA_WIN=0.01 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=18.7 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=40 AIB_LAUNCH_HI_SPAN=15 AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 \
  AIB_LAUNCH_LO_SPAN=15 AIB_OUTFIELD_W=10 AIB_DA_L1_W=0.2 AIB_ACT_EMA=0.5
B=logs/rsl_rl/aib1_tee_batting_mimic; RN=human_pitched_h32e; O=logs/h33
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
D=$(ls -td $B/*$RN | head -1); I=16300; BC=$D/model_$I.pt; T=h32e_$I
M=logs/H32E_16300_DIAG_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H32E_16300_DIAG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  H=$(date +%H%M); if [ $((10#$H)) -ge 515 ] && [ $((10#$H)) -lt 1200 ]; then MSG="no Isaac launch at $H (film19 window 05:15-12:00)"; exit 0; fi; }
[ -f $BC ] || { MSG="no $BC"; exit 1; }
[ -f $O/STOP ] && { MSG="stopped by STOP"; exit 0; }
MSG="diag"; idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $BC --percept_filter --envs 1024 --seed 0 --tag $T \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
  --physics_dt 0.00125 --contact_log --form_log > $O/diag_$T.txt 2>&1
! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ] || { MSG="diag crash"; exit 1; }
G=$($P logs/ema_gate.py form_15993 $T $O/joint_s6_15993_h32_pin.csv $O/joint_s6_15993_h32_pin.csv | cut -d'|' -f2-3)
FR=$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I $O/form_$I.csv | tail -1)
MSG="$G | FORM $FR"
echo "$(date +%H:%M) H32E 16300 diag (EMA 0.5 on): $MSG" >> $O/progress.txt
