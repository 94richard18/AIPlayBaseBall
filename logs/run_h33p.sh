#!/bin/bash
# H33 probes (planning 14, 11-02 22:xx, approved under the 11-06 extension): why H33R seg 1' dropped the ball by +150 (hit 0.26 ->
# 0.00, ss_ca 15.3 -> 12.8, pose 0.20 -> 0.23). From plain 12500 at 1.0x, H33R rewards (ramp on hits only, form factor), 2048
# envs, 150 its each, sequential: P1 = da L1 0 + wait torso L1 0 (penalty weights); P2 = form factor off the hit rewards (on the
# ramp only). Read at +150: train hit, pose_swing, ss_ca + the per-term mean episode rewards (bat/rt_*). Holds = hit >= 0.25 and
# ss_ca >= 15. Markers logs/H33P1_END.txt / logs/H33P2_END.txt (written by the EXIT trap per probe).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_FORM_S AIB_FORM_ON_HITS
export AIB_SS_CA_HIT_ONLY=1 AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=14.7 AIB_SS_CA_HI=24.7 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=15.5 AIB_EXIT_HINGE_CAP=14.0 \
  AIB_CARRY_REF=15 AIB_CARRY_SPAN=35 AIB_OUTFIELD_W=10 AIB_LAUNCH_HI_W=5 AIB_LAUNCH_HI_REF=35 AIB_LAUNCH_HI_SPAN=15 \
  AIB_LAUNCH_LO_W=5 AIB_LAUNCH_LO_REF=0 AIB_LAUNCH_LO_SPAN=15 AIB_DA_L1_W=0.2 AIB_WAIT_TORSO_W=0.03 AIB_FALL_AFTER_HIT_W=10.0
TASK=AIB-PitchedBatting-Human33-v0; O=logs/h33
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
probe() {  # $1 name, $2 env overrides
  ( M=logs/H33${1}_END.txt; rm -f $M; MSG="aborted"
    trap 'echo "H33${1}_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
    MSG="waiting for idle"; idle; MSG="training"
    env $2 AIB_PHYSICS_DT=0.00125 timeout 3600 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 150 \
      --run_name h33p_$1 --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_p$1.txt 2>&1
    grep -q Traceback $O/train_p$1.txt && { MSG="CRASH $(grep -A14 Traceback $O/train_p$1.txt | tail -1)"; exit 1; }
    CV=$($P logs/h33_curve.py $O/train_p$1.txt 50 12650)
    RT=$(for k in rt_ramp rt_hit rt_carry rt_exit_hinge rt_outfield rt_launch_pen rt_da_l1 rt_wait_torso rt_miss rt_fall rt_track; do echo -n "${k#rt_} $(grep -a "bat/$k:" $O/train_p$1.txt | tail -1 | awk '{print $NF}') "; done)
    HV=$(echo "$CV" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p'); SV=$(echo "$CV" | sed -n 's/.* ss_ca \([0-9.]*\) .*/\1/p')
    H=$(awk -v h="$HV" -v s="$SV" 'BEGIN{print (h + 0 >= 0.25 && s + 0 >= 15) ? "HOLDS" : "DROPS"}')
    MSG="$H ($2) | +150 $CV | episode reward terms: $RT | curve $($P logs/h33_curve.py $O/train_p$1.txt 25) | $(grep -o 'Iteration time: [0-9.]*s' $O/train_p$1.txt | tail -1)" )
}
probe P1 "AIB_DA_L1_W=0 AIB_WAIT_TORSO_W=0"
probe P2 "AIB_FORM_ON_HITS=0"
