#!/bin/bash
# H800L (planning 14, 11-01 03:1x; HEDGE, launch decided by 14 after DT_END): 800 Hz LONG fine-tune from the plain h9b
# model_12500 (not 13298 / 13796) on the unchanged Human30c recipe (AIB-PitchedBatting-Human30c-v0 = mimic + 30c fair-only
# carry, w_carry 20), AIB_PHYSICS_DT=0.00125 (decimation 8, control 0.01 s), 2048 envs, std free, adaptive lr (desired_kl
# 0.01, optimizer + lr restored at each resume; h9b at 12300-12694 ran 5.1e-5, H30C800 ran 7.6e-5 -> the same schedule,
# there is no low-lr override to undo), SAVE_EVERY 100, RAM watchdog 30 GB, AIB_ARMAIM_NOBOOK=1.
# NSEG segments of 500 its (default 6 = +3000); after each, one native 800 Hz eval row (saim, 4 joint clips, phase 2.6, j0,
# real heights) with the N4_400-style line (pool_stats hit / exit / launch / ss_max / dz + armaim_gate 300) appended to
# logs/h800l/progress.txt. Waits for logs/h800l/GO (14's) + idle; logs/h800l/STOP ends it before the next segment.
# Estimate: 800 Hz 12.2 s/it (H30C800, 2048 envs; 400 Hz long run 9.8 s/it) -> 500 its ~1.7 h + ~10 min row; +3000 ~11 h.
# -> H800L_<its>_END per segment, H800L_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
NSEG=${NSEG:-6}
TASK=AIB-PitchedBatting-Human30c-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/h800l; mkdir -p $O
M=logs/H800L_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H800L_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
MSG="waiting for logs/h800l/GO"
until [ -f $O/GO ]; do sleep 60; done
LR=2026-10-25_13-56-51_human_pitched9b; LC=model_12500.pt; PK=0
for k in $(seq 1 $NSEG); do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  idle
  RN=human_pitched_h800l_$k
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
  T=h800l_$I
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log > $O/diag_$T.txt 2>&1
  if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
    ROW="+$((k * 500)) model_$I (800 Hz, 4 clips): hit / exit / launch / ss_max / dz $($P $SC/pool_stats.py logs/diag_bat_contact_$T.json) ; $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label $T | head -1 | cut -c1-160)"
  else ROW="+$((k * 500)) model_$I: diag crash or no json"; fi
  ROW="$ROW ; $(grep -o 'Iteration time: [0-9.]*s' $O/train_$k.txt | tail -1), private peak ${PK} GB"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H800L_$((k * 500))_END $(date +%H:%M) $ROW" > logs/H800L_$((k * 500))_END.txt
  ROWS="$ROWS $ROW ||"
  LR=$(basename $D); LC=model_$I.pt
done
MSG="done (+$((NSEG * 500)))"
