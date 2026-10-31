#!/bin/bash
# H31 (planning 14, 11-01 05:5x, GO for the design): Human31 = Human30c + once-per-swing bat speed AT the ball, hit or not
# (40 x clamp((ss_ca - lo) / 10, 0, 1), ss_ca = max |v_ss| in the 50 ms before the ball's closest approach to the sweet spot /
# the impact; lo 12) + w_track 0.5; carry 20 fair-only, w_miss 5, spray 10 x q, contact / geometry 0. 800 Hz
# (AIB_PHYSICS_DT=0.00125), resume plain h9b model_12500 @ 1.5x, saim + percept_filter, 2048 envs, SAVE_EVERY 100, RAM
# watchdog 30 GB, NOBOOK. Start: logs/FILM3_END.txt + logs/PDT_END.txt (or no PDT_GO) + Isaac 0 + no logs/h31/VETO ->
# 64-env 20-it smoke -> logs/H31_SMOKE_END.txt -> waits for logs/h31/GO (written after 14 sees the smoke line).
# Segments of 500 its (NSEG, default 6): STOP checked before each, HOLD between segments. +50 check of segment 1:
# bat/ss_ca_lt_lo > 0.25 -> lo 10 from segment 2 (14, 05:5x) -> logs/H31_50_CHECK.txt. Per segment: the same eval row as the
# 800 Hz long run (next to N4_400 / B800) + the training curve every 100 its -> logs/H31_<its>_END.txt, final logs/H31_END.txt.
# Film4 hook (14, 11-01 05:4x / 06:0x): HOLD is honoured from 18:30 (11-01) on, or anytime when logs/h31/HOLD contains "now".
# At the first honoured HOLD boundary: joint s6 (64 envs, saim + R4 recover, phase 2.6) of logs/h31/FILM_CKPT (else the newest
# segment checkpoint with eval hit >= 0.40, logs/h31/passed.txt) -> logs/h31/joint_s6_<its>.csv + logs/H31_JOINT_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
export AIB_SS_CA_LO=${AIB_SS_CA_LO:-12.0} AIB_SS_CA_WIN=${AIB_SS_CA_WIN:-0.05}
NSEG=${NSEG:-6}
TASK=AIB-PitchedBatting-Human31-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/h31; mkdir -p $O
M=logs/H31_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H31_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
MSG="waiting for FILM3_END + PDT done + Isaac 0 (VETO absent)"
until [ -f logs/FILM3_END.txt ] && { [ ! -f logs/PDT_GO.txt ] || [ -f logs/PDT_END.txt ]; } && [ ! -f $O/VETO ] && [ "$(isaac)" -eq 0 ]; do sleep 60; done
idle
if [ ! -f logs/H31_SMOKE_END.txt ]; then
  AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 20     --run_name h31_smoke --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/smoke.txt 2>&1
  if grep -q Traceback $O/smoke.txt; then SM="CRASH $(grep -A14 Traceback $O/smoke.txt | tail -1)"
  elif ! grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/smoke.txt; then SM="no AIB_PHYSICS_DT line"
  else SM="ok: $($P $SC/h31_curve.py $O/smoke.txt 10) ; $(grep -o 'Iteration time: [0-9.]*s' $O/smoke.txt | tail -1)"; fi
  echo "H31_SMOKE_END $(date +%H:%M) $SM" > logs/H31_SMOKE_END.txt
  case "$SM" in ok*) ;; *) MSG="smoke failed: $SM"; exit 1;; esac
fi
MSG="smoke done, waiting for logs/h31/GO"
until [ -f $O/GO ]; do [ -f $O/STOP ] && { MSG="stopped before segment 1"; exit 0; }; sleep 60; done
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov4/recover_ckpt.txt)
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
joint() {  # film4 rows of the filmed checkpoint
  local JC JI
  JC=$([ -f $O/FILM_CKPT ] && tr -d '
' < $O/FILM_CKPT); [ -n "$JC" ] || JC=$(tail -1 $O/passed.txt 2>/dev/null)
  if [ -z "$JC" ] || [ ! -f "$JC" ]; then
    echo "H31_JOINT_END $(date +%H:%M) SKIPPED: no FILM_CKPT and no segment checkpoint with hit >= 0.40 (ckpt '$JC')" > logs/H31_JOINT_END.txt; return
  fi
  JI=$(basename $JC .pt); JI=${JI#model_}
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $JC --envs 64 --seed 6 --after_done recover_rl     --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --phase_horizon 2.6 --csv $O/joint_s6_$JI.csv     > $O/joint_s6_$JI.txt 2>&1
  if grep -q Traceback $O/joint_s6_$JI.txt || [ ! -f $O/joint_s6_$JI.csv ]; then
    echo "H31_JOINT_END $(date +%H:%M) CRASH $JC: $(grep -A14 Traceback $O/joint_s6_$JI.txt | tail -1)" > logs/H31_JOINT_END.txt
  else
    echo "H31_JOINT_END $(date +%H:%M) BC=$JC ROWS=$O/joint_s6_$JI.csv (recover $(basename $(dirname $RC))): $($P $SC/h31_joint_row.py $O/joint_s6_$JI.csv)" > logs/H31_JOINT_END.txt
  fi
}
jl() { $P -c "import sys; h, e, l, s, d = sys.argv[1:6]; print(f'ss {s} exit {e} launch {l} hit {h} dz {d}')" $1 $2 $3 $4 $5; }
J4=$(jl $($P $SC/pool_stats.py logs/diag_bat_contact_N4_400.json)); B8=$(jl $($P $SC/pool_stats.py logs/diag_bat_contact_h30c800_base.json))
LR=${LR0:-2026-10-25_13-56-51_human_pitched9b}; LC=${LC0:-model_12500.pt}; PK=0  # LR0 / LC0 / K0: resume a later segment
for k in $(seq ${K0:-1} $NSEG); do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  if [ $k -gt 1 ] && [ -f $O/HOLD ]; then  # between segments only; evening (film4) or "now"
    EV=$([ "$(date +%Y%m%d%H%M)" -ge 202611011830 ] && echo 1)
    if [ -n "$EV" ] || grep -q now $O/HOLD; then
      [ ! -f logs/H31_JOINT_END.txt ] && { MSG="joint s6 for film4 before segment $k"; joint; }
      while [ -f $O/HOLD ]; do MSG="on HOLD before segment $k"; sleep 60; done
    fi
  fi
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  idle
  RN=human_pitched_h31_$k
  AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name $RN \
    --resume True --load_run $LR --checkpoint $LC > $O/train_$k.txt 2>&1 &
  TP=$!; C50=""
  while kill -0 $TP 2>/dev/null; do
    G=$(priv $RN); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$RN' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: segment $k trainer private ${G} GB > 30, killed"; exit 1
    fi
    if [ $k -eq 1 ] && [ -z "$C50" ] && grep -q "Learning iteration 12550/" $O/train_$k.txt; then
      C50=$($P $SC/h31_curve.py $O/train_$k.txt 50 12550)
      LT=$(echo "$C50" | sed -n 's/.* lt_lo \([0-9.]*\).*//p')
      if awk -v x="$LT" 'BEGIN{exit !(x > 0.25)}'; then export AIB_SS_CA_LO=10.0; C50="$C50 -> lo 10 from segment 2"; else C50="$C50 -> lo 12 kept"; fi
      echo "H31_50_CHECK $(date +%H:%M) $C50" > logs/H31_50_CHECK.txt
    fi
    sleep 30
  done
  grep -q Traceback $O/train_$k.txt && fail $O/train_$k.txt train_$k
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/train_$k.txt || { MSG="segment $k: no AIB_PHYSICS_DT line"; exit 1; }
  D=$(ls -td $B/*$RN 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="segment $k: no checkpoint"; exit 1; }
  T=h31_$I
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log > $O/diag_$T.txt 2>&1
  if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
    PS=$($P $SC/pool_stats.py logs/diag_bat_contact_$T.json)
    awk -v x="${PS%% *}" 'BEGIN{exit !(x >= 0.40)}' && echo "$D/model_$I.pt" >> $O/passed.txt
    ROW="+$((k * 500)) model_$I (800 Hz, 4 clips): $(jl $PS) ($($P $SC/ssc.py logs/diag_bat_contact_$T.json)) [N4_400: $J4 | B800: $B8] ; $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label $T | head -1 | cut -c1-160)"
  else ROW="+$((k * 500)) model_$I: diag crash or no json"; fi
  ROW="$ROW ; $($P $SC/h31_curve.py $O/train_$k.txt 100) ; lo $AIB_SS_CA_LO win $AIB_SS_CA_WIN"
  ROW="$ROW ; $(grep -o 'Iteration time: [0-9.]*s' $O/train_$k.txt | tail -1), private peak ${PK} GB"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H31_$((k * 500))_END $(date +%H:%M) $ROW" > logs/H31_$((k * 500))_END.txt
  ROWS="$ROWS $ROW ||"
  LR=$(basename $D); LC=model_$I.pt
done
MSG="done (+$((NSEG * 500)))"
