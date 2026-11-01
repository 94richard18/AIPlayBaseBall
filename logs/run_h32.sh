#!/bin/bash
# H32 (planning 14, 11-01 20:xx; user: batter strength back to human level + no axis-alignment signal, it made the body shake
# in the swing): resume H31 model_14496 at AIB_EFFORT_SCALE 1.0 on AIB-PitchedBatting-Human32-v0 = Human31 rewards on the
# plain PitchedBattingEnv (no saim in the loop, percept_filter on), 800 Hz, 2048 envs, 500-it segments, SAVE_EVERY 100.
# Reward bounds come in from the caller env (AIB_SS_CA_LO / HI / WIN, AIB_EXIT_HINGE_W / REF / CAP, AIB_CARRY_REF / SPAN,
# AIB_LAUNCH_HI_W ...) - set from the E10 1.0x diag. 64-env 20-it smoke -> logs/H32_SMOKE_END.txt, then waits for logs/h32/GO.
# Per segment: STOP before, HOLD between segments (waits while logs/h32/HOLD exists), RAM watchdog 30 GB; eval = diag 4 clips
# (percept_filter, no saim, + jitter) and joint s6 (R5c, 6.5 s) -> logs/H32_<its>_END.txt; final logs/H32_END.txt.
# Segment-1 rule (14): hit >= 0.40, exit median >= EXREF - 1, torso jitter median <= JREF (14496 saim-off 1.0x row); else STOP.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EFFORT_PROBE
: ${EXREF:?} ${JREF:?}
NSEG=${NSEG:-6}
TASK=AIB-PitchedBatting-Human32-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
O=logs/h32; mkdir -p $O
M=logs/H32_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "H32_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
REW="ramp ${AIB_SS_CA_LO:-12.0}-${AIB_SS_CA_HI:-22.0} win ${AIB_SS_CA_WIN:-0.05}, exit hinge ${AIB_EXIT_HINGE_W:-0.0} ref ${AIB_EXIT_HINGE_REF:-16.0} cap ${AIB_EXIT_HINGE_CAP:-14.0}, carry ${AIB_CARRY_REF:-10.0}/${AIB_CARRY_SPAN:-30.0}, launch ceiling ${AIB_LAUNCH_HI_W:-0.0}"
LR=${LR0:-2026-11-01_11-42-57_human_pitched_h31_4}; LC=${LC0:-model_14496.pt}; PK=0
MSG="smoke"; idle
if [ ! -f logs/H32_SMOKE_END.txt ]; then
  AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task $TASK --headless --num_envs 64 --max_iterations 20 \
    --run_name h32_smoke --resume True --load_run $LR --checkpoint $LC > $O/smoke.txt 2>&1
  if grep -q Traceback $O/smoke.txt; then SM="CRASH $(grep -A14 Traceback $O/smoke.txt | tail -1)"
  elif ! grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/smoke.txt; then SM="no AIB_PHYSICS_DT line"
  elif grep -q "saim_swing" $O/smoke.txt; then SM="saim layer present"
  else SM="ok: $($P $SC/h31_curve.py $O/smoke.txt 10) ; $(grep -o 'Iteration time: [0-9.]*s' $O/smoke.txt | tail -1)"; fi
  echo "H32_SMOKE_END $(date +%H:%M) $SM ($REW)" > logs/H32_SMOKE_END.txt
  case "$SM" in ok*) ;; *) MSG="smoke failed: $SM"; exit 1;; esac
fi
MSG="smoke done, waiting for logs/h32/GO"
until [ -f $O/GO ]; do [ -f $O/STOP ] && { MSG="stopped before segment 1"; exit 0; }; sleep 60; done
RC=$(tr -d '\r\n' < logs/recov5c/recover_ckpt.txt)
for k in $(seq ${K0:-1} $NSEG); do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  while [ $k -gt 1 ] && [ -f $O/HOLD ]; do MSG="on HOLD before segment $k"; sleep 60; done
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k"; exit 0; }
  MSG="segment $k"; idle
  RN=human_pitched_h32_$k
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
  T=h32_$I; MSG="segment $k eval"
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log > $O/diag_$T.txt 2>&1
  if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
    ER=$($P logs/e10_row.py logs/diag_bat_contact_$T.json)
    ROW="+$((k * 500)) model_$I (1.0x, 800 Hz, 4 clips, no saim): $ER; pop-ups $($P logs/popup_share.py logs/diag_bat_contact_$T.json); $($P $SC/h31_carry.py logs/diag_bat_contact_$T.json 2>&1 | tail -1)"
  else ER=""; ROW="+$((k * 500)) model_$I: diag crash or no json"; fi
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s6_$I.csv > $O/joint_s6_$I.txt 2>&1
  if [ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt; then
    ROW="$ROW ; joint s6 (R5c): $($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1)"
  else ROW="$ROW ; joint s6 crashed"; fi
  ROW="$ROW ; $($P $SC/h31_curve.py $O/train_$k.txt 100) ; $REW ; $(grep -o 'Iteration time: [0-9.]*s' $O/train_$k.txt | tail -1), private peak ${PK} GB"
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H32_$((k * 500))_END $(date +%H:%M) $ROW" > logs/H32_$((k * 500))_END.txt
  ROWS="$ROWS $ROW ||"
  if [ $k -eq 1 ]; then  # segment-1 rule
    HT=$(echo "$ER" | sed -n 's/.* hit \([0-9.]*\) .*/\1/p'); EX=$(echo "$ER" | sed -n 's/.* exit \([0-9.]*\) .*/\1/p')
    JT=$(echo "$ER" | sed -n 's/.* torso_hp \([0-9.]*\) .*/\1/p')
    awk -v h="$HT" -v e="$EX" -v j="$JT" -v er="$EXREF" -v jr="$JREF" 'BEGIN{exit !(h ~ /^[0-9.]+$/ && e ~ /^[0-9.]+$/ && j ~ /^[0-9.]+$/ && h + 0 >= 0.40 && e + 0 >= er - 1 && j + 0 <= jr + 0)}' \
      || { MSG="segment-1 rule FAILED (hit '$HT' >= 0.40, exit '$EX' >= $EXREF - 1, torso jitter '$JT' <= $JREF) -> stopped"; exit 0; }
  fi
  LR=$(basename $D); LC=model_$I.pt
done
MSG="done (+$((NSEG * 500)))"
