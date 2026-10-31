#!/bin/bash
# v2 (batter 15, 10-31 11:5x): run_h30c_long.sh's control script was replaced while segment 1 trained (the trainer
# kept running); this copy attaches to the running human_pitched30c_long1 trainer, and adds 14's 11:55 measurement after
# FILM_1031B_END and before segment 2: the 12500 row-9 config, diag1024 seed 0 jitter 0 at AIB_EFFORT_SCALE 1.5 / 2.0 /
# 2.5 (AIB_EFFORT_PROBE=1, a zero-shot strength probe, no spec change) -> logs/EFFORT_SCALE_END.txt.
# Human30c LONG fine-tune (planning 14, 10-31 10:36): resume human_pitched30c_1 model_12799 with the unchanged 30c cfg
# (AIB-PitchedBatting-Human30c-v0, no reward change) for +1500 its, save every 100, RAM watchdog 30 GB, AIB_ARMAIM_NOBOOK=1.
# Released automatically after H30C_ADOPT_END (newer than this chain's start). One Isaac at a time, so no diag can run beside
# the trainer: 3 resumed segments of 500 its (optimizer + adaptive lr restored at each resume), and after each segment the
# two diag rows (jitter 0 / 0.03, gate 300 + carry_cols) are appended to logs/h30c_long/progress.txt. -> H30C_LONG_END.
# 14's stop rule: +500 and +1000 exit medians both <= 19.7 -> 14 stops it (logs/h30c_long/STOP makes the chain exit
# before the next segment).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD
TASK=AIB-PitchedBatting-Human30c-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
O=logs/h30c_long; mkdir -p $O
M=logs/H30C_LONG_END.txt; rm -f $M logs/H30C_LONG_500_END.txt; MSG="aborted"
touch $O/chain.stamp
trap '[ -f logs/H30C_LONG_500_END.txt ] || echo "H30C_LONG_500_END $(date +%H:%M) chain ended before +500: $MSG" > logs/H30C_LONG_500_END.txt; echo "H30C_LONG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
LR=2026-10-31_07-28-04_human_pitched30c_1; LC=model_12799.pt; PK=0; ROWS=""
for k in 1 2 3; do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k |$ROWS"; exit 0; }
  if [ $k = 1 ] && [ "$(priv human_pitched30c_long1)" -ge 0 ]; then
    TP=""  # attach: segment 1 was launched by the v1 control script
  else
    idle
    $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name human_pitched30c_long$k --resume True \
      --load_run $LR --checkpoint $LC > $O/train_$k.txt 2>&1 &
    TP=$!
  fi
  while { [ -n "$TP" ] && kill -0 $TP 2>/dev/null; } || { [ -z "$TP" ] && [ "$(priv human_pitched30c_long$k)" -ge 0 ]; }; do
    G=$(priv human_pitched30c_long$k); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'human_pitched30c_long$k' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: segment $k trainer private ${G} GB > 30, killed |$ROWS"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $O/train_$k.txt && fail $O/train_$k.txt train_$k
  D=$(ls -td $B/*human_pitched30c_long$k 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="segment $k: no checkpoint |$ROWS"; exit 1; }
  ROW="+$((k * 500)) model_$I"
  for j in 0 0.03; do
    T=h30cL_${I}_j$j
    idle
    timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag $T --timing_noise $j \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
    if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
      ROW="$ROW | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$T.json --label j$j)"
    else ROW="$ROW | j$j: crash or no json"; fi
  done
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  ROWS="$ROWS $ROW ||"
  LR=$(basename $D); LC=model_$I.pt
  # 14 (10:40): the 12-hour film (aiplaybaseball-20) goes between segments 1 and 2, no Isaac beside it
  if [ $k = 1 ]; then
    echo "H30C_LONG_500_END $(date +%H:%M) $ROW" > logs/H30C_LONG_500_END.txt
    until [ -f logs/FILM_1031B_END.txt ] && [ logs/FILM_1031B_END.txt -nt $O/chain.stamp ]; do sleep 60; done
    # 14 (11:55): effort-scale measurement before segment 2
    H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt; ES=""
    for e in 1.5 2.0 2.5; do
      T=es_${e/./}_r9
      idle
      AIB_EFFORT_SCALE=$e AIB_EFFORT_PROBE=1 timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 $SAIM --envs 1024 --seed 0 \
        --tag $T --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log --sat_log > $O/diag_$T.txt 2>&1
      if ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
        SAT=$($P -c "import json; t = json.load(open('logs/diag_bat_contact_$T.json'))['summary'].get('torque_sat', {}); print(', '.join(f'{k} {v}' for k, v in list(t.items())[:4]))")
        ES="$ES | effort $e: $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label e$e | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$T.json --label e$e) | torque_sat top4 $SAT"
      else ES="$ES | effort $e: crash or no json"; fi
    done
    echo "EFFORT_SCALE_END $(date +%H:%M)$ES" > logs/EFFORT_SCALE_END.txt
  fi
done
MSG="done +1500 |$ROWS trainer private peak ${PK} GB"
