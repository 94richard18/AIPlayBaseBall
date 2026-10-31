#!/bin/bash
# v3 (batter 15, 10-31 13:3x): replaces run_h30c_long2.sh while it waited for the film (killed in its wait loop, segment 1
# and its +500 rows done: model_13298). From here: wait for FILM_1031B_END (newer than logs/h30c_long/chain.stamp of v2) ->
# effort-scale probe (14 11:55: 12500 row 9, jitter 0, AIB_EFFORT_SCALE 1.5 / 2.0 / 2.5, AIB_EFFORT_PROBE=1, --sat_log)
# -> EFFORT_SCALE_END -> model_13298 0.03 tables 3 x 5 ONE OUTS + mechanism split (14 13:27) -> H30C_500_TABLES_END ->
# segments 2 and 3 (500 its each, resumed, two diag rows after each) -> H30C_LONG_END. 14's stop rule: the +1000 exit
# median must be >= 21.6 (logs/h30c_long/STOP makes the chain exit before segment 3).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=100 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD
TASK=AIB-PitchedBatting-Human30c-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
MECH=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad/one_outs_mechanism_run.py
O=logs/h30c_long
M=logs/H30C_LONG_END.txt; rm -f $M logs/EFFORT_SCALE_END.txt logs/H30C_500_TABLES_END.txt; MSG="aborted"
trap 'echo "H30C_LONG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
D1=$(ls -td $B/*human_pitched30c_long1 | head -1); I1=13298
[ -f $D1/model_$I1.pt ] || { MSG="missing $D1/model_$I1.pt"; exit 1; }
ROWS=" $(grep "model_$I1" $O/progress.txt | tail -1 | cut -d' ' -f2-) ||"
until [ -f logs/FILM_1031B_END.txt ] && [ logs/FILM_1031B_END.txt -nt $O/chain.stamp ]; do sleep 60; done
# 14 (11:55): effort-scale measurement
H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt; ES=""
for e in 1.5 2.0 2.5; do
  TG=es_${e/./}_r9
  idle
  AIB_EFFORT_SCALE=$e AIB_EFFORT_PROBE=1 timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 $SAIM --envs 1024 --seed 0 \
    --tag $TG --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log --sat_log > $O/diag_$TG.txt 2>&1
  if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
    SAT=$($P -c "import json; t = json.load(open('logs/diag_bat_contact_$TG.json'))['summary'].get('torque_sat', {}); print(', '.join(f'{k} {v}' for k, v in list(t.items())[:4]))")
    ES="$ES | effort $e: $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label e$e | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label e$e) | torque_sat top4 $SAT"
  else ES="$ES | effort $e: crash or no json"; fi
done
echo "EFFORT_SCALE_END $(date +%H:%M)$ES" > logs/EFFORT_SCALE_END.txt
# 14 (13:27): model_13298 through the 0.03 tables + mechanism split
HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_h30cL_${I1}_j0.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
: > $O/t3_$I1.txt
for j in 1 2 3; do
  idle
  timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D1/model_$I1.pt $SAIM --eval_envs 1024 --timing_noise 0.03 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_${I1}_$j.txt 2>&1
  grep -q Traceback $O/table_${I1}_$j.txt && fail $O/table_${I1}_$j.txt table_$j
  $P scripts/make_batter_table.py $O/table_${I1}_$j.txt $O/table_${I1}_$j.json > /dev/null || { MSG="table $j: make_batter_table failed"; exit 1; }
  echo "L${I1}_$j $(oo $O/table_${I1}_$j.json $HR)" >> $O/t3_$I1.txt
done
TB=$($P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$O/t3_$I1.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"T3 {st.mean(w):.1f}% (tables {', '.join(f'{x:.1f}' for x in w)}) OF {st.mean(o):.1f}% (adoption bar 19.1 / 3.0: {'MET' if st.mean(w) >= 19.1 and st.mean(o) >= 3.0 else 'not met'})")
PY
)
cp scripts/one_outs_mechanism.py $MECH
MC=$(cd C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad && \
  $P - <<PY 2>&1 | tail -1
import os, runpy, sys
os.chdir(r"D:/MyProject/AIPlayBaseBall"); sys.path.insert(0, r"D:/MyProject/AIPlayBaseBall")
sys.argv = ["mech", "--name", "h30cL$I1", "--hit_rate", "$HR", "--tables", "$O/table_${I1}_1.json", "$O/table_${I1}_2.json", "$O/table_${I1}_3.json"]
runpy.run_path(r"$MECH", run_name="__main__")
PY
)
echo "H30C_500_TABLES_END $(date +%H:%M) model_$I1 0.03: $TB | hit rate $HR | mechanism: $MC" > logs/H30C_500_TABLES_END.txt
# segments 2 and 3
LR=$(basename $D1); LC=model_$I1.pt; PK=0
for k in 2 3; do
  [ -f $O/STOP ] && { MSG="stopped by 14 before segment $k |$ROWS"; exit 0; }
  idle
  $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name human_pitched30c_long$k --resume True \
    --load_run $LR --checkpoint $LC > $O/train_$k.txt 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
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
    TG=h30cL_${I}_j$j
    idle
    timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag $TG --timing_noise $j \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$TG.txt 2>&1
    if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
      ROW="$ROW | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label j$j)"
    else ROW="$ROW | j$j: crash or no json"; fi
  done
  echo "$(date +%H:%M) $ROW" >> $O/progress.txt
  echo "H30C_LONG_${k}_END $(date +%H:%M) $ROW" > logs/H30C_LONG_${k}_END.txt
  ROWS="$ROWS $ROW ||"
  LR=$(basename $D); LC=model_$I.pt
done
MSG="done +1500 |$ROWS trainer private peak ${PK} GB (segments 2-3)"
