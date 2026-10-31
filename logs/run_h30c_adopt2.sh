#!/bin/bash
# Human30c long: pick + full adoption + segment 3 (planning 14, 10-31 14:25). run_h30c_long3.sh is held before segment 3 by
# logs/h30c_long/STOP (it exits after the +1000 rows, writing H30C_LONG_2_END and H30C_LONG_END). This chain then:
# 1) picks the better of model_13298 / the +1000 checkpoint on the jitter-0 diag rows: +1000 if its in-zone exit median
#    >= 21.6 (14's stop rule) AND D share >= 0.26 AND launch median in [10, 40], else 13298;
# 2) full adoption for the pick: 0.03 3 tables x 5 seeds (13298's existing set is reused), jitter-0 3 x 5, joint s6
#    (R3 recover + saim), mechanism split at 0.03 and at jitter 0 -> logs/H30C_ADOPT2_END.txt, PASS / FAIL on 14's amended
#    rule: 0.03 mean >= 16.6 AND outfield >= 6% AND s6 on the ground <= 2/64 (pitch columns 64/64);
# 3) segment 3 (+1500) from the +1000 checkpoint only if the stop rule held (exit >= 21.6), two diag rows ->
#    H30C_LONG_3_END. logs/h30c_long/STOP3 skips segment 3. One Isaac at a time; markers by EXIT trap.
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
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov3/recover_ckpt.txt)
MECH=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad/one_outs_mechanism_run.py
SCR=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/h30c_long; A=logs/h30cadopt2; mkdir -p $A
M=logs/H30C_ADOPT2_END.txt; rm -f $M logs/H30C_LONG_3_END.txt; MSG="aborted"
touch $A/chain.stamp
trap 'echo "H30C_ADOPT2_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
hr() { $P -c "import json; print(json.load(open('$1'))['summary']['by_speed']['slow in zone']['hit_rate'])"; }
# exit median / D share / launch median on in-zone hits (same selection as carry_cols.py)
cols() { $P - <<PY
import json, numpy as np
P = {k: np.array(v, dtype=float) for k, v in json.load(open("$1"))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10); hz = P["hit"].astype(bool) & z
D = P["D_any"][hz]; D = D[np.isfinite(D)]
print(f"{np.nanmedian(P['ev_any'][hz]):.2f} {((D >= 0) & (D <= 0.025)).mean():.3f} {np.nanmedian(P['launch_any'][hz]):.1f}")
PY
}
tables() {  # $1 ckpt, $2 timing noise, $3 prefix, $4 hit rate, $5 out list file
  : > $5
  for j in 1 2 3; do
    idle
    timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $SAIM --eval_envs 1024 --timing_noise $2 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > ${3}_$j.txt 2>&1
    grep -q Traceback ${3}_$j.txt && fail ${3}_$j.txt $(basename $3)_$j
    $P scripts/make_batter_table.py ${3}_$j.txt ${3}_$j.json > /dev/null || { MSG="$(basename $3) $j: make_batter_table failed"; exit 1; }
    echo "t_$j $(oo ${3}_$j.json $4)" >> $5
  done
}
mech() {  # $1 name, $2 hit rate, $3 table prefix
  (cd $SCR && $P - <<PY 2>&1 | tail -1
import os, runpy, sys
os.chdir(r"D:/MyProject/AIPlayBaseBall"); sys.path.insert(0, r"D:/MyProject/AIPlayBaseBall")
sys.argv = ["mech", "--name", "$1", "--hit_rate", "$2", "--tables", "${3}_1.json", "${3}_2.json", "${3}_3.json"]
runpy.run_path(r"$MECH", run_name="__main__")
PY
)
}
t3() { $P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$1") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"{st.mean(w):.2f} {st.mean(o):.2f} {','.join(f'{x:.1f}' for x in w)}")
PY
}
until [ -f logs/H30C_LONG_END.txt ] && [ logs/H30C_LONG_END.txt -nt $A/chain.stamp ]; do sleep 60; done
[ -f logs/H30C_LONG_2_END.txt ] || { MSG="long3 ended without the +1000 rows: $(cut -c1-300 logs/H30C_LONG_END.txt)"; exit 1; }
D2=$(ls -td $B/*human_pitched30c_long2 | head -1); I2=$(last $D2)
D1=$(ls -td $B/*human_pitched30c_long1 | head -1); I1=13298
J2=logs/diag_bat_contact_h30cL_${I2}_j0.json; J1=logs/diag_bat_contact_h30cL_${I1}_j0.json
[ -f $J2 ] || { MSG="no +1000 jitter-0 diag json ($J2)"; exit 1; }
read EX2 DS2 LA2 <<< "$(cols $J2)"; read EX1 DS1 LA1 <<< "$(cols $J1)"
KEEP=$($P -c "print(int($EX2 >= 21.6))")
if [ "$($P -c "print(int($EX2 >= 21.6 and $DS2 >= 0.26 and 10 <= $LA2 <= 40))")" = 1 ]; then CK=$D2/model_$I2.pt; L=$I2; else CK=$D1/model_$I1.pt; L=$I1; fi
PICK="pick model_$L (+1000 model_$I2: exit $EX2, D $DS2, launch $LA2; 13298: exit $EX1, D $DS1, launch $LA1; stop rule exit >= 21.6: $([ $KEEP = 1 ] && echo held || echo FAILED))"
echo "$(date +%H:%M) $PICK" >> $A/progress.txt
HR=$(hr logs/diag_bat_contact_h30cL_${L}_j0.json)
# 0.03 set
if [ $L = $I1 ]; then
  for j in 1 2 3; do cp $O/table_${I1}_$j.json $A/t3_$j.json; done; cp $O/t3_$I1.txt $A/t3.txt
else tables $CK 0.03 $A/t3 $HR $A/t3.txt; fi
# jitter-0 set
tables $CK 0 $A/t0 $HR $A/t0.txt
# joint s6
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $CK --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --trig_dump $A/trig_joint.npz \
  --csv $A/joint_s6.csv > $A/joint_s6.txt 2>&1
grep -q Traceback $A/joint_s6.txt && fail $A/joint_s6.txt joint_s6
read W3 O3 TB3 <<< "$(t3 $A/t3.txt)"; read W0 O0 TB0 <<< "$(t3 $A/t0.txt)"
ST=$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$A/joint_s6.csv')))
cols = ['clip', 'kmh', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b)); g = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b)
hits = sum(x['hit'] == 'True' for x in b)
ok = $W3 >= 16.6 and $O3 >= 6.0 and same == 64 and g <= 2
print(("PASS" if ok else "FAIL") + f" (amended rule 16.6 / OF 6 / ground <= 2) 0.03 T3 $W3% (tables $TB3) OF $O3% | jitter 0 T3 $W0% (tables $TB0) OF $O0% | "
      f"joint s6 (recover {'$RC'.split('/')[-2]}): pitch columns {same}/64, on the ground {g}/64, hits {hits}/64")
PY
)
cp scripts/one_outs_mechanism.py $MECH
MC3=$(mech h30cL${L} $HR $A/t3); MC0=$(mech h30cL${L}_j0 $HR $A/t0)
MSG="model_$L: $ST | hit rate $HR | $PICK | mechanism 0.03: $MC3 | mechanism jitter 0: $MC0"
echo "$(date +%H:%M) $MSG" >> $A/progress.txt
echo "H30C_ADOPT2_END $(date +%H:%M) $MSG" > $M
# segment 3 (its own marker; the adoption marker is already written)
trap - EXIT
[ $KEEP = 1 ] || { echo "H30C_LONG_3_END $(date +%H:%M) skipped: +1000 exit $EX2 < 21.6 (14's stop rule)" > logs/H30C_LONG_3_END.txt; exit 0; }
[ -f $O/STOP3 ] && { echo "H30C_LONG_3_END $(date +%H:%M) skipped: STOP3" > logs/H30C_LONG_3_END.txt; exit 0; }
S3="aborted"; trap 'echo "H30C_LONG_3_END $(date +%H:%M) $S3" > logs/H30C_LONG_3_END.txt' EXIT
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 500 --run_name human_pitched30c_long3 --resume True \
  --load_run $(basename $D2) --checkpoint model_$I2.pt > $O/train_3.txt 2>&1 &
TP=$!; PK=0
while kill -0 $TP 2>/dev/null; do
  G=$(priv human_pitched30c_long3); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'human_pitched30c_long3' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    S3="RAM watchdog: trainer private ${G} GB > 30, killed"; exit 1
  fi
  sleep 30
done
grep -q Traceback $O/train_3.txt && { S3="crash train_3"; exit 1; }
D3=$(ls -td $B/*human_pitched30c_long3 2>/dev/null | head -1); I3=$([ -n "$D3" ] && last $D3)
[ -n "$I3" ] || { S3="segment 3: no checkpoint"; exit 1; }
ROW="+1500 model_$I3"
for j in 0 0.03; do
  TG=h30cL_${I3}_j$j
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D3/model_$I3.pt $SAIM --envs 1024 --seed 0 --tag $TG --timing_noise $j \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$TG.txt 2>&1
  if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
    ROW="$ROW | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label j$j)"
  else ROW="$ROW | j$j: crash or no json"; fi
done
echo "$(date +%H:%M) $ROW" >> $O/progress.txt
S3="$ROW | trainer private peak ${PK} GB"
