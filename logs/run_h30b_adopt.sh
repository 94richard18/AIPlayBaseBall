#!/bin/bash
# Human30b adoption (planning 14, 10-31 06:53): model_12799 of human_pitched30b_1 with the row-9 scripted aim (saim bound
# 0.4, full comp, filtered percept, window 0.60). 3 showcase tables x 5 ONE OUTS (hit rate = diag h30b_12799 'slow in
# zone'), joint s6 with recover_rl = R3 model_1596 + saim, then the ONE OUTS mechanism split (CPU, run from the scratchpad
# copy so idle() of other chains is not blocked). Adoption: 3-table mean >= 19.1% AND outfield >= 3% AND s6 on the ground
# <= 2/64 (pitch columns 64/64). -> logs/H30B_ADOPT_END.txt "ADOPT" / "NO" (EXIT trap). Chain stops there.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
H=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-31_06-09-39_human_pitched30b_1/model_12799.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov3/recover_ckpt.txt)
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv; GY=logs/armaim_auth/scripted_gain.json
SX="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
MECH=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad/one_outs_mechanism_run.py
O=logs/h30badopt; M=logs/H30B_ADOPT_END.txt; mkdir -p $O
rm -f $M; MSG="aborted"
trap 'echo "H30B_ADOPT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
[ -f "$H" ] && [ -f "$RC" ] || { MSG="missing ckpt ($H / $RC)"; exit 1; }
HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_h30b_12799.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
: > $O/t3.txt
for j in 1 2 3; do
  idle
  timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $H $SX --eval_envs 1024 --timing_noise 0.03 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_$j.txt 2>&1
  grep -q Traceback $O/table_$j.txt && fail $O/table_$j.txt table_$j
  $P scripts/make_batter_table.py $O/table_$j.txt $O/table_$j.json > /dev/null || { MSG="table $j: make_batter_table failed"; exit 1; }
  echo "h30b_$j $(oo $O/table_$j.json $HR)" >> $O/t3.txt
done
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --trig_dump $O/trig_joint.npz \
  --csv $O/joint_s6.csv > $O/joint_s6.txt 2>&1
grep -q Traceback $O/joint_s6.txt && fail $O/joint_s6.txt joint_s6
ST=$($P - <<PY
import csv, statistics as st
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
r = [list(map(float, l.split()[1:])) for l in open("$O/t3.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$O/joint_s6.csv')))
cols = ['clip', 'kmh', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b)); g = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b)
tt = sorted(f(x['torso_tilt_1p5_deg']) for x in b if f(x['torso_tilt_1p5_deg']) == f(x['torso_tilt_1p5_deg']))
hits = sum(x['hit'] == 'True' for x in b)
ok = len(w) == 3 and st.mean(w) >= 19.1 and st.mean(o) >= 3.0 and same == 64 and g <= 2
print(("ADOPT" if ok else "NO") + f" T3 {st.mean(w):.1f}% (tables {', '.join(f'{x:.1f}' for x in w)}) OF {st.mean(o):.1f}% (bar 19.1 / 3.0) | "
      f"joint s6 (recover {'$RC'.split('/')[-2]}): pitch columns {same}/64, on the ground {g}/64, hits {hits}/64, torso tilt +1.5 s med "
      f"{tt[len(tt) // 2] if tt else float('nan'):.1f} deg")
PY
)
echo "$(date +%H:%M) $ST" >> $O/progress.txt
cp scripts/one_outs_mechanism.py $MECH
MC=$(cd C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad && \
  $P - <<PY 2>&1 | tail -1
import os, runpy, sys
os.chdir(r"D:/MyProject/AIPlayBaseBall"); sys.path.insert(0, r"D:/MyProject/AIPlayBaseBall")
sys.argv = ["mech", "--name", "h30b", "--hit_rate", "$HR", "--tables", "$O/table_1.json", "$O/table_2.json", "$O/table_3.json"]
runpy.run_path(r"$MECH", run_name="__main__")
PY
)
MSG="$ST | hit rate $HR | mechanism: $MC"
