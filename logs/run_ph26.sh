#!/bin/bash
# Phase-horizon re-table (planning 14, 10-31 17:20), conditional: runs only after TRAINLIB_DIAG_END (newer than this chain's
# start) AND logs/ph26/GO (14 releases it if row B or C of the (f) falsifier closes the gap). Every diag / showcase / joint
# here sets --phase_horizon 2.6 (the swing policy's phase input on the training scale; all runs so far fed t / 9 s or t / 30 s):
#  1) model_13298 + row 9 saim: diag j0 + j0.03 (hit rates), tables 0.03 x 3 and jitter 0 x 3, 5 ONE OUTS seeds each;
#  2) plain 12500 (no saim, delivery config): diag j0 + tables j0 x 3 (replaces REF12500_J0);
#  3) row 9 reference (12500 + saim): diag j0 + tables j0 x 3;
#  4) joint s6 + s7 of 13298 (R3 recover + saim): on the ground (min pelvis z 2 s < 0.35), batter_fell flag, hits, tilt.
# -> logs/PH26_END.txt (EXIT trap). One Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
PH="--phase_horizon 2.6"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov3/recover_ckpt.txt)
H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
H=$(ls -td $B/*human_pitched30c_long1 | head -1)/model_13298.pt
O=logs/ph26; mkdir -p $O
M=logs/PH26_END.txt; rm -f $M; MSG="aborted"; RES_ALL=""
touch $O/chain.stamp
trap 'echo "PH26_END $(date +%H:%M) $MSG$RES_ALL" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
hr() { $P -c "import json; print(json.load(open('$1'))['summary']['by_speed']['slow in zone']['hit_rate'])"; }
diag() {  # $1 ckpt, $2 extra (saim or empty), $3 jitter, $4 tag -> prints the hit rate
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $2 --envs 1024 --seed 0 --tag $4 --timing_noise $3 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $PH --contact_log > $O/diag_$4.txt 2>&1
  grep -q Traceback $O/diag_$4.txt && fail $O/diag_$4.txt diag_$4
  [ -f logs/diag_bat_contact_$4.json ] || { MSG="diag $4: no json"; exit 1; }
}
tables() {  # $1 ckpt, $2 extra, $3 jitter, $4 prefix, $5 hit rate -> prints "T3 mean, OF, tables"
  : > ${4}.txt
  for j in 1 2 3; do
    idle
    timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $2 --eval_envs 1024 --timing_noise $3 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $PH > ${4}_$j.txt 2>&1
    grep -q Traceback ${4}_$j.txt && fail ${4}_$j.txt $(basename $4)_$j
    $P scripts/make_batter_table.py ${4}_$j.txt ${4}_$j.json > /dev/null || { MSG="$(basename $4) $j: make_batter_table failed"; exit 1; }
    echo "t_$j $(oo ${4}_$j.json $5)" >> ${4}.txt
  done
  $P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("${4}.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"T3 {st.mean(w):.1f}% (tables {', '.join(f'{x:.1f}' for x in w)}) OF {st.mean(o):.1f}% | hit rate $5")
PY
}
joint() {  # $1 seed
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX $PH --trig_dump $O/trig_joint_s$1.npz \
    --csv $O/joint_s$1.csv > $O/joint_s$1.txt 2>&1
  grep -q Traceback $O/joint_s$1.txt && fail $O/joint_s$1.txt joint_s$1
}
until [ -f logs/TRAINLIB_DIAG_END.txt ] && [ logs/TRAINLIB_DIAG_END.txt -nt $O/chain.stamp ] && [ -f $O/GO ]; do sleep 60; done
# 1) 13298
diag $H "$SAIM" 0 ph26_13298_j0; diag $H "$SAIM" 0.03 ph26_13298_j003
G0=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_ph26_13298_j0.json --stage 300 --label j0 | head -1 | cut -c1-160)
RES_ALL=" | 13298 diag j0: $($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_ph26_13298_j0.json --label j0) ; $G0"
MSG="13298 tables"
RES_ALL="$RES_ALL | 13298 0.03: $(tables $H "$SAIM" 0.03 $O/t13298_003 $(hr logs/diag_bat_contact_ph26_13298_j003.json))"
RES_ALL="$RES_ALL | 13298 j0: $(tables $H "$SAIM" 0 $O/t13298_j0 $(hr logs/diag_bat_contact_ph26_13298_j0.json))"
# 2) plain 12500
MSG="12500 plain"
diag $H0 "" 0 ph26_12500_plain_j0
RES_ALL="$RES_ALL | plain 12500 j0: $(tables $H0 "" 0 $O/t12500_j0 $(hr logs/diag_bat_contact_ph26_12500_plain_j0.json))"
# 3) row 9
MSG="row 9"
diag $H0 "$SAIM" 0 ph26_12500_row9_j0
RES_ALL="$RES_ALL | row 9 j0: $(tables $H0 "$SAIM" 0 $O/t12500r9_j0 $(hr logs/diag_bat_contact_ph26_12500_row9_j0.json))"
# 4) joint
MSG="joint"
joint 6; joint 7
RES_ALL="$RES_ALL | $($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
out = []
for s in (6, 7):
    b = list(csv.DictReader(open('$O/joint_s%d.csv' % s)))
    g = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b); fl = sum(x['batter_fell'] == 'True' for x in b)
    t30 = sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b); h = sum(x['hit'] == 'True' for x in b)
    out.append(f"joint s{s} 13298 ph2.6: ground {g}/{len(b)}, batter_fell {fl}, tilt<=30 {t30}, hits {h}")
print('; '.join(out))
PY
)"
MSG="done (phase_horizon 2.6 everywhere)"
