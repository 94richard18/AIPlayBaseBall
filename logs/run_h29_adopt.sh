#!/bin/bash
# Human29 adoption (planning 14, 10-30 22:3x GO): after SAIM_ADOPT_END, before ArmAim2c. Swing = Human29 model_12799
# (trained on the quadratic-filtered percept; hit 0.908 but slope -0.86). Candidates:
#   h29   = 12799 alone, filter on
#   h29s  = 12799 + the row-9 config (scripted aim, filter, full comp, window 0.60, bound = the SAIM_ADOPT pick)
# (i) diag1024 s0 of h29s (gate + percept + torque lines); h29 reuses logs/diag_bat_contact_h29_12799.json
# (ii) per candidate: 3 showcase tables x 5 ONE OUTS (hit rate = its diag 'slow in zone') + joint seed 6 with recover_rl
#     (pitch columns vs fd_s6 64/64, on the ground <= 2/64, torso tilt at t_ref + 1.5 s = the post-contact stance)
# Adoption: beat the delivery bar (3-table mean >= delivery + 1.5, outfield >= 3%) AND the 12500 + saim 3-table mean
# (SAIM_ADOPT); among passers the higher mean, within 1.5 pts the flatter hit slope. -> logs/h29adopt/winner.txt
# ("<swing ckpt> <tag>", 12500 when nobody passes) for run_armaim2c.sh. Marker logs/H29_ADOPT_END.txt (EXIT trap).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
H29=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-30_21-28-13_human_pitched29_1/model_12799.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2/model_1297.pt
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv; GY=logs/armaim_auth/scripted_gain.json
O=logs/h29adopt; M=logs/H29_ADOPT_END.txt; mkdir -p $O
rm -f $M $O/winner.txt; MSG="aborted"; OUT=""
trap '[ -f $O/winner.txt ] || echo "$SW0 d12500" > $O/winner.txt; echo "H29_ADOPT_END $(date +%H:%M) $MSG | winner: $(cat $O/winner.txt)" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="$OUT || crash $2"; exit 1; }
note() { OUT="$OUT || $1"; echo "$(date +%H:%M) $1" >> $O/progress.txt; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
until [ -f logs/SAIM_ADOPT_END.txt ]; do sleep 60; done
BND=$(grep -o "aim_bound [0-9.]*" logs/saim/progress.txt 2>/dev/null | tail -1 | awk '{print $2}')
[ -n "$BND" ] || BND=0.4
SAIM="--aim_scripted $GY --aim_bound $BND --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full"
# (i) diag 12799 + saim (skipped when this chain already logged it; relaunch 10-30 23:1x)
PREV=$(grep "h29s 12799 + saim bound $BND" $O/progress.txt 2>/dev/null | tail -1)
if [ -n "$PREV" ] && [ -f logs/diag_bat_contact_h29s.json ]; then OUT=" || (reused) ${PREV#* }"; else
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H29 --armaim $RES $SAIM --percept_filter \
  --aim_scripted_dump $O/pair_h29s.npz --envs 1024 --seed 0 --tag h29s --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 \
  --release_csv $R --contact_log > $O/diag_h29s.txt 2>&1
grep -q Traceback $O/diag_h29s.txt && fail $O/diag_h29s.txt diag_h29s
[ -f logs/diag_bat_contact_h29s.json ] || { MSG="$OUT || h29s: no json"; exit 1; }
note "$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h29s.json --stage 300 --label "h29s 12799 + saim bound $BND" | tr '\n' ' ') | $($P scripts/armaim_pair.py --wet $O/pair_h29s.npz --label h29s | grep -E '\(c\)|total' | tr '\n' ' ') | $(grep 'window torque' $O/diag_h29s.txt | tail -1)"
fi
# (ii) tables + joint per candidate
for C in h29 h29s; do
  if [ $C = h29 ]; then J=logs/diag_bat_contact_h29_12799.json; SX="--percept_filter"; JX="--percept_filter"
  else J=logs/diag_bat_contact_h29s.json; SX="--armaim $RES $SAIM --percept_filter"; JX="--saim $GY --saim_bound $BND --saim_window 0.60 --saim_comp full --percept_filter"; fi
  HR=$($P -c "import json; print(json.load(open('$J'))['summary']['by_speed']['slow in zone']['hit_rate'])")
  : > $O/t3_$C.txt
  for j in 1 2 3; do
    idle
    timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $H29 $SX --eval_envs 1024 --timing_noise 0.03 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_${C}_$j.txt 2>&1
    grep -q Traceback $O/table_${C}_$j.txt && fail $O/table_${C}_$j.txt table_${C}_$j
    $P scripts/make_batter_table.py $O/table_${C}_$j.txt $O/table_${C}_$j.json > /dev/null || { MSG="$OUT || $C table $j failed"; exit 1; }
    echo "${C}_$j $(oo $O/table_${C}_$j.json $HR)" >> $O/t3_$C.txt
  done
  # joint s6: h29s runs variant A (tau 0.05 decay) and B (planning 14, 10-30: on a miss, smoothstep the offset / comp to 0
  # over 0.2 s from the plate crossing); JOINT OK when either passes. Each run dumps the hand-off pose (trig_nn.py vs the bank).
  VS="A"; [ $C = h29s ] && VS="A B"
  for V in $VS; do
    JF=joint_$C; [ $V = B ] && JF=joint_${C}B
    VX=""; [ $V = B ] && VX="--saim_miss_blend 0.2"
    # h29s B was run by hand at 23:3x (same command, the GPU smoke of the new flags); reused when present
    if [ $V = B ] && [ -f $O/$JF.csv ] && [ -f $O/trig_$JF.npz ]; then note "$JF: reused the 23:3x run"; else
    idle
    timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H29 --envs 64 --seed 6 --after_done recover_rl \
      --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX $VX --trig_dump $O/trig_$JF.npz \
      --csv $O/$JF.csv > $O/$JF.txt 2>&1
    grep -q Traceback $O/$JF.txt && fail $O/$JF.txt $JF
    fi
    note "$($P scripts/trig_nn.py --trig $O/trig_$JF.npz --csv $O/$JF.csv --label $JF)"
  done
  JB=""; JL=""
  for V in $VS; do
    JF=joint_$C; [ $V = B ] && JF=joint_${C}B
  ST=$($P - <<PY
import csv, statistics as st
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
r = [list(map(float, l.split()[1:])) for l in open("$O/t3_$C.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$O/$JF.csv')))
cols = ['clip', 'kmh', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b)); g = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b)
tt = sorted(f(x['torso_tilt_1p5_deg']) for x in b if f(x['torso_tilt_1p5_deg']) == f(x['torso_tilt_1p5_deg']))
hits = sum(x['hit'] == 'True' for x in b)
print(f"T3 {st.mean(w):.1f} OF {st.mean(o):.1f} tables {', '.join(f'{x:.1f}' for x in w)} | joint s6 recover_rl: pitch columns {same}/64, on the ground {g}/64, "
      f"hits {hits}/64, torso tilt +1.5 s med {tt[len(tt) // 2] if tt else float('nan'):.1f} q90 {tt[int(0.9 * (len(tt) - 1))] if tt else float('nan'):.1f} deg "
      f"JOINT {'OK' if same == 64 and g <= 2 else 'BAD'}")
PY
)
    note "$JF variant $V: $ST"
    case "$ST" in *"JOINT OK"*) [ -z "$JB" ] && JB="$ST (variant $V)";; esac
    JL="$ST"
  done
  [ -n "$JB" ] && JL="$JB"
  note "$C (hit rate $HR): $JL"
done
# adoption
PICK=$($P - <<PY
import json, re, statistics as st, numpy as np
def rows(f, pre):
    r = [list(map(float, l.split()[1:])) for l in open(f) if l.startswith(pre) and len(l.split()) == 11]
    return st.mean(st.mean(v[0::2]) for v in r)
dw = rows("logs/t3_results.txt", "e15_"); bar = dw + 1.5
m = re.search(r"ONE OUTS 3-table mean ([0-9.]+)%", open("logs/saim/progress.txt").read())
s12500 = float(m.group(1)) if m else 0.0
def slope(t):
    P = {k: np.array(v, dtype=float) for k, v in json.load(open(f"logs/diag_bat_contact_{t}.json"))["per_pitch"].items()}
    z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10); h = P["hit"].astype(bool) & z
    ax = np.where(np.isfinite(P["dx_any"]), P["dx_any"], P["x_on_bat"] - 0.7112); k = h & np.isfinite(ax)
    return np.polyfit(P["plate_y"][k], ax[k], 1)[0]
cand = {}
for c, t in (("h29", "h29_12799"), ("h29s", "h29s")):
    line = [l for l in open("$O/progress.txt") if f" {c} (hit rate" in l][-1]
    w, o = float(re.search(r"T3 ([0-9.]+)", line).group(1)), float(re.search(r"OF ([0-9.]+)", line).group(1))
    ok = w >= bar and o >= 3.0 and w > s12500 and "JOINT OK" in line
    cand[c] = (w, o, slope(t), ok)
ps = [c for c in cand if cand[c][3]]
if ps:
    top = max(ps, key=lambda c: cand[c][0])
    near = [c for c in ps if cand[top][0] - cand[c][0] <= 1.5]
    win = max(near, key=lambda c: cand[c][2])
else:
    win = "none"
print(win, f"bar {bar:.1f}%, 12500+saim {s12500:.1f}% | " + "; ".join(f"{c}: {w:.1f}% OF {o:.1f}% slope {s:+.2f} {'pass' if ok else 'fail'}" for c, (w, o, s, ok) in cand.items()))
PY
)
read WIN REST <<< "$PICK"
note "adoption: winner $WIN | $REST"
case $WIN in h29|h29s) echo "$H29 $WIN" > $O/winner.txt;; *) echo "$SW0 d12500" > $O/winner.txt;; esac
MSG="$OUT"
