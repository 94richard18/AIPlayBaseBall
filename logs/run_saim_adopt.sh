#!/bin/bash
# Scripted arm aim adoption (planning 14, 10-30 21:4x): row 9 = frozen 12500 @ 1.5x + scripted aim on the quadratic-filtered
# percept + full obs comp (q + bat), window 0.60. Waits for the Human29 chain to release the GPU (logs/H29_1000_END.txt,
# written by its EXIT trap on every outcome), then:
#  0. joint zero-diff: joint_smoke seed 6 after_done policy, every flag off, vs logs/falldiag2/fd_s6.csv (all common columns 64/64)
#  1. row 10 = row 9 + aim_bound 0.6 (diag1024 seed 0). Adopt the bound with the flatter hit slope among those with hit >= 0.45
#     and |axis_off| <= 0.08 (row 9: -0.40 / 0.526 / 0.071)
#  2. 3 showcase tables (1024, timing noise 0.03 as run_tables3.sh) -> 5 seeds x 3 ONE OUTS, hit rate = that row's diag
#     'slow in zone'; bar = delivery 3-table mean (logs/t3_results.txt e15_*) + 1.5 and outfield >= 3%
#  3. joint seed 6 with recover_rl (model_1297) + saim: pitch columns vs fd_s6 64/64, on the ground (min_pelvis_z_2s < 0.35) <= 2/64
# Marker logs/SAIM_ADOPT_END.txt (EXIT trap). One Isaac at a time: Isaac 0 twice 60 s apart before every launch.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2/model_1297.pt
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv; GY=logs/armaim_auth/scripted_gain.json
O=logs/saim; M=logs/SAIM_ADOPT_END.txt; mkdir -p $O
rm -f $M; MSG="aborted"; OUT=""
trap 'echo "SAIM_ADOPT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="$OUT || crash $2"; exit 1; }
note() { OUT="$OUT || $1"; echo "$(date +%H:%M) $1" >> $O/progress.txt; }
until [ -f logs/H29_1000_END.txt ]; do sleep 60; done
# 0. joint zero-diff, flags off
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $SW --envs 64 --seed 6 --after_done policy \
  --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --csv $O/zero_s6.csv > $O/zero_s6.txt 2>&1
grep -q Traceback $O/zero_s6.txt && fail $O/zero_s6.txt zero_s6
Z0=$($P -c "
import csv
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$O/zero_s6.csv')))
cols = [c for c in a[0] if c in b[0]]
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b)) if len(a) == len(b) else 0
bad = sorted({c for x, y in zip(a, b) for c in cols if x[c] != y[c]})
print(('OK' if same == len(a) == 64 else 'BAD') + f' flags-off joint s6 vs falldiag2: {same}/{len(a)} rows identical over {len(cols)} columns' + (f' (differ: {bad[:8]})' if bad else ''))")
note "$Z0"
case "$Z0" in OK*) ;; *) MSG="$OUT"; exit 1;; esac
# 1. row 10
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted $GY --aim_window 0.60 \
  --aim_bound 0.6 --percept_filter --aim_obs_comp --aim_obs_comp_full --aim_scripted_dump $O/pair_r10.npz --envs 1024 --seed 0 \
  --tag r10 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_r10.txt 2>&1
grep -q Traceback $O/diag_r10.txt && fail $O/diag_r10.txt r10
[ -f logs/diag_bat_contact_r10.json ] || { MSG="$OUT || r10: no json"; exit 1; }
G10=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_r10.json --stage 300 --label "row10 bound 0.6" | tr '\n' ' ')
note "$G10 | $(grep 'sign +1' $O/diag_r10.txt | tail -1)"
PICK=$($P -c "
import json, numpy as np
def m(t):
    P = {k: np.array(v, dtype=float) for k, v in json.load(open(f'logs/diag_bat_contact_{t}.json'))['per_pitch'].items()}
    z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10); h = P['hit'].astype(bool) & z
    ax = np.where(np.isfinite(P['dx_any']), P['dx_any'], P['x_on_bat'] - 0.7112)
    k = h & np.isfinite(ax)
    return np.polyfit(P['plate_y'][k], ax[k], 1)[0], h.sum() / z.sum(), np.nanmedian(np.abs(ax[k]))
ok = {t: (s, hit, a) for t, (s, hit, a) in ((t, m(t)) for t in ('r9wet', 'r10')) if hit >= 0.45 and a <= 0.08}
best = max(ok, key=lambda t: ok[t][0]) if ok else 'r9wet'
print(best, 0.6 if best == 'r10' else 0.4, ' '.join(f'{t}: slope {s:+.2f} hit {h:.3f} ax {a:.3f}' for t, (s, h, a) in ok.items()))")
read TAG BND REST <<< "$PICK"
HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_$TAG.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
note "pick $TAG (aim_bound $BND, in-zone hit rate $HR): $REST"
# 2. 3 showcase tables x 5 ONE OUTS seeds
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
: > $O/t3.txt
for j in 1 2 3; do
  idle
  timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $SW --armaim $RES --aim_scripted $GY \
    --aim_bound $BND --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter --eval_envs 1024 --timing_noise 0.03 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_$j.txt 2>&1
  grep -q Traceback $O/table_$j.txt && fail $O/table_$j.txt table_$j
  $P scripts/make_batter_table.py $O/table_$j.txt $O/table_$j.json > /dev/null || { MSG="$OUT || table $j: make_batter_table failed"; exit 1; }
  echo "saim_$j $(oo $O/table_$j.json $HR)" >> $O/t3.txt
done
OO=$($P - <<PY
import statistics as st
def rows(f, pre):
    r = [list(map(float, l.split()[1:])) for l in open(f) if l.startswith(pre) and len(l.split()) == 11]
    return [st.mean(v[0::2]) for v in r], [st.mean(v[1::2]) for v in r]
dw, do = rows("logs/t3_results.txt", "e15_"); sw, so = rows("$O/t3.txt", "saim_")
bar = st.mean(dw) + 1.5
ok = len(sw) == 3 and st.mean(sw) >= bar and st.mean(so) >= 3.0
print(("PASS" if ok else "FAIL") + f" ONE OUTS 3-table mean {st.mean(sw):.1f}% (tables {', '.join(f'{x:.1f}' for x in sw)}), outfield "
      f"{st.mean(so):.1f}% | delivery {st.mean(dw):.1f}% (outfield {st.mean(do):.1f}%), bar {bar:.1f}% + outfield >= 3%")
PY
)
note "$OO"
# 3. joint seed 6, recover_rl + saim
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $SW --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --saim $GY --saim_bound $BND --saim_window 0.60 \
  --saim_comp full --percept_filter --csv $O/joint_s6.csv > $O/joint_s6.txt 2>&1
grep -q Traceback $O/joint_s6.txt && fail $O/joint_s6.txt joint_s6
J6=$($P -c "
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$O/joint_s6.csv')))
cols = ['clip', 'kmh', 'plate_y', 'plate_z', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b))
g = sum(f(r['min_pelvis_z_2s']) < 0.35 for r in b); hits = sum(r['hit'] == 'True' for r in b)
ev = sorted(f(r['exit_mps']) for r in b if r['hit'] == 'True')
ok = same == 64 and g <= 2
print(('PASS' if ok else 'FAIL') + f' joint s6 recover_rl + saim: pitch columns {same}/64, on the ground {g}/64, hits {hits}/64 (swing alone {sum(r[\"hit\"] == \"True\" for r in a)}), '
      f'exit median {ev[len(ev) // 2] if ev else float(\"nan\"):.1f}')")
note "$J6"
MSG="$OUT"
