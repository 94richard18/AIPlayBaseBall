#!/bin/bash
# After Human30f (planning 14, 10-31 16:33). Waits for H30F_500_END (newer than this chain's start), then:
# (a) plain 12500 (no saim, the delivery config: aim_offset_z 0, P3c releases, effort 1.5) jitter-0 reference, 3 showcase
#     tables x 5 ONE OUTS (hit rate = diag h17_12500 'slow in zone', as the delivery's b_e15 run) -> REF12500_J0_END
#     (tables kept in logs/ref12500_j0/ for reuse);
# (b) if S7_13298_END says s6 + s7 ground > 4 (more than 2/64 per seed on average): R4 = recover bank from the model_13298
#     swing + saim (as R3), resume R3 model_1596 +300 on AIB-Recover-R2-v0, then joint s6 + s7 with it -> RECOV4_END
#     (PASS = s6 pitch columns 64/64 and s6 + s7 ground <= 4 and torso tilt +1.5 s <= 30 deg on >= 56/64 in each seed);
#     else RECOV4_END "skipped". One Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
unset AIB_FIX_STD AIB_STD_FROM
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
GY=logs/armaim_auth/scripted_gain.json
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
H=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched30c_long1 | head -1)/model_13298.pt
O=logs/ref12500_j0; Q=logs/recov4; mkdir -p $O $Q
M=logs/REF12500_J0_END.txt; rm -f $M logs/RECOV4_END.txt; MSG="aborted"
touch $Q/chain.stamp
trap '[ -n "$M" ] && echo "$(basename $M .txt) $(date +%H:%M) $MSG" > $M && cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
joint() {  # $1 seed, $2 recover ckpt, $3 out prefix
  idle
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $2 --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --trig_dump ${3}.npz --csv ${3}.csv > ${3}.txt 2>&1
  grep -q Traceback ${3}.txt && fail ${3}.txt $(basename $3)
}
until [ -f logs/H30F_500_END.txt ] && [ logs/H30F_500_END.txt -nt $Q/chain.stamp ]; do sleep 60; done
# (a)
HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_h17_12500.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
: > $O/t0.txt
for j in 1 2 3; do
  idle
  timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $H0 --eval_envs 1024 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/t0_$j.txt 2>&1
  grep -q Traceback $O/t0_$j.txt && fail $O/t0_$j.txt t0_$j
  $P scripts/make_batter_table.py $O/t0_$j.txt $O/t0_$j.json > /dev/null || { MSG="t0 $j: make_batter_table failed"; exit 1; }
  echo "t_$j $(oo $O/t0_$j.json $HR)" >> $O/t0.txt
done
MSG=$($P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$O/t0.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"plain 12500 (no saim, delivery config) jitter 0: T3 {st.mean(w):.1f}% (tables {', '.join(f'{x:.1f}' for x in w)}) OF {st.mean(o):.1f}% | hit rate $HR")
PY
)
echo "REF12500_J0_END $(date +%H:%M) $MSG" > $M
# (b)
M=logs/RECOV4_END.txt; MSG="aborted"
S7=$(cat logs/S7_13298_END.txt 2>/dev/null)
N=$(echo "$S7" | grep -o "= [0-9]* (R4 needed" | grep -o "[0-9]*")
[ -n "$N" ] || { MSG="skipped: no s6 + s7 count in S7_13298_END ($S7)"; exit 0; }
[ "$N" -gt 4 ] || { MSG="skipped: s6 + s7 ground $N <= 4"; exit 0; }
BK=logs/recover_bank_r4_s0.npz
idle
timeout 1800 $P -u scripts/recover_bank.py --headless --checkpoint $H --armaim --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 \
  --aim_obs_comp --aim_obs_comp_full --percept_filter --envs 2048 --seed 0 --out $BK > $Q/bank.txt 2>&1
grep -q Traceback $Q/bank.txt && fail $Q/bank.txt bank
[ -f $BK ] || { MSG="bank: no npz"; exit 1; }
BL=$(grep '^\[bank\] seed' $Q/bank.txt)
idle
AIB_RECOVER_BANK=$BK timeout 5400 $P -u scripts/train.py --task AIB-Recover-R2-v0 --headless --num_envs 4096 --max_iterations 300 \
  --run_name h25r4 --resume True --load_run 2026-10-31_01-47-37_h25r3 --checkpoint model_1596.pt > $Q/train.txt 2>&1
grep -q Traceback $Q/train.txt && fail $Q/train.txt train
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r4 2>/dev/null | head -1)
CL=$([ -n "$RUN" ] && ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
[ -n "$CL" ] || { MSG="train: no checkpoint"; exit 1; }
joint 6 $RUN/model_$CL.pt $Q/joint_s6
joint 7 $RUN/model_$CL.pt $Q/joint_s7
MSG="$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b6 = list(csv.DictReader(open('$Q/joint_s6.csv'))); b7 = list(csv.DictReader(open('$Q/joint_s7.csv')))
cols = ['clip', 'kmh', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b6))
g = lambda b: sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b); h = lambda b: sum(x['hit'] == 'True' for x in b)
t30 = lambda b: sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b); fl = lambda b: sum(x['batter_fell'] == 'True' for x in b)
ok = same == 64 and g(b6) + g(b7) <= 4 and t30(b6) >= 56 and t30(b7) >= 56  # 14 (17:06): tilt +1.5 s <= 30 deg on >= 56/64
print(("PASS" if ok else "FAIL") + f" R4 (model_13298 + saim + recover h25r4 model_$CL): s6 pitch columns {same}/64, ground s6 {g(b6)}/64 s7 {g(b7)}/64, "
      f"tilt +1.5 s <= 30 deg s6 {t30(b6)}/64 s7 {t30(b7)}/64 (bar 56), batter_fell flag s6 {fl(b6)} s7 {fl(b7)}, hits s6 {h(b6)} s7 {h(b7)}")
PY
) | $BL | ckpt $RUN/model_$CL.pt"
case "$MSG" in PASS*) echo "$RUN/model_$CL.pt" > $Q/recover_ckpt.txt;; esac
