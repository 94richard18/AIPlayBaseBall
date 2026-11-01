#!/bin/bash
# H32 jitter-0 tables (planning 14, 11-02 02:4x, protocol addition): adoption tables at BOTH jitter 0 (= the joint game, which hands
# the batter the engine's true arrival time, 10-21) and 0.03 (robustness); the bar is re-derived per column with the same rule
# (plain 12500 @1.0x no-saim + 1.5, outfield >= 3 %). This script: jitter-0 showcase tables (otherwise the E10 protocol) of plain
# 12500, 15494 + filter, 14995 + filter, beside H32 segment 3 (starts once logs/h32/train_3.txt exists; each Isaac launch at
# Isaac <= 1, RAM >= 20 GB, no film); the jitter-0 hit-rate diag json is reused when it exists -> logs/H32T0_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
DT="--physics_dt 0.00125 --phase_horizon 2.6"
B=logs/rsl_rl/aib1_tee_batting_mimic
C15=$B/2026-11-02_00-24-42_human_pitched_h32_2/model_15494.pt; C149=$B/2026-11-01_20-56-20_human_pitched_h32_1/model_14995.pt
C12=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
O=logs/h32/tables0; mkdir -p $O
M=logs/H32T0_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H32T0_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
beside() { until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 \
  | awk '{for (i = 1; i < NF; i++) { if ($i == "batter") b = $(i + 1); if ($i == "outfield_hit") o = $(i + 1) }} END { gsub(/[%,]/, "", b); gsub(/[%,]/, "", o); printf "%s %s ", b, (o == "" ? "0" : o) }'; done; echo; }
t3() { $P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$1") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"{st.mean(w):.2f} {st.mean(o):.2f} {','.join(f'{x:.1f}' for x in w)}" if len(r) == 3 else "nan nan incomplete")
PY
}
hitrate() {  # $1 ckpt, $2 tag, $3 flags -> HR
  [ -f logs/diag_bat_contact_$2.json ] || { beside
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $3 --envs 1024 --seed 0 --tag $2 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT --contact_log > $O/diag_$2.txt 2>&1; }
  [ -f logs/diag_bat_contact_$2.json ] && ! grep -q Traceback $O/diag_$2.txt 2>/dev/null || { MSG="FAILED: hit-rate diag $2 crashed ($(grep -A14 Traceback $O/diag_$2.txt | tail -1))"; exit 1; }
  HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_$2.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
}
tables() {  # $1 ckpt, $2 prefix, $3 hit rate, $4 out list, $5 flags
  : > $4
  for j in 1 2 3; do
    beside
    timeout 1800 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $5 --eval_envs 1024 --timing_noise 0 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT > ${2}_$j.txt 2>&1
    grep -q Traceback ${2}_$j.txt && { MSG="FAILED: showcase $(basename $2)_$j crashed ($(grep -A14 Traceback ${2}_$j.txt | tail -1))"; exit 1; }
    grep -q "physics dt 0.001250" ${2}_$j.txt || { MSG="FAILED: showcase $(basename $2)_$j has no 800 Hz line"; exit 1; }
    $P scripts/make_batter_table.py ${2}_$j.txt ${2}_$j.json > /dev/null || { MSG="FAILED: make_batter_table $(basename $2)_$j"; exit 1; }
    echo "t_$j $(oo ${2}_$j.json $3)" >> $4
  done
}
MSG="FAILED: waiting for logs/h32/train_3.txt"
until [ -f logs/h32/train_3.txt ]; do sleep 60; done
S0=$(date +%s)
hitrate $C12 e10t_12500_j0 ""; H12=$HR
tables $C12 $O/t3_12500 $H12 $O/t3_12500.txt ""
read W12 O12 P12 <<< "$(t3 $O/t3_12500.txt)"
BAR=$($P -c "print(f'{$W12 + 1.5:.2f}')")
hitrate $C15 h32t_15494_j0 "--percept_filter"; H15=$HR
tables $C15 $O/t3_15494 $H15 $O/t3_15494.txt "--percept_filter"
read W15 O15 P15 <<< "$(t3 $O/t3_15494.txt)"
hitrate $C149 h32t_14995_j0 "--percept_filter"; H149=$HR
tables $C149 $O/t3_14995 $H149 $O/t3_14995.txt "--percept_filter"
read W149 O149 P149 <<< "$(t3 $O/t3_14995.txt)"
d() { $P -c "print(f'{$1 - $BAR:+.2f}')"; }
MSG="1.0x, saim off, 800 Hz, PH 2.6, JITTER 0: plain 12500 $W12% outfield $O12% (tables $P12), hit rate $H12 -> jitter-0 BAR $BAR% | 15494 + filter $W15% outfield $O15% (tables $P15), hit rate $H15, $(d $W15) | 14995 + filter $W149% outfield $O149% (tables $P149), hit rate $H149, $(d $W149) | jitter-0.03 column: bar 16.73, 15494 13.92 / 6.56, 14995 14.74 / 5.26 | $((($(date +%s) - S0) / 60)) min"
