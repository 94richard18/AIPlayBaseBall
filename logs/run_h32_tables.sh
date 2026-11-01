#!/bin/bash
# H32 tables (planning 14, 11-01 23:0x): ONE OUTS 800 Hz 0.03-jitter tables of H32 seg-1 model_14995 + percept_filter at batter
# effort 1.0, saim off, vs the E10 bar 16.73% (12500 15.23% + 1.5) with outfield >= 3%. Protocol as run_e10_tables.sh.
# Runs BESIDE H32 segment 2 (starts once logs/h32/train_2.txt exists); each Isaac launch waits for Isaac <= 1, RAM >= 20 GB,
# no film -> logs/H32T_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
DT="--physics_dt 0.00125 --phase_horizon 2.6"
B=logs/rsl_rl/aib1_tee_batting_mimic
C14=${CK:-$B/2026-11-01_20-56-20_human_pitched_h32_1/model_14995.pt}; HI=$(basename $C14 .pt); HI=${HI#model_}; TRIG=${TRIG:-logs/h32/train_2.txt}
O=logs/h32/tables; mkdir -p $O
M=logs/H32T_${HI}_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H32T_${HI}_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
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
  beside
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $3 --envs 1024 --seed 0 --tag $2 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT --contact_log > $O/diag_$2.txt 2>&1
  [ -f logs/diag_bat_contact_$2.json ] && ! grep -q Traceback $O/diag_$2.txt || { MSG="FAILED: hit-rate diag $2 crashed ($(grep -A14 Traceback $O/diag_$2.txt | tail -1))"; exit 1; }
  HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_$2.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
}
tables() {  # $1 ckpt, $2 prefix, $3 hit rate, $4 out list, $5 flags
  : > $4
  for j in 1 2 3; do
    beside
    timeout 1800 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $5 --eval_envs 1024 --timing_noise 0.03 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT > ${2}_$j.txt 2>&1
    grep -q Traceback ${2}_$j.txt && { MSG="FAILED: showcase $(basename $2)_$j crashed ($(grep -A14 Traceback ${2}_$j.txt | tail -1))"; exit 1; }
    grep -q "physics dt 0.001250" ${2}_$j.txt || { MSG="FAILED: showcase $(basename $2)_$j has no 800 Hz line"; exit 1; }
    $P scripts/make_batter_table.py ${2}_$j.txt ${2}_$j.json > /dev/null || { MSG="FAILED: make_batter_table $(basename $2)_$j"; exit 1; }
    echo "t_$j $(oo ${2}_$j.json $3)" >> $4
  done
}
MSG="FAILED: waiting for $TRIG"
until [ -f $TRIG ]; do sleep 60; done
S0=$(date +%s); BAR=16.73
hitrate $C14 h32t_${HI}_j0 "--percept_filter"; H14=$HR
tables $C14 $O/t3_${HI} $H14 $O/t3_${HI}.txt "--percept_filter"
read W14 O14 P14 <<< "$(t3 $O/t3_${HI}.txt)"
MSG="1.0x, saim off, 800 Hz, PH 2.6, jitter 0.03: H32 $HI + percept_filter $W14% outfield $O14% (tables $P14), hit rate $H14, delta vs bar $BAR: $($P -c "print(f'{$W14 - $BAR:+.2f}')") (14496 was 16.57% / 2.91%) | $((($(date +%s) - S0) / 60)) min"
