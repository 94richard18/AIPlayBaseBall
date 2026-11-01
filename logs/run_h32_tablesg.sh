#!/bin/bash
# H32 game-scene tables (planning 14, 11-02 03:0x, protocol, logged): the adoption table = ONE OUTS 800 Hz, PH 2.6, percept_filter,
# saim off, REAL plate height, the 4 joint clips, P3c release csv (the scene the diag tracks within ~1 deg of the joint game); two
# jitter columns (0.03 robustness, 0 = the game), each bar = plain 12500 @1.0x in the same scene + 1.5, outfield >= 3 %. The
# re-centred all-clip tables (bar 16.73, 14.74, 13.92) are history. Order: 12500 @0.03, 15494 @0.03, 12500 @0, 15494 @0,
# 14995 @0.03, 14995 @0; hit rate = the existing jitter-0 game-scene diag (real height, 4 clips). Beside H32 segment 3 (starts
# once logs/h32/train_3.txt exists; each Isaac launch at Isaac <= 1, RAM >= 20 GB, no film). Partial lines ->
# logs/h32/tables_game.txt; marker logs/H32TG_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
DT="--physics_dt 0.00125 --phase_horizon 2.6"
B=logs/rsl_rl/aib1_tee_batting_mimic
C15=$B/2026-11-02_00-24-42_human_pitched_h32_2/model_15494.pt; C149=$B/2026-11-01_20-56-20_human_pitched_h32_1/model_14995.pt
C12=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
O=logs/h32/tables_game; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"; mkdir -p $O
M=logs/H32TG_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H32TG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
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
tables() {  # $1 ckpt, $2 prefix, $3 hit rate, $4 out list, $5 flags, $6 jitter
  : > $4
  for j in 1 2 3; do
    beside
    timeout 1800 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $5 --eval_envs 1024 --timing_noise $6 --real_height --clips $C4 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT > ${2}_$j.txt 2>&1
    grep -q Traceback ${2}_$j.txt && { MSG="FAILED: showcase $(basename $2)_$j crashed ($(grep -A14 Traceback ${2}_$j.txt | tail -1))"; exit 1; }
    grep -q "real_height True" ${2}_$j.txt || { MSG="FAILED: showcase $(basename $2)_$j not in the game scene"; exit 1; }
    grep -q "physics dt 0.001250" ${2}_$j.txt || { MSG="FAILED: showcase $(basename $2)_$j has no 800 Hz line"; exit 1; }
    $P scripts/make_batter_table.py ${2}_$j.txt ${2}_$j.json > /dev/null || { MSG="FAILED: make_batter_table $(basename $2)_$j"; exit 1; }
    echo "t_$j $(oo ${2}_$j.json $3)" >> $4
  done
}
hr() { $P -c "import json; print(json.load(open('logs/diag_bat_contact_$1.json'))['summary']['by_speed']['slow in zone']['hit_rate'])"; }
G=logs/h32/tables_game.txt
MSG="FAILED: waiting for logs/h32/train_3.txt"
until [ -f logs/h32/train_3.txt ]; do sleep 60; done
S0=$(date +%s); H12=$(hr e10_12500); H15=$(hr h32_15494); H149=$(hr h32_14995)
echo "$(date +%H:%M) game scene (real height, 4 clips), hit rates (jitter-0 diag): 12500 $H12, 15494 $H15, 14995 $H149" >> $G
one() {  # $1 ckpt, $2 name, $3 hit rate, $4 flags, $5 jitter -> W O P
  tables $1 $O/t3_${2}_j$5 $3 $O/t3_${2}_j$5.txt "$4" $5
  read W O_ PT <<< "$(t3 $O/t3_${2}_j$5.txt)"
  echo "$(date +%H:%M) $2 @$5: $W% outfield $O_% (tables $PT)" >> $G
}
d() { $P -c "print(f'{$1 - $2:+.2f}')"; }
b() { $P -c "print(f'{$1 + 1.5:.2f}')"; }
one $C12 12500 $H12 "" 0.03; W12a=$W; O12a=$O_; BARa=$(b $W12a); echo "  -> jitter-0.03 BAR $BARa%" >> $G
one $C15 15494 $H15 "--percept_filter" 0.03; W15a=$W; O15a=$O_
one $C12 12500 $H12 "" 0; W12b=$W; O12b=$O_; BARb=$(b $W12b); echo "  -> jitter-0 BAR $BARb%" >> $G
one $C15 15494 $H15 "--percept_filter" 0; W15b=$W; O15b=$O_
one $C149 14995 $H149 "--percept_filter" 0.03; W149a=$W; O149a=$O_
one $C149 14995 $H149 "--percept_filter" 0; W149b=$W; O149b=$O_
MSG="GAME SCENE (real height, 4 joint clips, P3c, 800 Hz, PH 2.6, saim off) | jitter 0.03: 12500 $W12a% / OF $O12a% -> bar $BARa%; 15494 $W15a% / $O15a% ($(d $W15a $BARa)); 14995 $W149a% / $O149a% ($(d $W149a $BARa)) | jitter 0 (game): 12500 $W12b% / OF $O12b% -> bar $BARb%; 15494 $W15b% / $O15b% ($(d $W15b $BARb)); 14995 $W149b% / $O149b% ($(d $W149b $BARb)) | $((($(date +%s) - S0) / 60)) min"
