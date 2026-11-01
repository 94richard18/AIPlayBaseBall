#!/bin/bash
# H32 game-scene tables for the segment-3 ckpt (planning 14, 11-02 06:1x, standing rule; 06:5x: 15993 as information only): if the seg-3 ckpt passes the H32_1500_END
# gates (the chain then goes on to segment 4 after the film7 HOLD), run its game-scene tables in BOTH columns (same protocol as
# run_h32_tablesg.sh: 3 tables @0.03 + 3 @0) beside segment 4 (starts once logs/h32/train_4.txt exists; never beside film7 -
# nofilm()). Bars from tables_game.txt: 16.93 (0.03) / 16.76 (0). The film / delivery pair changes from 15494 only if the
# candidate is >= 17.76 at jitter 0 AND >= 14.47 at jitter 0.03, with outfield >= 3 % in both. Hit rate = the chain's jitter-0
# game-scene diag (h32_<I>). Lines appended to logs/h32/tables_game.txt; marker logs/H32TG2_END.txt
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
M=logs/H32TG2_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H32TG2_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
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
# 14 (11-02 06:5x): 15993 failed 2 seg-3 gates -> its tables run as INFORMATION only (pair unchanged), beside segment 4
I=15993
MSG="waiting for logs/h32/train_4.txt (segment 4)"
until [ -f logs/h32/train_4.txt ]; do
  [ -f logs/H32S4_END.txt ] && { MSG="skipped: segment 4 ended before training ($(cut -c1-300 logs/H32S4_END.txt))"; exit 0; }
  sleep 60
done
D=$(ls -td $B/*human_pitched_h32_3 | head -1); C=$D/model_$I.pt
[ -f $C ] || { MSG="FAILED: $C missing"; exit 1; }
S0=$(date +%s); H=$(hr h32_$I)
echo "$(date +%H:%M) seg-3 ckpt model_$I (hit rate $H, jitter-0 diag), bars 16.93 (0.03) / 16.76 (0); INFORMATION only (it failed 2 seg-3 gates), pair unchanged" >> $G
one() {  # $1 jitter -> W O_ PT
  tables $C $O/t3_${I}_j$1 $H $O/t3_${I}_j$1.txt "--percept_filter" $1
  read W O_ PT <<< "$(t3 $O/t3_${I}_j$1.txt)"
  echo "$(date +%H:%M) $I @$1: $W% outfield $O_% (tables $PT)" >> $G
}
one 0.03; Wa=$W; Oa=$O_
one 0; Wb=$W; Ob=$O_
V="information only, pair unchanged"
MSG="GAME SCENE model_$I | jitter 0.03: $Wa% / OF $Oa% (bar 16.93, 15494 14.47) | jitter 0: $Wb% / OF $Ob% (bar 16.76, need 17.76) -> $V | $((($(date +%s) - S0) / 60)) min"
