#!/bin/bash
# run_h33_tables_j03.sh (14, 11-03 14:3x): the jitter-0.03 column only (H33J +500 table check); marker H33_TABLES_J03_<I>_END.
# run_h33_tables.sh <ckpt path> <iter> (14, 11-03 06:0x): game-scene adoption tables for a form-first H33 ckpt, protocol of
# 11-02 (run_h32_tablesg2.sh): 3 tables @ jitter 0.03 + 3 @ 0, hit rate = the jitter-0 game-scene diag of the H33 chain
# (diag_bat_contact_h33f_<I>.json). Bars 16.93 (0.03) / 16.76 (0). Information for 14 (how much distance the line still needs).
# Lines -> logs/h33/tables_game_<I>.txt; marker logs/H33_TABLES_<I>_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
DT="--physics_dt 0.00125 --phase_horizon 2.6"
C=$1; I=$2; DG=${3:-h33f_$2}  # $3: the jitter-0 diag tag for the hit rate (default h33f_<I>)
O=logs/h33/tables_game; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"; mkdir -p $O
M=logs/H33_TABLES_J03_${I}_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H33_TABLES_J03_${I}_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
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
G=logs/h33/tables_game_$I.txt
[ -f $C ] || { MSG="FAILED: $C missing"; exit 1; }
S0=$(date +%s); H=$(hr $DG)
echo "$(date +%H:%M) H33 form-first ckpt model_$I (swing policy, 11-02 protocol; hit rate $H, jitter-0 diag $DG), bars 16.93 (0.03) / 16.76 (0)" >> $G
one() {  # $1 jitter -> W O_ PT
  tables $C $O/t3_${I}_j$1 $H $O/t3_${I}_j$1.txt "--percept_filter" $1
  read W O_ PT <<< "$(t3 $O/t3_${I}_j$1.txt)"
  echo "$(date +%H:%M) $I @$1: $W% outfield $O_% (tables $PT)" >> $G
}
one 0.03; Wa=$W; Oa=$O_
MSG="GAME SCENE j0.03 only model_$I | jitter 0.03: $Wa% / OF $Oa% (bar 16.93) | $((($(date +%s) - S0) / 60)) min"
