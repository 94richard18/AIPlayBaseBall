#!/bin/bash
# H32 seg-4b ckpt adoption checks (planning 14, 11-02 10:4x): only if seg 4b passes its gates (logs/H32S4B_END.txt starts with
# PASS) -> its game-scene tables in both columns (same protocol as tables_game.txt) + joint s7 with R6b (s6b = the chain's row).
# Replacement of 15993 + R6b needs: tables >= 21.75 (0.03) and >= 26.19 (0) and two-seed after-done grounds <= 5/128 (tilt ok >= 52 each).
# Isaac <= 1, RAM >= 20 GB, nofilm per launch. Lines appended to logs/h32/tables_game.txt; marker logs/H32TG4B_END.txt
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
M=logs/H32TG4B_END.txt; rm -f $M; MSG="FAILED: aborted"
trap 'echo "H32TG4B_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
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
MSG="waiting for logs/H32S4B_END.txt"
until [ -f logs/H32S4B_END.txt ]; do sleep 60; done
grep -q '^H32S4B_END [0-9:]* PASS' logs/H32S4B_END.txt || { MSG="skipped: seg 4b did not pass ($(cut -c1-300 logs/H32S4B_END.txt))"; exit 0; }
I=$(grep -o '+2000 model_[0-9]*' logs/H32_2000B_END.txt | grep -o '[0-9]*$')
D=$(ls -td $B/*human_pitched_h32_4b | head -1); C=$D/model_$I.pt
[ -f $C ] || { MSG="FAILED: $C missing"; exit 1; }
S0=$(date +%s); H=$(hr h32b_$I)
echo "$(date +%H:%M) seg-4b ckpt model_$I (hit rate $H, jitter-0 diag); replaces 15993 only if >= 21.75 @0.03 and >= 26.19 @0 and recover <= 5/128" >> $G
one() {  # $1 jitter -> W O_ PT
  tables $C $O/t3_b${I}_j$1 $H $O/t3_b${I}_j$1.txt "--percept_filter" $1
  read W O_ PT <<< "$(t3 $O/t3_b${I}_j$1.txt)"
  echo "$(date +%H:%M) 4b $I @$1: $W% outfield $O_% (tables $PT)" >> $G
}
one 0.03; Wa=$W; Oa=$O_
one 0; Wb=$W; Ob=$O_
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt; RC=$(tr -d '\r\n' < logs/recov6/recover_ckpt.txt)
beside
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $C --envs 64 --seed 7 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --csv logs/h32/joint_s7b_$I.csv > logs/h32/joint_s7b_$I.txt 2>&1
[ -f logs/h32/joint_s7b_$I.csv ] && ! grep -q Traceback logs/h32/joint_s7b_$I.txt || { MSG="FAILED: joint s7 crash"; exit 1; }
RV=$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b = [list(csv.DictReader(open(p))) for p in ('logs/h32/joint_s6b_$I.csv', 'logs/h32/joint_s7b_$I.csv')]
ga = [sum(f(x['min_pelvis_z_2s']) < 0.35 and x['batter_fell_before_done'] != 'True' for x in r) for r in b]
t = [sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in r) for r in b]
bd = [sum(x['batter_fell_before_done'] == 'True' for x in r) for r in b]
ok = sum(ga) <= 5 and min(t) >= 52
print(f"{'ok' if ok else 'worse'} after-done ground {ga[0]} + {ga[1]} = {sum(ga)} (<= 5), tilt ok {t[0]} / {t[1]} (>= 52), before-done falls {bd[0]} / {bd[1]}")
PY
)
V=$($P -c "print('REPLACES 15993' if $Wa >= 21.75 and $Wb >= 26.19 and '$RV'.startswith('ok') else 'does not replace 15993')")
MSG="GAME SCENE 4b model_$I | jitter 0.03: $Wa% / OF $Oa% (need 21.75) | jitter 0: $Wb% / OF $Ob% (need 26.19) | recover R6b: $RV -> $V | $((($(date +%s) - S0) / 60)) min"
