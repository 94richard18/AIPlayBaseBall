#!/bin/bash
# ONE OUTS at 800 Hz (planning 14, 11-01 12:0x). Usage: run_h31_oo800.sh base | adopt | basesaim
#  base:  delivery baseline (14, 12:1x) = plain h9b model_12500 @ 1.5x, NO saim and no percept_filter (the delivery batter: M2
#         percept, filter flag off), beside the H31 trainer
#         once logs/H31_JOINT_13997_END.txt exists; each Isaac launch waits for Isaac <= 1, RAM >= 20 GB and no FILM* / PDT GO
#         without a newer END -> logs/H31_BASE800_END.txt
#  adopt: candidate = BC from logs/h31/ADOPT_GO (written by 14 after FILM4_END, HOLD on), Isaac 0 -> logs/H31_ADOPT_END.txt
#         (always written, FAILED / SKIPPED included: 14's job removes HOLD on it); + the jitter-0 set if the 0.03 set took
#         < 45 min (total < 90 min)
#  basesaim: analysis row only (not the bar) = 12500 + saim + percept_filter, after H31_ADOPT_END, only if the GPU is idle within
#         30 min of it (else SKIPPED) -> logs/H31_BASESAIM800_END.txt
# Protocol (run_h30c_adopt2.sh) at 800 Hz: one jitter-0 diag for the hit rate (all clips, phase 2.6, --physics_dt 0.00125),
# then showcase --eval_envs 1024 x 3 tables at the 0.03 jitter (--physics_dt 0.00125 --phase_horizon 2.6), make_batter_table,
# one_outs.py 1000 matches x 5 seeds per table -> 5x3 mean batter win %, outfield-hit %, per-table means.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
MODE=$1
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
DT="--physics_dt 0.00125 --phase_horizon 2.6"
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
beside() { until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
t3() { $P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$1") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
print(f"{st.mean(w):.2f} {st.mean(o):.2f} {','.join(f'{x:.1f}' for x in w)}" if len(r) == 3 else "nan nan incomplete")
PY
}
hitrate() {  # $1 ckpt, $2 tag -> HR (jitter-0 in-zone slow hit rate at 800 Hz)
  $WAIT
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $AIM --envs 1024 --seed 0 --tag $2 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT --contact_log > $O/diag_$2.txt 2>&1
  [ -f logs/diag_bat_contact_$2.json ] && ! grep -q Traceback $O/diag_$2.txt || { MSG="FAILED: hit-rate diag $2 crashed ($(grep -A14 Traceback $O/diag_$2.txt | tail -1))"; exit 1; }
  HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_$2.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
}
tables() {  # $1 ckpt, $2 timing noise, $3 prefix, $4 hit rate, $5 out list file
  : > $5
  for j in 1 2 3; do
    $WAIT
    timeout 1800 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $1 $AIM --eval_envs 1024 --timing_noise $2 \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R $DT > ${3}_$j.txt 2>&1
    grep -q Traceback ${3}_$j.txt && { MSG="FAILED: showcase $(basename $3)_$j crashed ($(grep -A14 Traceback ${3}_$j.txt | tail -1))"; exit 1; }
    grep -q "physics dt 0.001250" ${3}_$j.txt || { MSG="FAILED: showcase $(basename $3)_$j has no 800 Hz line"; exit 1; }
    $P scripts/make_batter_table.py ${3}_$j.txt ${3}_$j.json > /dev/null || { MSG="FAILED: make_batter_table $(basename $3)_$j"; exit 1; }
    echo "t_$j $(oo ${3}_$j.json $4)" >> $5
  done
}
fmt() { read W OF PT <<< "$(t3 $1)"; echo "5x3 mean $W% outfield $OF% (tables $PT)"; }
if [ "$MODE" = base ]; then
  O=logs/h31/base800; mkdir -p $O; M=logs/H31_BASE800_END.txt; WAIT=beside; AIM=""
  rm -f $M; MSG="aborted"; trap 'echo "H31_BASE800_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
  until [ -f logs/H31_JOINT_13997_END.txt ]; do sleep 60; done
  CK=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
  S0=$(date +%s)
  hitrate $CK base800_j0
  tables $CK 0.03 $O/t3 $HR $O/t3.txt
  MSG="delivery 12500 @ 1.5x (no saim, no percept_filter), 800 Hz, phase 2.6, jitter 0.03: $(fmt $O/t3.txt), hit rate $HR, $((($(date +%s) - S0) / 60)) min (beside the trainer)"
elif [ "$MODE" = adopt ]; then
  O=logs/h31/adopt800; mkdir -p $O; M=logs/H31_ADOPT_END.txt; WAIT=idle; AIM=$SAIM
  rm -f $M; MSG="FAILED: aborted"; trap 'echo "H31_ADOPT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
  until [ -f logs/h31/ADOPT_GO ]; do sleep 60; done
  CK=$(sed -n 's/.*BC=\([^ ]*\).*/\1/p' logs/h31/ADOPT_GO | head -1 | tr -d '\r')
  [ -n "$CK" ] && [ -f "$CK" ] || { MSG="SKIPPED: no BC checkpoint in ADOPT_GO ('$CK')"; exit 0; }
  I=$(basename $CK .pt); I=${I#model_}
  S0=$(date +%s)
  hitrate $CK adopt800_${I}_j0
  tables $CK 0.03 $O/t3_$I $HR $O/t3_$I.txt
  read W OF PT <<< "$(t3 $O/t3_$I.txt)"
  BL=""; [ -f logs/H31_BASE800_END.txt ] && BW=$(sed -n 's/.*5x3 mean \([0-9.]*\)%.*/\1/p' logs/H31_BASE800_END.txt) && [ -n "$BW" ] \
    && BL=", delta vs base800 $($P -c "print(f'{$W - $BW:+.2f}')") (base $BW%)"
  MSG="model_$I 800 Hz jitter 0.03: $(fmt $O/t3_$I.txt)$BL, hit rate $HR"
  T1=$((($(date +%s) - S0) / 60))
  if [ $T1 -lt 45 ]; then
    tables $CK 0 $O/t0_$I $HR $O/t0_$I.txt
    MSG="$MSG | jitter 0: $(fmt $O/t0_$I.txt)"
  else MSG="$MSG | jitter 0 SKIPPED (0.03 set took $T1 min)"; fi
  MSG="$MSG | $((($(date +%s) - S0) / 60)) min"
elif [ "$MODE" = basesaim ]; then
  O=logs/h31/basesaim800; mkdir -p $O; M=logs/H31_BASESAIM800_END.txt; WAIT=idle; AIM=$SAIM
  rm -f $M; MSG="aborted"; trap 'echo "H31_BASESAIM800_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
  until [ -f logs/H31_ADOPT_END.txt ]; do sleep 60; done
  A0=$(date +%s)
  until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do
    [ $(($(date +%s) - A0)) -gt 1800 ] && { MSG="SKIPPED: GPU not idle within 30 min of H31_ADOPT_END"; exit 0; }; sleep 30
  done
  CK=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
  S0=$(date +%s)
  hitrate $CK basesaim800_j0
  tables $CK 0.03 $O/t3 $HR $O/t3.txt
  MSG="ANALYSIS ROW (not the bar): 12500 + saim + percept_filter, 800 Hz, phase 2.6, jitter 0.03: $(fmt $O/t3.txt), hit rate $HR, $((($(date +%s) - S0) / 60)) min"
else echo "usage: $0 base | adopt | basesaim"; exit 2; fi
