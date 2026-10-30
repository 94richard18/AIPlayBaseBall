#!/bin/bash
# Human30 (planning 14, 10-31 00:2x; launch only after 14's confirmation: logs/h30/GO must exist): fine-tune 12500 @ 1.5x
# on AIB-PitchedBatting-Human30-v0 = SaimSwingEnv (row-9 scripted aim + full obs comp IN the loop, filtered percept,
# bound 0.4, window 0.60) with Human29's rewards + launch term (w_launch 20 toward 20 deg, x q) + hits-only exit hinge
# (0.5 x clamp(exit - 18, 0, 10)), w_hit_q 0 (w_miss 20 stays as the miss floor); aim sigmas 0.20 / 0.10 (Human29's).
# Slot: after ARMAIM2_300_END's verdict + R3 (RECOV3_END, both newer than this chain's start). ArmAim2c PASS at +300 ->
# wait for ARMAIM2_1000_END; FAIL -> next. Film slot 10-31 06:41 (20): the +300 launches only if it can end before
# 06:30 (launch <= 05:15) and the +700 only if launched <= 03:50; else wait for 20's film marker (logs/M3ROUTINE4_END.txt
# newer than this chain's start) or 08:00.
# Diag = the game config: diag1024 s0 with the scripted aim (--armaim + saim flags, as row 9), jitter 0, bias 0.055, aim 0.
# +300: ArmAim's +300 gate -> H30_300_END; FAIL stops. +1000 (700 more): ArmAim's +1000 gate AND 3 showcase tables x 5
# ONE OUTS (hit rate = its diag 'slow in zone') >= 19.1% with outfield >= 3% -> H30_1000_END "ADOPT" / "NO".
# Info: joint s6 with recover_rl (R3's ckpt if it passed, else model_1297) + saim. One Isaac at a time; markers by EXIT trap.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD
TASK=AIB-PitchedBatting-Human30-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
O=logs/h30; mkdir -p $O
M3=logs/H30_300_END.txt; M=logs/H30_1000_END.txt
rm -f $M3 $M; MSG="aborted"; M3DONE=0
touch $O/chain.stamp
trap '[ $M3DONE -eq 1 ] || echo "H30_300_END $(date +%H:%M) $MSG" > $M3; echo "H30_1000_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
note() { echo "$(date +%H:%M) $1" >> $O/progress.txt; }
filmwait() {  # $1 latest launch time on 10-31: later (before 08:00) -> wait for 20's film marker
  NOW=$(date +%s)
  if [ $NOW -gt $(date -d "2026-10-31 $1" +%s) ] && [ $NOW -lt $(date -d "2026-10-31 08:00" +%s) ]; then
    until [ logs/M3ROUTINE4_END.txt -nt $O/chain.stamp ] || [ $(date +%s) -ge $(date -d "2026-10-31 08:00" +%s) ]; do sleep 60; done
  fi
}
diag() {  # $1 ckpt $2 tag
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $SAIM --envs 1024 --seed 0 --tag $2 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$2.txt 2>&1
  grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt "diag $2"
  [ -f logs/diag_bat_contact_$2.json ] || { MSG="diag $2: no json"; exit 1; }
}
until [ -f $O/GO ]; do sleep 60; done
# 10-31 04:0x relaunch: ArmAim2c +300 PASS 01:44 and R3 PASS 02:18 are done; wait for the resumed +1000 (run_armaim2d.sh)
until [ logs/ARMAIM2_1000_END.txt -nt $O/chain.stamp ]; do sleep 60; done
grep -q "RAM watchdog|aborted|crash" logs/ARMAIM2_1000_END.txt && { MSG="held: ArmAim2c resume did not finish cleanly ($(cut -c1-120 logs/ARMAIM2_1000_END.txt))"; exit 1; }
# smoke
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 256 --max_iterations 2 --run_name human_pitched30_smoke --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_smoke.txt 2>&1
grep -q Traceback $O/train_smoke.txt && fail $O/train_smoke.txt smoke
grep -q "saim_swing" $O/train_smoke.txt || { MSG="smoke: SaimSwingEnv not built"; exit 1; }
# +300
# (filmwait 05:15: dropped 10-31 04:0x, 20's film ran 03:35)
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched30_1 --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_1.txt 2>&1
grep -q Traceback $O/train_1.txt && fail $O/train_1.txt train_1
D=$(ls -td $B/*human_pitched30_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
diag $D/model_$I.pt h30_$I
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h30_$I.json --stage 300 --label "model_$I ($D)" | tr '\n' ' ')
echo "H30_300_END $(date +%H:%M) $G | base row 9 (12500 + saim zero-shot): $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_r9wet.json --label r9 | tr '\n' ' ')" > $M3
M3DONE=1; cat $M3
case "$G" in PASS*) ;; *) MSG="stopped at +300: $G"; exit 0;; esac
# +700
# (filmwait 03:50: dropped 10-31 04:0x, 20's film ran 03:35)
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 700 --run_name human_pitched30_2 --resume True \
  --load_run $(basename $D) --checkpoint model_$I.pt > $O/train_2.txt 2>&1
grep -q Traceback $O/train_2.txt && fail $O/train_2.txt train_2
D2=$(ls -td $B/*human_pitched30_2 2>/dev/null | head -1); I2=$([ -n "$D2" ] && last $D2)
[ -n "$I2" ] || { MSG="train_2: no checkpoint"; exit 1; }
diag $D2/model_$I2.pt h30_$I2
G2=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h30_$I2.json --stage 1000 --label "model_$I2 ($D2)" | tr '\n' ' ')
note "+1000 gate: $G2"
# ONE OUTS adoption line: 3 showcase tables x 5 seeds
HR=$($P -c "import json; print(json.load(open('logs/diag_bat_contact_h30_$I2.json'))['summary']['by_speed']['slow in zone']['hit_rate'])")
oo() { for s in 0 1 2 3 4; do $P scripts/one_outs.py --matches 1000 --seed $s --batter_table $1 --hit_rate $2 --pitches $T --show 0 2>&1 | tail -2 | grep -o "batter [0-9.]*%\|outfield_hit [0-9.]*%" | grep -o "[0-9.]*" | tr '\n' ' '; done; echo; }
: > $O/t3.txt
for j in 1 2 3; do
  idle
  timeout 1200 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D2/model_$I2.pt $SAIM --eval_envs 1024 \
    --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_$j.txt 2>&1
  grep -q Traceback $O/table_$j.txt && fail $O/table_$j.txt table_$j
  $P scripts/make_batter_table.py $O/table_$j.txt $O/table_$j.json > /dev/null || { MSG="$G2 | table $j: make_batter_table failed"; exit 1; }
  echo "h30_$j $(oo $O/table_$j.json $HR)" >> $O/t3.txt
done
OO=$($P - <<PY
import statistics as st
r = [list(map(float, l.split()[1:])) for l in open("$O/t3.txt") if len(l.split()) == 11]
w = [st.mean(v[0::2]) for v in r]; o = [st.mean(v[1::2]) for v in r]
ok = len(w) == 3 and st.mean(w) >= 19.1 and st.mean(o) >= 3.0
print(("PASS" if ok else "FAIL") + f" ONE OUTS 3-table mean {st.mean(w):.1f}% (tables {', '.join(f'{x:.1f}' for x in w)}), outfield {st.mean(o):.1f}% (line 19.1% + outfield >= 3%)")
PY
)
note "$OO"
# info: joint s6 with recover_rl + saim
RCK=$(cat logs/recov3/recover_ckpt.txt 2>/dev/null); [ -n "$RCK" ] || RCK=logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2/model_1297.pt
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D2/model_$I2.pt --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RCK --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --saim $GY --saim_bound 0.4 --saim_window 0.60 \
  --saim_comp full --percept_filter --csv $O/joint_s6.csv > $O/joint_s6.txt 2>&1
JS=$($P -c "
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b = list(csv.DictReader(open('$O/joint_s6.csv')))
print(f'joint s6 (recover {\"$RCK\".split(\"/\")[-2]}): on the ground {sum(f(r[\"min_pelvis_z_2s\"]) < 0.35 for r in b)}/64, hits {sum(r[\"hit\"] == \"True\" for r in b)}/64')" 2>/dev/null)
case "$G2" in PASS*) case "$OO" in PASS*) V=ADOPT;; *) V=NO;; esac;; *) V=NO;; esac
MSG="$V | $G2 | $OO | $JS | +300: $G"
