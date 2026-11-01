#!/bin/bash
# R5b = R5 +300 from h25r5 model_2194 on the same r5 bank (R5 FAIL 14:26: ground 5+7, tilt 57/55, reward still rising), gated on logs/recov5b/GO.
# R5 (planning 14, 11-01 12:2x): the H31 13997 swing falls 34/64 in joint s6 with R4 (27 after the hand-off) - R4 was built on
# ~16 m/s hand-offs at 400 Hz. R5 = the R4 recipe on 13997-class hand-off states at the joint scene's 800 Hz: recover bank from
# model_13997 + saim + percept_filter at phase 2.6 (--physics_dt 0.00125), resume R4 h25r4 model_1895 +300 on AIB-Recover-R2-v0
# with AIB_PHYSICS_DT=0.00125, then joint s6 + s7 of 13997 + saim + R5 -> logs/R5_END.txt (R4 rule: PASS = s6 + s7 ground <= 4
# and torso tilt +1.5 s <= 30 deg on >= 56/64 in each seed; also batter_fell vs the R4 s6 row 34/64). Runs beside the H31 trainer
# after H31_BASE800_END and outside the film4 / adoption window: each Isaac step waits for Isaac <= 1, RAM >= 20 GB, no FILM* /
# PDT GO without a newer END, and (ADOPT_GO present) H31_ADOPT_END; training does not start after 15:00 before H31_ADOPT_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
GY=logs/armaim_auth/scripted_gain.json
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
PH="--phase_horizon 2.6"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
H=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-01_09-54-34_human_pitched_h31_3/model_13997.pt
R4=logs/rsl_rl/aib1_recover/2026-11-01_12-34-46_h25r5
Q=logs/recov5b; mkdir -p $Q
M=logs/R5B_END.txt; rm -f $M; MSG="aborted"
trap 'echo "R5B_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
noadopt() { [ ! -f logs/h31/ADOPT_GO ] || [ -f logs/H31_ADOPT_END.txt ]; }
beside() { until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm && noadopt; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
joint() {  # $1 seed, $2 recover ckpt, $3 out prefix
  beside
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $2 --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX $PH --csv ${3}.csv > ${3}.txt 2>&1
  grep -q Traceback ${3}.txt && fail ${3}.txt $(basename $3)
}
MSG="waiting for logs/recov5b/GO"
BK=logs/recover_bank_r5_s0.npz; until [ -f $Q/GO ]; do sleep 60; done
BL=$(grep '^\[bank\] seed' logs/recov5/bank.txt)
MSG="train ($BL)"
until { [ "$(date +%H%M)" -lt 1500 ] || [ -f logs/H31_ADOPT_END.txt ]; } && [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm && noadopt; do sleep 60; done
AIB_PHYSICS_DT=0.00125 AIB_RECOVER_BANK=$BK timeout 7200 $P -u scripts/train.py --task AIB-Recover-R2-v0 --headless --num_envs 4096 \
  --max_iterations 300 --run_name h25r5b --resume True --load_run $(basename $R4) --checkpoint model_2194.pt > $Q/train.txt 2>&1
grep -q Traceback $Q/train.txt && fail $Q/train.txt train
grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $Q/train.txt || { MSG="train: no AIB_PHYSICS_DT line"; exit 1; }
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r5b 2>/dev/null | head -1)
CL=$([ -n "$RUN" ] && ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
[ -n "$CL" ] || { MSG="train: no checkpoint"; exit 1; }
MSG="joint ($BL; ckpt $RUN/model_$CL.pt)"
joint 6 $RUN/model_$CL.pt $Q/joint_s6
joint 7 $RUN/model_$CL.pt $Q/joint_s7
MSG="$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b6 = list(csv.DictReader(open('$Q/joint_s6.csv'))); b7 = list(csv.DictReader(open('$Q/joint_s7.csv')))
r4 = list(csv.DictReader(open('logs/h31/joint_s6_13997.csv')))
g = lambda b: sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b); h = lambda b: sum(x['hit'] == 'True' for x in b)
t30 = lambda b: sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b); fl = lambda b: sum(x['batter_fell'] == 'True' for x in b)
bd = lambda b: sum(x['batter_fell_before_done'] == 'True' for x in b)
ok = g(b6) + g(b7) <= 4 and t30(b6) >= 56 and t30(b7) >= 56
print(("PASS" if ok else "FAIL") + f" R5 (13997 + saim + recover h25r5b model_$CL, 800 Hz): ground s6 {g(b6)}/64 s7 {g(b7)}/64, "
      f"tilt +1.5 s <= 30 deg s6 {t30(b6)}/64 s7 {t30(b7)}/64 (bar 56), batter_fell s6 {fl(b6)} s7 {fl(b7)} (before done {bd(b6)} / {bd(b7)}), "
      f"hits s6 {h(b6)} s7 {h(b7)} | R4 on 13997 s6: ground {g(r4)}, tilt ok {t30(r4)}, fell {fl(r4)} (before done {bd(r4)}), hits {h(r4)}")
PY
) | $BL | $(grep -o 'Iteration time: [0-9.]*s' $Q/train.txt | tail -1) | ckpt $RUN/model_$CL.pt"
case "$MSG" in PASS*) echo "$RUN/model_$CL.pt" > $Q/recover_ckpt.txt;; esac
