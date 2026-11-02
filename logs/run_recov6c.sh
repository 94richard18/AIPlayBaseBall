#!/bin/bash
# R6c (planning 14, 11-02 08:2x): R6b (h25r6b model_3091) failed the two-seed recover rule on H32 15993 by one (after-done
# ground s6 2 + s7 3 = 5 > 4; tilt ok 56 / 58) -> R6b +300 on a 15993 bank (run_recov6b.sh style). Bank beside segment 4 (Isaac <= 1,
# RAM >= 20 GB, nofilm); training after segment 4 lands (logs/H32S4_END.txt) at idle; joint s6 + s7 of 15993 + filter, 6.5 s ->
# logs/R6C_END.txt. Same rule: after-done grounds s6 + s7 <= 4, tilt ok >= 52 each. Compared with R6b model_3091 on 15993 (s6).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE
PH="--phase_horizon 2.6"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
H=${BC:?BC = the batter ckpt}; HI=$(basename $H .pt); HI=${HI#model_}
R5=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b
Q=logs/recov6c; mkdir -p $Q
M=logs/R6C_END.txt; rm -f $M; MSG="aborted"
trap 'echo "R6C_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
[ -f $H ] || { MSG="FAILED: no ckpt $H"; exit 1; }
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt logs/CUTTEST_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
beside() { until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
joint() {  # $1 seed, $2 recover ckpt, $3 out prefix
  beside
  timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed $1 --after_done recover_rl \
    --recover_ckpt $2 --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter $PH --seconds 6.5 --csv ${3}.csv > ${3}.txt 2>&1
  grep -q Traceback ${3}.txt && fail ${3}.txt $(basename $3)
}
R6REF=${R6REF:-logs/h32/joint_s6_$HI.csv}  # the H32 seg-3 row = R6b model_3091 on 15993
BK=logs/recover_bank_r6c_${HI}_s0.npz
if [ -f $BK ] && grep -q "physics dt 0.001250" $Q/bank_$HI.txt 2>/dev/null && grep -q '^\[bank\] seed' $Q/bank_$HI.txt; then :; else
  MSG="bank ($HI)"; beside
  timeout 1800 $P -u scripts/recover_bank.py --headless --checkpoint $H --percept_filter $PH --physics_dt 0.00125 --envs 2048 --seed 0 \
    --out $BK > $Q/bank_$HI.txt 2>&1
  grep -q Traceback $Q/bank_$HI.txt && fail $Q/bank_$HI.txt bank
  [ -f $BK ] || { MSG="bank: no npz"; exit 1; }
  grep -q "physics dt 0.001250" $Q/bank_$HI.txt || { MSG="bank: no 800 Hz line"; exit 1; }
fi
BL=$(grep '^\[bank\] seed' $Q/bank_$HI.txt)
MSG="bank done ($BL), waiting for logs/H32S4_END.txt (segment 4 lands)"
until [ -f logs/H32S4_END.txt ]; do sleep 60; done
MSG="train ($BL)"; idle
AIB_PHYSICS_DT=0.00125 AIB_RECOVER_BANK=$BK timeout 7200 $P -u scripts/train.py --task AIB-Recover-R2-v0 --headless --num_envs ${NENVS:-4096} \
  --max_iterations ${R6C_ITS:-300} --run_name h25r6c --resume True --load_run ${R6C_LOAD_RUN:-$(basename $R5)} --checkpoint ${R6C_CKPT:-model_3091.pt} > $Q/train.txt 2>&1 &
TP=$!; while kill -0 $TP 2>/dev/null; do sleep 60
  tail -c 20000 $Q/train.txt | grep -o 'Iteration time: [0-9.]*' | tail -1 | awk '{exit !($3 > 120)}' && {
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'h25r6c' -and \$_.CommandLine -match 'train.py' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    MSG="train: killed by the watchdog ($(tail -c 20000 $Q/train.txt | grep -o 'Iteration time: [0-9.]*s' | tail -1) beside $(isaac) Isaac, VRAM thrash); not relaunched, waits for 14"; exit 1; }
done
grep -q Traceback $Q/train.txt && fail $Q/train.txt train
grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $Q/train.txt || { MSG="train: no AIB_PHYSICS_DT line"; exit 1; }
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r6c 2>/dev/null | head -1)
CL=$([ -n "$RUN" ] && ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
[ -n "$CL" ] || { MSG="train: no checkpoint"; exit 1; }
MSG="joint ($BL; ckpt $RUN/model_$CL.pt)"
joint 6 $RUN/model_$CL.pt $Q/joint_s6_$HI
joint 7 $RUN/model_$CL.pt $Q/joint_s7_$HI
MSG="$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b6 = list(csv.DictReader(open('$Q/joint_s6_$HI.csv'))); b7 = list(csv.DictReader(open('$Q/joint_s7_$HI.csv')))
r5 = list(csv.DictReader(open('$R6REF')))
g = lambda b: sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b); ga = lambda b: sum(f(x['min_pelvis_z_2s']) < 0.35 and x['batter_fell_before_done'] != 'True' for x in b); h = lambda b: sum(x['hit'] == 'True' for x in b)
t30 = lambda b: sum(f(x['torso_tilt_1p5_deg']) <= 30 for x in b); fl = lambda b: sum(x['batter_fell'] == 'True' for x in b)
bd = lambda b: sum(x['batter_fell_before_done'] == 'True' for x in b)
ok = ga(b6) + ga(b7) <= 4 and t30(b6) >= 52 and t30(b7) >= 52
print(("PASS" if ok else "FAIL") + f" R6c ($HI + filter, no saim, 1.0x, joints 6.5 s + recover h25r6c model_$CL, 800 Hz): after-done ground s6 {ga(b6)} s7 {ga(b7)} (bar <= 4 total), all ground s6 {g(b6)}/64 s7 {g(b7)}/64, "
      f"falls before done {bd(b6)} / {bd(b7)}, tilt +1.5 s <= 30 deg s6 {t30(b6)}/64 s7 {t30(b7)}/64 (bar 52 each), batter_fell s6 {fl(b6)} s7 {fl(b7)}, "
      f"hits s6 {h(b6)} s7 {h(b7)} | R6b model_3091 on $HI s6 ($R6REF): ground {g(r5)} (after done {ga(r5)}), tilt ok {t30(r5)}, fell {fl(r5)} (before done {bd(r5)}), hits {h(r5)}")
PY
) | $BL | $(grep -o 'Iteration time: [0-9.]*s' $Q/train.txt | tail -1) | ckpt $RUN/model_$CL.pt | rows $Q/joint_s6_$HI.csv $Q/joint_s7_$HI.csv"
case "$MSG" in PASS*) echo "$RUN/model_$CL.pt" > $Q/recover_ckpt.txt;; esac
