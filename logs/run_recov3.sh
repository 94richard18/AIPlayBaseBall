#!/bin/bash
# Human25-R3 (planning 14, 10-30 23:xx): the miss-side falls of the game batter with the scripted aim (12500 + saim joint s6
# 12/64 on the ground, h29s variant B 15/64: the recover_rl hand-off fires at hit | done with the aim offset still ~0.3 rad
# in the arm pose and the anchor, states the R2 bank never had). Called by run_armaim2c.sh after ARMAIM2_300_END.
#   0. needed? the winner's joint s6 line (H29_ADOPT / SAIM_ADOPT) JOINT OK -> skip
#   1. bank: the winner swing inside AIB-ArmAim-Play-v0 + saim (filter, full comp, window 0.60, the SAIM_ADOPT bound),
#      2048 envs seed 0, misses included -> logs/recover_bank_r3_s0.npz
#   2. resume recover_rl model_1297 +300 on AIB-Recover-R2-v0 (same R2 rewards) with that bank (AIB_RECOVER_BANK)
#   3. held-out: joint s6 with the winner + saim + the new recover ckpt: pitch columns vs fd_s6 64/64, on the ground <= 2/64
# Marker logs/RECOV3_END.txt (EXIT trap); on PASS logs/recov3/recover_ckpt.txt. One Isaac at a time (Isaac 0 twice 60 s apart).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
unset AIB_FIX_STD  # (run_armaim2c.sh exports it for the residual; the recovery resume trains its std as R2 did)
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-10-29_08-51-50_h25r2
GY=logs/armaim_auth/scripted_gain.json
O=logs/recov3; M=logs/RECOV3_END.txt; BK=logs/recover_bank_r3_s0.npz; mkdir -p $O
rm -f $M $O/recover_ckpt.txt; MSG="aborted"
trap 'echo "RECOV3_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
read SW WTAG < logs/h29adopt/winner.txt
BND=$(grep -o "aim_bound [0-9.]*" logs/saim/progress.txt 2>/dev/null | tail -1 | awk '{print $2}'); [ -n "$BND" ] || BND=0.4
# 0. needed?
case $WTAG in
  d12500) JL=$(grep "joint s6 recover_rl + saim" logs/saim/progress.txt | tail -1);;
  *) JL=$(grep " $WTAG (hit rate" logs/h29adopt/progress.txt | tail -1);;
esac
case "$JL" in *"JOINT OK"*|*"PASS joint s6"*) MSG="skipped: winner $WTAG joint s6 already passes ($JL)"; exit 0;; esac
echo "$(date +%H:%M) R3 for winner $WTAG ($SW), bound $BND: $JL" >> $O/progress.txt
# 1. bank
idle
timeout 1800 $P -u scripts/recover_bank.py --headless --checkpoint $SW --armaim --aim_scripted $GY --aim_bound $BND --aim_window 0.60 \
  --aim_obs_comp --aim_obs_comp_full --percept_filter --envs 2048 --seed 0 --out $BK > $O/bank.txt 2>&1
grep -q Traceback $O/bank.txt && fail $O/bank.txt bank
[ -f $BK ] || { MSG="bank: no npz"; exit 1; }
BL=$(grep '^\[bank\] seed' $O/bank.txt)
echo "$(date +%H:%M) $BL" >> $O/progress.txt
# 2. train
idle
AIB_RECOVER_BANK=$BK timeout 5400 $P -u scripts/train.py --task AIB-Recover-R2-v0 --headless --num_envs 4096 --max_iterations 300 \
  --run_name h25r3 --resume True --load_run $(basename $RC) --checkpoint model_1297.pt > $O/train.txt 2>&1
grep -q Traceback $O/train.txt && fail $O/train.txt train
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r3 2>/dev/null | head -1)
CL=$([ -n "$RUN" ] && ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
[ -n "$CL" ] || { MSG="train: no checkpoint"; exit 1; }
# 3. held-out joint s6
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $SW --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RUN/model_$CL.pt --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --saim $GY --saim_bound $BND \
  --saim_window 0.60 --saim_comp full --percept_filter --trig_dump $O/trig_s6.npz --csv $O/joint_s6.csv > $O/joint_s6.txt 2>&1
grep -q Traceback $O/joint_s6.txt && fail $O/joint_s6.txt joint_s6
J=$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
a = list(csv.DictReader(open('logs/falldiag2/fd_s6.csv'))); b = list(csv.DictReader(open('$O/joint_s6.csv')))
cols = ['clip', 'kmh', 'handoff_t', 'handed', 'pred_y', 'pred_z', 'flight_pred', 'handoff_jump_m', 'pitcher_fell']
same = sum(all(x[c] == y[c] for c in cols) for x, y in zip(a, b)); g = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in b)
gm = sum(f(x['min_pelvis_z_2s']) < 0.35 and x['hit'] != 'True' for x in b)
tt = sorted(f(x['torso_tilt_1p5_deg']) for x in b if f(x['torso_tilt_1p5_deg']) == f(x['torso_tilt_1p5_deg']))
print(('PASS' if same == 64 and g <= 2 else 'FAIL') + f" R3 joint s6 (winner + saim + model_$CL): pitch columns {same}/64, on the ground {g}/64 "
      f"(misses {gm}), hits {sum(x['hit'] == 'True' for x in b)}/64, torso tilt +1.5 s med {tt[len(tt) // 2] if tt else float('nan'):.1f} deg")
PY
)
NN=$($P scripts/trig_nn.py --trig $O/trig_s6.npz --csv $O/joint_s6.csv --bank $BK --label r3)
case "$J" in PASS*) echo "$RUN/model_$CL.pt" > $O/recover_ckpt.txt;; esac
MSG="$J | $NN | $BL | ckpt $RUN/model_$CL.pt"
