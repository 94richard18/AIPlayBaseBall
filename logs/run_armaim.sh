#!/bin/bash
# Human28 ArmAim (planning 14, user 10-30 "動打者架構"; 14's go 10-30: launch without waiting if the smoke is clean).
# Waits for 20's routine film (logs/M3ROUTINE3_END.txt), then:
#  1. smoke: AIB-ArmAim-v0, 256 envs, 2 its (crash check)
#  2. zero check: diag1024 seed 0 with the smoke residual at aim_bound 0 must reproduce the 12500 baseline
#     (logs/diag_bat_contact_band12500.json: same hit vector, same closest-approach stations)
#  3. train 300 its (2048 envs, from scratch) -> diag -> scripts/armaim_gate.py --stage 300 -> logs/ARMAIM_300_END.txt
#  4. PASS only: resume +700 -> diag -> --stage 1000 -> logs/ARMAIM_1000_END.txt
# Diag: diag1024 seed 0, P3c t18 release csv, jitter 0, bias 0.055, aim 0, --contact_log (as the 12500 baseline).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-ArmAim-v0
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
B=logs/rsl_rl/aib1_armaim
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim; mkdir -p $O
M3=logs/ARMAIM_300_END.txt; M=logs/ARMAIM_1000_END.txt
rm -f $M3 $M; MSG="aborted"; M3DONE=0
trap '[ $M3DONE -eq 1 ] || echo "ARMAIM_300_END $(date +%H:%M) $MSG" > $M3; echo "ARMAIM_1000_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
diag() {  # $1 residual ckpt, $2 tag, [$3 extra args]
  idle
  timeout 1200 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $1 --envs 1024 --seed 0 --tag $2 \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log $3 > $O/diag_$2.txt 2>&1
  grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt "diag $2"
  [ -f logs/diag_bat_contact_$2.json ] || { MSG="diag $2: no json"; exit 1; }
}
until [ -f logs/M3ROUTINE3_END.txt ]; do sleep 60; done
idle
# 1. smoke
$P -u scripts/train.py --task $TASK --headless --num_envs 256 --max_iterations 2 --run_name armaim_smoke > $O/train_smoke.txt 2>&1
grep -q Traceback $O/train_smoke.txt && fail $O/train_smoke.txt smoke
DS=$(ls -td $B/*armaim_smoke 2>/dev/null | head -1); IS=$([ -n "$DS" ] && last $DS)
[ -n "$IS" ] || { MSG="smoke: no checkpoint"; exit 1; }
# 2. zero check
diag $DS/model_$IS.pt armaim_zero "--aim_bound 0"
ZC=$($P -c "
import json, numpy as np
a = {k: np.array(v, dtype=float) for k, v in json.load(open('logs/diag_bat_contact_band12500.json'))['per_pitch'].items()}
b = {k: np.array(v, dtype=float) for k, v in json.load(open('logs/diag_bat_contact_armaim_zero.json'))['per_pitch'].items()}
same = (a['hit'] == b['hit']).mean(); dx = np.nanmax(np.abs(a['x_on_bat'] - b['x_on_bat']))
print(('OK' if same >= 0.99 else 'BAD') + f' zero check: hits {int(b[\"hit\"].sum())} vs 12500 {int(a[\"hit\"].sum())}, same hit flag {same:.4f}, max |station diff| {dx:.4f} m')")
echo "$ZC" > $O/zero_check.txt
case "$ZC" in OK*) ;; *) MSG="$ZC"; exit 1;; esac
BASE=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_band12500.json --label 12500 | tr '\n' ' ')
# 3. train 300
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name armaim_1 > $O/train_1.txt 2>&1
grep -q Traceback $O/train_1.txt && fail $O/train_1.txt train_1
D=$(ls -td $B/*armaim_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
diag $D/model_$I.pt armaim_$I
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_armaim_$I.json --stage 300 --label "model_$I ($D)" | tr '\n' ' ')
echo "ARMAIM_300_END $(date +%H:%M) $G | $ZC | base: $BASE" > $M3; M3DONE=1; cat $M3
case "$G" in PASS*) ;; *) MSG="stopped at +300: $G"; exit 0;; esac
# 4. resume +700
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 700 --run_name armaim_2 --resume True \
  --load_run $(basename $D) --checkpoint model_$I.pt > $O/train_2.txt 2>&1
grep -q Traceback $O/train_2.txt && fail $O/train_2.txt train_2
D2=$(ls -td $B/*armaim_2 2>/dev/null | head -1); I2=$([ -n "$D2" ] && last $D2)
[ -n "$I2" ] || { MSG="train_2: no checkpoint"; exit 1; }
diag $D2/model_$I2.pt armaim_$I2
G2=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_armaim_$I2.json --stage 1000 --label "model_$I2 ($D2)" | tr '\n' ' ')
MSG="$G2 | +300: $G"
