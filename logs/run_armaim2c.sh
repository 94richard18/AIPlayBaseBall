#!/bin/bash
# ArmAim2 + comp (planning 14, 10-30 21:3x): ArmAim2 (one held residual per swing, +-0.2 rad = aim_delta 0.2 / bound, actor head
# zeroed = the row-9 scripted aim, std fixed 0.1, entropy 0) on the row-9 base (quadratic-filtered percept, window 0.60, full
# obs comp), aim_bound = the SAIM_ADOPT pick (row 9 0.4 / row 10 0.6), reward = aim term + w_hit_q x q + 0.5 x clamp(exit - 18,
# 0, 10) on hits. Waits for SAIM_ADOPT_END + H29_ADOPT_END; the frozen swing = logs/h29adopt/winner.txt (AIB_SAIM_SWING). Gates = ArmAim's +300 / +1000 -> ARMAIM2_300_END / ARMAIM2_1000_END.
# Film slot 10-31 06:41 (20): the +700 resume (~2.4 h) launches only if it can end before 06:30 (launch <= 03:50), else it
# waits for 20's film marker (logs/M3ROUTINE4_END.txt newer than this chain's start, written by logs/routine1031.sh; or 08:00).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-ArmAim2Comp-v0
export AIB_FIX_STD=1
GJ=logs/armaim_auth/scripted_gain.json
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
B=logs/rsl_rl/aib1_armaim2
PR=logs/saim/progress.txt
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim2c; mkdir -p $O
M3=logs/ARMAIM2_300_END.txt; M=logs/ARMAIM2_1000_END.txt
rm -f $M3 $M; MSG="aborted"; M3DONE=0
touch $O/chain.stamp
trap '[ $M3DONE -eq 1 ] || echo "ARMAIM2_300_END $(date +%H:%M) $MSG" > $M3; echo "ARMAIM2_1000_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
filmwait() {  # $1 latest launch time on 10-31: later (before 08:00) -> wait for 20's film marker
  NOW=$(date +%s)
  if [ $NOW -gt $(date -d "2026-10-31 $1" +%s) ] && [ $NOW -lt $(date -d "2026-10-31 08:00" +%s) ]; then
    until [ logs/M3ROUTINE4_END.txt -nt $O/chain.stamp ] || [ $(date +%s) -ge $(date -d "2026-10-31 08:00" +%s) ]; do sleep 60; done
  fi
}
diag() {  # $1 residual ckpt, $2 tag, [$3 extra args]
  idle
  timeout 1200 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $1 --aim_scripted $GJ --aim_hold --aim_bound $AIB_SAIM_BOUND --aim_delta $DELTA --aim_window 0.60 --percept_filter --aim_obs_comp --aim_obs_comp_full --envs 1024 --seed 0 --tag $2 \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log $3 > $O/diag_$2.txt 2>&1
  grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt "diag $2"
  [ -f logs/diag_bat_contact_$2.json ] || { MSG="diag $2: no json"; exit 1; }
}
until [ -f logs/SAIM_ADOPT_END.txt ] && [ -f logs/H29_ADOPT_END.txt ] && [ logs/H29_ADOPT_END.txt -nt $O/chain.stamp ]; do sleep 60; done
read SW WTAG < logs/h29adopt/winner.txt
[ -f "$SW" ] || { MSG="no winner swing in logs/h29adopt/winner.txt"; exit 1; }
export AIB_SAIM_SWING=$(cd $(dirname $SW) && pwd -W)/$(basename $SW)  # the frozen swing (ArmAim2CompEnvCfg.swing_checkpoint)
export AIB_SAIM_BOUND=$(grep -o "aim_bound [0-9.]*" $PR 2>/dev/null | tail -1 | awk '{print $2}')
[ -n "$AIB_SAIM_BOUND" ] || { MSG="no aim_bound pick in $PR"; exit 1; }
DELTA=$($P -c "print(0.2 / $AIB_SAIM_BOUND)")
idle
# 1. smoke
$P -u scripts/train.py --task $TASK --headless --num_envs 256 --max_iterations 2 --run_name armaim2c_smoke > $O/train_smoke.txt 2>&1
grep -q Traceback $O/train_smoke.txt && fail $O/train_smoke.txt smoke
DS=$(ls -td $B/*armaim2c_smoke 2>/dev/null | head -1); IS=$([ -n "$DS" ] && last $DS)
[ -n "$IS" ] || { MSG="smoke: no checkpoint"; exit 1; }
ZC="swing $WTAG ($SW) aim_bound $AIB_SAIM_BOUND delta $DELTA | adopt: $(head -c 300 logs/SAIM_ADOPT_END.txt) | $(head -c 300 logs/H29_ADOPT_END.txt)"
BASE=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_band12500.json --label 12500 | tr '\n' ' ')
# 3. train 300 (~1 h + diag): only if its diag lands before 06:30
filmwait 05:15
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name armaim2c_1 > $O/train_1.txt 2>&1
grep -q Traceback $O/train_1.txt && fail $O/train_1.txt train_1
D=$(ls -td $B/*armaim2c_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
diag $D/model_$I.pt armaim2c_$I
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_armaim2c_$I.json --stage 300 --label "model_$I ($D)" | tr '\n' ' ')
echo "ARMAIM2_300_END $(date +%H:%M) $G | $ZC | base: $BASE" > $M3; M3DONE=1; cat $M3
case "$G" in PASS*) ;; *) MSG="stopped at +300: $G"; exit 0;; esac
# 4. resume +700 (~2.4 h): only if it ends before 06:30
filmwait 03:50
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 700 --run_name armaim2c_2 --resume True \
  --load_run $(basename $D) --checkpoint model_$I.pt > $O/train_2.txt 2>&1
grep -q Traceback $O/train_2.txt && fail $O/train_2.txt train_2
D2=$(ls -td $B/*armaim2c_2 2>/dev/null | head -1); I2=$([ -n "$D2" ] && last $D2)
[ -n "$I2" ] || { MSG="train_2: no checkpoint"; exit 1; }
diag $D2/model_$I2.pt armaim2c_$I2
G2=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_armaim2c_$I2.json --stage 1000 --label "model_$I2 ($D2)" | tr '\n' ' ')
MSG="$G2 | +300: $G"
