#!/bin/bash
# Human25-R (planning 14, 10-29 05:3x; user "直接做新機制"): separate post-contact recovery policy.
# Stage 0: state bank (frozen 12500 @ 1.5x, standalone PitchedBattingEnv, P3c, joint clips) seed 0 (train, 2048 envs)
#          and seed 6 (held out, 1024 envs) + the a = 0 baseline on seed 6 -> marker logs/RECOV_BANK_END.txt
# Stage 1: PPO on AIB-Recover-v0 to +300 -> falsifier on the seed-6 bank (on the ground within 2 s < 60%, else stop)
#          -> resume to +1000 -> falsifier (< 30%) -> marker logs/RECOV_TRAIN_END.txt
# One Isaac at a time: Isaac 0 twice 60 s apart before every launch; markers written by EXIT traps.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
MB=logs/RECOV_BANK_END.txt; MT=logs/RECOV_TRAIN_END.txt; O=logs/recov
rm -f $MB $MT; STAGE=bank; MSG="aborted"
trap 'if [ "$STAGE" = bank ]; then echo "RECOV_BANK_END $(date +%H:%M) $MSG" > $MB; cat $MB; echo "RECOV_TRAIN_END $(date +%H:%M) not started ($MSG)" > $MT; else echo "RECOV_TRAIN_END $(date +%H:%M) $MSG" > $MT; fi; cat $MT' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
mkdir -p $O
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
for sp in "0 2048" "6 1024"; do
  set -- $sp
  idle
  timeout 1200 $P -u scripts/recover_bank.py --headless --checkpoint $BC --envs $2 --seed $1 --out logs/recover_bank_s$1.npz > $O/bank_s$1.txt 2>&1
  grep '^\[bank\]' $O/bank_s$1.txt
  if grep -q Traceback $O/bank_s$1.txt || [ ! -s logs/recover_bank_s$1.npz ]; then fail $O/bank_s$1.txt bank_s$1; fi
done
idle
timeout 900 $P -u scripts/recover_eval.py --headless --zero > $O/eval_zero.txt 2>&1
grep -E '^\[reval\]|^REVAL' $O/eval_zero.txt
grep -q '^REVAL' $O/eval_zero.txt || fail $O/eval_zero.txt eval_zero
MSG="banks s0 / s6 + a=0 baseline done"
echo "RECOV_BANK_END $(date +%H:%M) $MSG" > $MB; cat $MB
STAGE=train; MSG="aborted in training"
idle
timeout 9000 $P -u scripts/train.py --task AIB-Recover-v0 --headless --num_envs 4096 --max_iterations 300 --run_name h25r > $O/train_a.txt 2>&1
RUN=$(ls -td logs/rsl_rl/aib1_recover/*_h25r | head -1)
echo "$(date +%H:%M) run $RUN: $(ls $RUN | grep -c model_) checkpoints"
grep -q Traceback $O/train_a.txt && fail $O/train_a.txt train_a
C300=$(ls $RUN/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
idle
timeout 900 $P -u scripts/recover_eval.py --headless --checkpoint $RUN/model_$C300.pt --tag m$C300 > $O/eval_$C300.txt 2>&1
grep -E '^\[reval\]|^REVAL' $O/eval_$C300.txt
G=$(grep '^REVAL' $O/eval_$C300.txt | awk '{print $4}')
[ -n "$G" ] || fail $O/eval_$C300.txt eval_$C300
if ! $P -c "import sys; sys.exit(0 if float('$G') < 0.60 else 1)"; then MSG="FALSIFIED at model_$C300: on the ground $G >= 0.60 - line closed ($RUN)"; exit 0; fi
idle
timeout 18000 $P -u scripts/train.py --task AIB-Recover-v0 --headless --num_envs 4096 --max_iterations 700 --run_name h25r_b --resume True --load_run $(basename $RUN) --checkpoint model_$C300.pt > $O/train_b.txt 2>&1
RUN2=$(ls -td logs/rsl_rl/aib1_recover/*_h25r_b | head -1)
grep -q Traceback $O/train_b.txt && fail $O/train_b.txt train_b
CL=$(ls $RUN2/model_*.pt | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
idle
timeout 900 $P -u scripts/recover_eval.py --headless --checkpoint $RUN2/model_$CL.pt --tag m$CL > $O/eval_$CL.txt 2>&1
grep -E '^\[reval\]|^REVAL' $O/eval_$CL.txt
G2=$(grep '^REVAL' $O/eval_$CL.txt | awk '{print $4}')
[ -n "$G2" ] || fail $O/eval_$CL.txt eval_$CL
V=$($P -c "print('PASS' if float('$G2') < 0.30 else 'FAIL')")
MSG="model_$C300 ground $G, model_$CL ground $G2 -> +1000 bar 0.30 $V ($RUN2)"
