#!/bin/bash
# Human30e (planning 14, 10-31 09:15, rebuilt): Human30c (training timing noise OFF) with the carry knee at 0 m,
# 20 x clamp(carry / 40, 0, 1) x fair (AIB-PitchedBatting-Human30e-v0), checkpoint every 25 its for a mid-run pick.
# 1) after H30D_12650_END (newer than this chain's start): diag model_12650 of human_pitched30c_1 at jitter 0 / 0.03 (gate 300
#    + carry_cols) -> H30C_12650_END (eval, no GO needed). 2) waits for logs/h30e/GO: smoke -> +300 from 12500 -> diag jitter 0
#    + 0.03 of the last checkpoint + the training trend at every saved checkpoint -> H30E_300_END. RAM watchdog 30 GB.
# One Isaac at a time.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=25 AIB_EFFORT_SCALE=1.5
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD
TASK=AIB-PitchedBatting-Human30e-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
O=logs/h30e; mkdir -p $O
M=logs/H30E_300_END.txt; rm -f $M; MSG="aborted"
touch $O/chain.stamp
trap 'echo "H30E_300_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
until [ -f logs/H30D_12650_END.txt ] && [ logs/H30D_12650_END.txt -nt $O/chain.stamp ]; do sleep 30; done
# 1) 30c mid-run checkpoint (14, 09:15): is the mid-run peak + checkpoint pick a general lever?
C12=""
for j in 0 0.03; do
  T=h30c_12650_j$j
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $B/2026-10-31_07-28-04_human_pitched30c_1/model_12650.pt $SAIM     --envs 1024 --seed 0 --tag $T --timing_noise $j --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > logs/h30c/diag_$T.txt 2>&1
  if ! grep -q Traceback logs/h30c/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]; then
    C12="$C12 | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$T.json --label j$j)"
  else C12="$C12 | j$j: crash or no json"; fi
done
echo "H30C_12650_END $(date +%H:%M) model_12650$C12" > logs/H30C_12650_END.txt
until [ -f $O/GO ]; do sleep 60; done
# smoke
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 256 --max_iterations 2 --run_name human_pitched30e_smoke --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_smoke.txt 2>&1
grep -q Traceback $O/train_smoke.txt && fail $O/train_smoke.txt smoke
grep -q "saim_swing" $O/train_smoke.txt || { MSG="smoke: SaimSwingEnv not built"; exit 1; }
# +300
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched30e_1 --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_1.txt 2>&1 &
TP=$!
PK=0
while kill -0 $TP 2>/dev/null; do
  G=$(priv human_pitched30e_1); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'human_pitched30e_1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    MSG="RAM watchdog: trainer private ${G} GB > 30, killed"; exit 1
  fi
  sleep 30
done
grep -q Traceback $O/train_1.txt && fail $O/train_1.txt train_1
D=$(ls -td $B/*human_pitched30e_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag h30e_$I --timing_noise 0 \
  --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_h30e_$I.txt 2>&1
grep -q Traceback $O/diag_h30e_$I.txt && fail $O/diag_h30e_$I.txt "diag $I"
[ -f logs/diag_bat_contact_h30e_$I.json ] || { MSG="diag $I: no json"; exit 1; }
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h30e_$I.json --stage 300 --label "model_$I ($D)" | head -1)
C=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_h30e_$I.json --label h30e_$I)
C9=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_r9wet.json --label r9)
# 14 (10-31 07:25): one jittered diag row (timing noise 0.03 = the showcase's, same 1024 pitches) for launch robustness
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag h30e_${I}_j3 --timing_noise 0.03   --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_h30e_${I}_j3.txt 2>&1
J3="jitter 0.03: crash or no json"
if ! grep -q Traceback $O/diag_h30e_${I}_j3.txt && [ -f logs/diag_bat_contact_h30e_${I}_j3.json ]; then
  J3="jitter 0.03: $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h30e_${I}_j3.json --stage 300 --label j3 | head -1 | cut -c1-200) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_h30e_${I}_j3.json --label j3)"
fi
C0=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_h30c_12799.json --label h30c)
TR=$(for it in $(seq 12525 25 12775) 12799; do L=$(grep -n "iteration $it/" $O/train_1.txt | cut -d: -f1); [ -n "$L" ] && echo -n "$it: $(sed -n "$L,$((L+60))p" $O/train_1.txt | grep -E "bat/(fair_carry|fair_carry_lt10|fair_rate|launch_angle|hit_rate):" | tr -s " " | tr "\n" " ")| "; done)
MSG="$G | $C | $J3 | train $TR | start $C0 | row 9 $C9 | trainer private peak ${PK} GB"
