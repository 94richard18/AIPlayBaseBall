#!/bin/bash
# ArmAim2c resume (planning 14, 10-31 03:3x): armaim2c_2 was stopped at 03:30 for RAM (private 58.9 GB); the probe
# (logs/armaim2c/memprobe) showed the scripted diag bookkeeping in ArmAimEnv (_rows / _tq / _rl_frac lists) grows
# private memory and accelerates (+0.95 GB in 60 its), flat with AIB_ARMAIM_NOBOOK=1. Resume +350 from model_650 with the
# bookkeeping off (training only; the diag keeps it) -> model_999, +1000 gate -> ARMAIM2_1000_END. RAM watchdog: trainer
# private > 30 GB -> kill (checkpoints stay). One Isaac at a time; markers by EXIT trap.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.5 AIB_FIX_STD=1
TASK=AIB-ArmAim2Comp-v0
GJ=logs/armaim_auth/scripted_gain.json
B=logs/rsl_rl/aib1_armaim2
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim2c
M=logs/ARMAIM2_1000_END.txt
rm -f $M; MSG="aborted"
touch $O/chain2d.stamp
trap 'echo "ARMAIM2_1000_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
read SW WTAG < logs/h29adopt/winner.txt
export AIB_SAIM_SWING=$(cd $(dirname $SW) && pwd -W)/$(basename $SW)
export AIB_SAIM_BOUND=$(grep -o "aim_bound [0-9.]*" logs/saim/progress.txt | tail -1 | awk '{print $2}')
DELTA=$($P -c "print(0.2 / $AIB_SAIM_BOUND)")
idle
AIB_ARMAIM_NOBOOK=1 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 350 --run_name armaim2c_3 --resume True \
  --load_run 2026-10-31_02-20-04_armaim2c_2 --checkpoint model_650.pt > $O/train_3.txt 2>&1 &
TP=$!
PK=0
while kill -0 $TP 2>/dev/null; do
  G=$(priv armaim2c_3); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'armaim2c_3' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    MSG="RAM watchdog: trainer private ${G} GB > 30, killed"; exit 1
  fi
  sleep 30
done
grep -q Traceback $O/train_3.txt && fail $O/train_3.txt train_3
D3=$(ls -td $B/*armaim2c_3 2>/dev/null | head -1); I3=$([ -n "$D3" ] && last $D3)
[ -n "$I3" ] || { MSG="train_3: no checkpoint"; exit 1; }
idle
timeout 1200 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $D3/model_$I3.pt --aim_scripted $GJ --aim_hold --aim_bound $AIB_SAIM_BOUND --aim_delta $DELTA --aim_window 0.60 --percept_filter --aim_obs_comp --aim_obs_comp_full --envs 1024 --seed 0 --tag armaim2c_$I3 \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_armaim2c_$I3.txt 2>&1
grep -q Traceback $O/diag_armaim2c_$I3.txt && fail $O/diag_armaim2c_$I3.txt "diag $I3"
[ -f logs/diag_bat_contact_armaim2c_$I3.json ] || { MSG="diag $I3: no json"; exit 1; }
G2=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_armaim2c_$I3.json --stage 1000 --label "model_$I3 ($D3)" | tr '\n' ' ')
MSG="$G2 | trainer private peak ${PK} GB (bookkeeping off) | +300: $(cut -c17-120 logs/ARMAIM2_300_END.txt)"
