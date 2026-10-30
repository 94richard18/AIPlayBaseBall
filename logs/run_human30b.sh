#!/bin/bash
# Human30b (planning 14, 10-31 06:01): Human30 with the 10-31 reward rule applied (AIB-PitchedBatting-Human30b-v0:
# Gaussian launch exp(-((LA - 20) / 15)^2 / 2) x q on fair balls only, exit hinge fair only, spray 10 x q, w_miss 5;
# everything else as Human30: 12500 @ 1.5x resume, saim in the loop bound 0.4, filtered percept, window 0.60).
# After TEMPO_FALSIFIER_END (newer than this chain's start). smoke -> +300 -> diag (gate 300 + carry_cols) -> H30B_300_END;
# stops at +300 for 14's call. RAM watchdog: trainer private > 30 GB -> kill. One Isaac at a time; markers by EXIT trap.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.5
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD
TASK=AIB-PitchedBatting-Human30b-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
O=logs/h30b; mkdir -p $O
M=logs/H30B_300_END.txt; rm -f $M; MSG="aborted"
touch $O/chain.stamp
trap 'echo "H30B_300_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
until [ -f logs/TEMPO_FALSIFIER_END.txt ] && [ logs/TEMPO_FALSIFIER_END.txt -nt $O/chain.stamp ]; do sleep 30; done
# smoke
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 256 --max_iterations 2 --run_name human_pitched30b_smoke --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_smoke.txt 2>&1
grep -q Traceback $O/train_smoke.txt && fail $O/train_smoke.txt smoke
grep -q "saim_swing" $O/train_smoke.txt || { MSG="smoke: SaimSwingEnv not built"; exit 1; }
# +300
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched30b_1 --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $O/train_1.txt 2>&1 &
TP=$!
PK=0
while kill -0 $TP 2>/dev/null; do
  G=$(priv human_pitched30b_1); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
  if [ "$G" -gt 30 ] 2>/dev/null; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'human_pitched30b_1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    MSG="RAM watchdog: trainer private ${G} GB > 30, killed"; exit 1
  fi
  sleep 30
done
grep -q Traceback $O/train_1.txt && fail $O/train_1.txt train_1
D=$(ls -td $B/*human_pitched30b_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt $SAIM --envs 1024 --seed 0 --tag h30b_$I --timing_noise 0 \
  --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_h30b_$I.txt 2>&1
grep -q Traceback $O/diag_h30b_$I.txt && fail $O/diag_h30b_$I.txt "diag $I"
[ -f logs/diag_bat_contact_h30b_$I.json ] || { MSG="diag $I: no json"; exit 1; }
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h30b_$I.json --stage 300 --label "model_$I ($D)" | head -1)
C=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_h30b_$I.json --label h30b_$I)
C9=$($P scripts/carry_cols.py --diag logs/diag_bat_contact_r9wet.json --label r9)
MSG="$G | $C | start $C9 | trainer private peak ${PK} GB"
