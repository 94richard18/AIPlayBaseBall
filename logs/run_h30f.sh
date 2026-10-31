#!/bin/bash
# (c) falsifier + Human30f (planning 14, 10-31 15:50). After H30C_ADOPT2_END (newer than this chain's start):
# 1) diag model_13797 jitter 0 with --stochastic (the swing policy sampled, as in training) + the same deterministic row's
#    train-comparable stats (scripts/diag_fair_carry.py) vs the training EMA at 13797 (hit 0.82, fair_carry 19.6, launch
#    12.5) -> logs/STOCH_DIAG_END.txt; (c) confirmed = stochastic in-zone hit within 0.05 of 0.82 AND fair_carry within 2 m
#    of 19.6.
# 2) waits for logs/h30f/GO: Human30f (AIB-PitchedBatting-Human30f-v0) resumed from human_pitched30c_long1 model_13298,
#    2 x 250 its (one Isaac at a time: the +250 diags run between the halves), SAVE_EVERY 50 so the +250 checkpoint exists;
#    if (c) is confirmed the action std is copied from 12500 (2.78) and fixed (AIB_STD_FROM + AIB_FIX_STD), else it is free.
#    Two diag rows (gate + carry_cols with the spin median + train-comparable) at +250 -> H30F_250_END and at +500 ->
#    H30F_500_END. RAM watchdog 30 GB, NOBOOK. Markers by EXIT trap.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM
TASK=AIB-PitchedBatting-Human30f-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
H0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
O=logs/h30f; mkdir -p $O
rm -f logs/STOCH_DIAG_END.txt logs/H30F_250_END.txt logs/H30F_500_END.txt
MK=logs/STOCH_DIAG_END.txt; MSG="aborted"
touch $O/chain.stamp
trap '[ -n "$MK" ] && echo "$(basename $MK .txt) $(date +%H:%M) $MSG" > $MK && cat $MK' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
rows() {  # $1 ckpt, $2 tag prefix -> two diag rows
  local OUT=""
  for j in 0 0.03; do
    local TG=${2}_j$j
    idle
    timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $SAIM --envs 1024 --seed 0 --tag $TG --timing_noise $j \
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$TG.txt 2>&1
    if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
      OUT="$OUT | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label j$j) | $($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_$TG.json --label j$j)"
    else OUT="$OUT | j$j: crash or no json"; fi
  done
  echo "$OUT"
}
until [ -f logs/H30C_ADOPT2_END.txt ] && [ logs/H30C_ADOPT2_END.txt -nt $O/chain.stamp ]; do sleep 60; done
# 1) (c) falsifier
D2=$(ls -td $B/*human_pitched30c_long2 | head -1)
TG=h30cL_13797_stoch
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D2/model_13797.pt $SAIM --stochastic --envs 1024 --seed 0 --tag $TG \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$TG.txt 2>&1
grep -q Traceback $O/diag_$TG.txt && fail $O/diag_$TG.txt stoch_diag
[ -f logs/diag_bat_contact_$TG.json ] || { MSG="stoch diag: no json"; exit 1; }
SC=$($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_$TG.json --label stochastic)
DC=$($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_h30cL_13797_j0.json --label deterministic)
CONF=$($P -c "
import re; s = '''$SC'''
h = float(re.search(r'in-zone hit ([0-9.]+)', s)[1]); c = float(re.search(r'fair_carry mean ([0-9.]+)', s)[1])
print(int(abs(h - 0.82) <= 0.05 and abs(c - 19.6) <= 2.0))")
MSG="(c) $([ "$CONF" = 1 ] && echo CONFIRMED || echo NOT confirmed) | $SC | $DC | training EMA at 13797: hit 0.82, fair_carry 19.6, launch 12.5 | stochastic gate: $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label stoch | head -1 | cut -c1-200) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label stoch)"
echo "STOCH_DIAG_END $(date +%H:%M) $MSG" > $MK; MK=""
# 2) Human30f
until [ -f $O/GO ]; do sleep 60; done
if [ "$CONF" = 1 ]; then export AIB_STD_FROM=$(cd $(dirname $H0) && pwd -W)/$(basename $H0) AIB_FIX_STD=1; STD="std fixed at 12500's (2.78)"; else STD="std free"; fi
LR=$(basename $(ls -td $B/*human_pitched30c_long1 | head -1)); LC=model_13298.pt; PK=0
for k in 1 2; do
  MK=logs/H30F_$((k * 250))_END.txt; MSG="aborted ($STD)"
  idle
  $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 250 --run_name human_pitched30f_$k --resume True \
    --load_run $LR --checkpoint $LC > $O/train_$k.txt 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    G=$(priv human_pitched30f_$k); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'human_pitched30f_$k' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: trainer private ${G} GB > 30, killed ($STD)"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $O/train_$k.txt && fail $O/train_$k.txt train_$k
  grep -q "AIB_FIX_STD: std fixed" $O/train_$k.txt && STD="$STD; log: $(grep -o 'std fixed at mean [0-9.]*' $O/train_$k.txt)"
  D=$(ls -td $B/*human_pitched30f_$k 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
  [ -n "$I" ] || { MSG="train_$k: no checkpoint ($STD)"; exit 1; }
  L=$(grep -n "iteration $I/" $O/train_$k.txt | cut -d: -f1)
  TR=$([ -n "$L" ] && sed -n "$L,$((L+70))p" $O/train_$k.txt | grep -E "bat/(hit_rate|fair_carry|fair_rate|launch_angle|exit_velo):|noise std" | tr -s ' ' | tr '\n' ' ')
  MSG="+$((k * 250)) model_$I ($STD)$(rows $D/model_$I.pt h30f_$I) | train $TR | trainer private peak ${PK} GB"
  echo "$(date +%H:%M) $MSG" >> $O/progress.txt
  echo "$(basename $MK .txt) $(date +%H:%M) $MSG" > $MK; MK=""
  LR=$(basename $D); LC=model_$I.pt
done
