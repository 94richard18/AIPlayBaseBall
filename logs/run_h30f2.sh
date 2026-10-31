#!/bin/bash
# Human30f only (re-gated by planning 14, 10-31 18:56): GPU order R4 -> JTIME joint s6 -> adopt film -> 30f. Waits for
# logs/ADOPT_FILM_END.txt newer than this chain's start AND logs/h30f/GO (14's go-ahead), then the 2 x 250 Human30f
# segments of run_h30f.sh (step 2 unchanged; (c) was not confirmed -> std free) -> H30F_250_END / H30F_500_END.
# Diag rows at --phase_horizon 2.6 (14, 17:58: every game-level number at the training phase scale).
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
MK=""; MSG="aborted"
touch $O/chain2.stamp
rm -f logs/H30F_250_END.txt logs/H30F_500_END.txt
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
      --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --contact_log > $O/diag_$TG.txt 2>&1
    if ! grep -q Traceback $O/diag_$TG.txt && [ -f logs/diag_bat_contact_$TG.json ]; then
      OUT="$OUT | $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$TG.json --stage 300 --label j$j | head -1) | $($P scripts/carry_cols.py --diag logs/diag_bat_contact_$TG.json --label j$j) | $($P scripts/diag_fair_carry.py --diag logs/diag_bat_contact_$TG.json --label j$j)"
    else OUT="$OUT | j$j: crash or no json"; fi
  done
  echo "$OUT"
}
# 2) Human30f
until [ -f logs/ADOPT_FILM_END.txt ] && [ logs/ADOPT_FILM_END.txt -nt $O/chain2.stamp ] && [ -f $O/GO ]; do sleep 60; done
CONF=0  # STOCH_DIAG_END: (c) NOT confirmed -> std free
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
