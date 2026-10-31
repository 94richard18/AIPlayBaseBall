#!/bin/bash
# Human30c at 800 Hz physics (planning 14, 10-31 19:25): the dt is the only variable against model_13298. Waits for
# JTIME_END (newer than this chain's start) AND logs/h30c800/GO (14's) AND idle; runs only if JTIME says dt is the cause
# (pooled native 800 Hz rows of 13298 on the 4 joint clips: in-zone hit <= the 400 Hz row - 0.15 OR ss_max <= 400 Hz - 3 m/s),
# else H30C800_END "dropped". Then:
#  B800 = 13298 on the 4 joint clips together at 800 Hz (baseline row, same shape as the gate rows);
#  fine-tune 13298 on AIB-PitchedBatting-Human30c-v0 at AIB_PHYSICS_DT=0.00125 (decimation 8, control 0.01 s), std free,
#  +300 its (SAVE_EVERY 50); rows at +150 and +300 = native 800 Hz diag of the 4 joint clips (hit, exit / launch med,
#  ss_max, dz) + the +300 gate, phase 2.6, saim, j0, real heights; if +300 moved toward the 400 Hz row (hit > B800 + 0.05)
#  but is not there (hit < N4_400 - 0.05 or launch med < 10), +200 more and a row at +500 (14: without asking);
#  then joint s6 at phase 2.6 with the last checkpoint (R4 recover if RECOV4 passed, else R3) -> H30C800_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50 AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
TASK=AIB-PitchedBatting-Human30c-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
GY=logs/armaim_auth/scripted_gain.json
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
SAIM="--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
D1=$(ls -td $B/*human_pitched30c_long1 | head -1); H=$D1/model_13298.pt
O=logs/h30c800; mkdir -p $O
M=logs/H30C800_END.txt; rm -f $M; MSG="aborted"; OUT=""
touch $O/chain.stamp
trap 'echo "H30C800_END $(date +%H:%M) $MSG$OUT" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
priv() { powershell.exe -NoProfile -Command "\$x = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' } | Select-Object -First 1; if (\$x) { [int]((Get-Process -Id \$x.ProcessId).PrivateMemorySize64/1GB) } else { -1 }" | tr -d '\r'; }
# pooled in-zone stats of diag jsons -> "hit exit_med launch_med ss_max dz" (helper in the scratchpad: CPU, not on scripts/)
stats() { $P $SC/pool_stats.py "$@"; }
row() {  # $1 ckpt, $2 tag -> appends to OUT, sets RS (stats)
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $SAIM --envs 1024 --seed 0 --tag $2 --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --physics_dt 0.00125 \
    --contact_log > $O/diag_$2.txt 2>&1
  grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt diag_$2
  RS=$(stats logs/diag_bat_contact_$2.json)
  OUT="$OUT | $2 (800 Hz, 4 clips): hit / exit / launch / ss_max / dz $RS ; $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$2.json --stage 300 --label $2 | head -1 | cut -c1-160)"
}
train() {  # $1 load run, $2 ckpt, $3 its, $4 run name
  idle
  AIB_PHYSICS_DT=0.00125 $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations $3 --run_name $4 --resume True \
    --load_run $1 --checkpoint $2 > $O/train_$4.txt 2>&1 &
  local TP=$! PK=0 G
  while kill -0 $TP 2>/dev/null; do
    G=$(priv $4); [ "$G" -gt $PK ] 2>/dev/null && PK=$G
    if [ "$G" -gt 30 ] 2>/dev/null; then
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$4' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      MSG="RAM watchdog: trainer private ${G} GB > 30, killed"; exit 1
    fi
    sleep 30
  done
  grep -q Traceback $O/train_$4.txt && fail $O/train_$4.txt train_$4
  grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/train_$4.txt || { MSG="train_$4: no AIB_PHYSICS_DT line"; exit 1; }
  OUT="$OUT | train $4: $(grep -o 'Iteration time: [0-9.]*s' $O/train_$4.txt | tail -1), private peak ${PK} GB"
}
until [ -f logs/JTIME_END.txt ] && [ logs/JTIME_END.txt -nt $O/chain.stamp ] && [ -f $O/GO ]; do sleep 60; done
J8=$(stats logs/diag_bat_contact_N8_obp_167-7.json logs/diag_bat_contact_N8_obp_282-1.json logs/diag_bat_contact_N8_obp_42-1.json logs/diag_bat_contact_N8_obp_42-2.json 2>/dev/null)
J4=$(stats logs/diag_bat_contact_N4_400.json 2>/dev/null)
[ -n "$J8" ] && [ -n "$J4" ] || { MSG="dropped: JTIME rows missing"; exit 0; }
DT=$($P -c "a = list(map(float, '$J8'.split())); b = list(map(float, '$J4'.split())); print(int(a[0] <= b[0] - 0.15 or a[3] <= b[3] - 3.0))")
OUT=" | JTIME pooled N8 (800 Hz): $J8 | N4_400: $J4 (hit / exit / launch / ss_max / dz)"
[ "$DT" = 1 ] || { MSG="dropped: native 800 ~ native 400 (dt not the cause)"; exit 0; }
MSG="running"
row $H h30c800_base; B8=$RS
train $(basename $D1) model_13298.pt 300 human_pitched30c800_1
D=$(ls -td $B/*human_pitched30c800_1 | head -1); I=$(last $D)
row $D/model_$((13298 + 150)).pt h30c800_$((13298 + 150))
row $D/model_$I.pt h30c800_$I; R3S=$RS
EXT=$($P -c "a = list(map(float, '$R3S'.split())); b = list(map(float, '$B8'.split())); c = list(map(float, '$J4'.split())); print(int(a[0] > b[0] + 0.05 and (a[0] < c[0] - 0.05 or a[2] < 10)))")
if [ "$EXT" = 1 ]; then
  train $(basename $D) model_$I.pt 200 human_pitched30c800_2
  D=$(ls -td $B/*human_pitched30c800_2 | head -1); I=$(last $D)
  row $D/model_$I.pt h30c800_$I
fi
RC=$([ -f logs/recov4/recover_ckpt.txt ] && cat logs/recov4/recover_ckpt.txt || cat logs/recov3/recover_ckpt.txt)
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/model_$I.pt --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --phase_horizon 2.6 --trig_dump $O/trig_joint_s6.npz \
  --csv $O/joint_s6.csv > $O/joint_s6.txt 2>&1
grep -q Traceback $O/joint_s6.txt && fail $O/joint_s6.txt joint_s6
OUT="$OUT | joint s6 ph2.6 model_$I (recover $(basename $(dirname $RC))): $($P $SC/joint_row.py $O/joint_s6.csv)"
MSG="done (EXT $EXT, last model_$I)"
