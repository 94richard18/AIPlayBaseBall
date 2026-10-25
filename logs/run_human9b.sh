#!/bin/bash
# Human9 continuation (decision session 10-25 13:56): model_12300 -> 12695 (395 its), checkpoint every 50 (yield at the next one).
# Eval: checkpoints 12400 / 12500 / 12600 / 12695, one at a time (slot A only), 1024-ball table each -> CPU ONE OUTS 1000 matches
# (ranking at hit_rate 0.691), then diag on the best and ONE OUTS again at its measured hit rate.
# Gate vs the delivery model_12300: replace only if ONE OUTS >= 13.1% and diag hit rate >= 0.68.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
R=2026-10-25_12-31-03_human_pitched9
date +"%H:%M start Human9b"
$P -u scripts/train.py --task AIB-PitchedBatting-Human9-v0 --headless --num_envs 2048 --max_iterations 395 --run_name human_pitched9b --resume True --load_run $R --checkpoint model_12300.pt > logs/train_human9b.txt 2>&1
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched9b | head -1)
date +"%H:%M Human9b ended $D: $(ls $D | grep model | tr '\n' ' ')"
grep -E "bat/(hit_rate|success_rate|fair_rate|carry:)" logs/train_human9b.txt | tail -4 | tr -s ' '
table() {  # $1 = iteration
    timeout 900 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_$1.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h9b_$1.txt 2>&1
    $P scripts/make_batter_table.py logs/b_aimy_h9b_$1.txt logs/b_aimy_table_h9b_$1.json
}
for it in 12400 12500 12600 12695; do [ -f $D/model_$it.pt ] && table $it; done  # one at a time: slot B stays the pitcher's
best=""; bw=0
for it in 12400 12500 12600 12695; do
    [ -f logs/b_aimy_table_h9b_$it.json ] || continue
    w=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9b_$it.json --hit_rate 0.691 --show 0 2>&1 | grep -o "batter [0-9.]*%" | grep -o "[0-9.]*")
    echo "model_$it ONE OUTS at 0.691: $w%"
    if $P -c "import sys; sys.exit(0 if float('$w') > float('$bw') else 1)"; then best=$it; bw=$w; fi
done
echo "best model_$best ($bw%)"
$P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$best.pt --tag human9b_$best --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h9b_${best}_contact.txt 2>&1
HR=$(grep -o "hit rate [0-9.]*" logs/h9b_${best}_contact.txt | grep -o "[0-9.]*$"); echo "model_$best diag hit rate $HR"
$P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9b_$best.json --hit_rate $HR --show 0 2>&1 | tail -2
echo "gate: replace 12300 if win >= 13.1% and hit rate $HR >= 0.68"
date +"%H:%M done"
