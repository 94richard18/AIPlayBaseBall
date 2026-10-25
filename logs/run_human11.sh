#!/bin/bash
# Human11 (Human9 + follow_through_s 0.4 -> 1.2 s) from Human9b model_12500: 300 its, checkpoint every 50. NOT started -
# needs the decision session's go-ahead and a slot. Only one batter Isaac process at a time (slot B beside the pitcher's
# evals): train 100 -> diag -> train the other 200 -> evals one by one.
# Falsifier: after the first 100 its (model_12599) the 1024-pitch diag (seed 0, aim -0.08) fall-after-hit share must be
# <= 47% (delivery 57% - 10), else stop.
# Gate per checkpoint (12599 / 12700 / last), decision session 10-25: fall-after-hit <= 37%, diag1024 >= 0.68 (absolute),
# ONE OUTS >= 17.0%, fair and >= 29 m >= 14.8% (delivery minus ~1 s.e.). Pick the lowest fall; within 5 points the higher win.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
diag() {  # $1 checkpoint path, $2 tag -> "hit_rate fall%"
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h11_$2 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 > logs/h11_$2_contact.txt 2>&1
    $P -c "import json; s=json.load(open('logs/diag_bat_contact_h11_$2.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))"
}
date +"%H:%M start Human11"
$P -u scripts/train.py --task AIB-PitchedBatting-Human11-v0 --headless --num_envs 2048 --max_iterations 100 --run_name human_pitched11 --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > logs/train_human11.txt 2>&1
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched11 | head -1); I1=$(last $D)
read HR FALL <<< "$(diag $D/model_$I1.pt $I1)"; echo "$(date +%H:%M) $I1 diag1024 $HR fall $FALL% (stop above 47%)"
LIST="$D/model_$I1.pt"
if [ -z "$FALL" ] || [ "$FALL" -gt 47 ]; then echo "falsifier: stop at $I1"; else
    $P -u scripts/train.py --task AIB-PitchedBatting-Human11-v0 --headless --num_envs 2048 --max_iterations 200 --run_name human_pitched11b --resume True --load_run $(basename $D) --checkpoint model_$I1.pt >> logs/train_human11.txt 2>&1
    D2=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched11b | head -1); I2=$(last $D2)
    [ -f $D2/model_12700.pt ] && LIST="$LIST $D2/model_12700.pt"
    LIST="$LIST $D2/model_$I2.pt"
fi
date +"%H:%M Human11 training ended: $LIST"
for M in $LIST; do
    it=$(basename $M .pt); it=${it#model_}
    if [ -f logs/diag_bat_contact_h11_$it.json ]; then read HR FALL <<< "$($P -c "import json; s=json.load(open('logs/diag_bat_contact_h11_$it.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))")"
    else read HR FALL <<< "$(diag $M $it)"; fi
    [ "${FALL:-100}" -gt 47 ] && { echo "$(date +%H:%M) model_$it diag1024 ${HR:-none} | fall ${FALL:-?}% (no showcase)"; continue; }
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $M --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 > logs/b_h11_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_h11_$it.txt logs/b_table_h11_$it.json > /dev/null
    F=$($P -c "import json; s=json.load(open('logs/b_table_h11_$it.json'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_h11_$it.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$it diag1024 ${HR:-none} | fall $FALL% | fair >= 29 m $F% | $W"
done
echo "gate: fall <= 37%, diag1024 >= 0.68, ONE OUTS >= 17.0%, fair >= 29 m >= 14.8%; lowest fall, within 5 points the higher win"
date +"%H:%M done"
