#!/bin/bash
# Human13 (Human12 + ref_hold_end + clamp_ref_phase) from Human9b model_12500, 300 its, checkpoint every 50, slot B, one
# batter process at a time. Decision session 10-25: train only if the falsifier passed; early stop at +50 if diag1024 < 0.66.
# Gate per checkpoint: fall-after-hit <= 37%, diag1024 >= 0.68, ONE OUTS >= 17.0%, fair and >= 29 m >= 14.8%;
# lowest fall wins, within 5 points the higher win. Eval in play with the same clamp (PLAYFLAGS).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
PLAYFLAGS="${PLAYFLAGS:---clamp_ref_phase}"
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
diag() {  # $1 checkpoint, $2 tag -> "hit_rate fall%"
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h13_$2 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 $PLAYFLAGS > logs/h13_$2_contact.txt 2>&1
    $P -c "import json; s=json.load(open('logs/diag_bat_contact_h13_$2.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))"
}
date +"%H:%M start Human13 (play flags $PLAYFLAGS)"
$P -u scripts/train.py --task AIB-PitchedBatting-Human13-v0 --headless --num_envs 2048 --max_iterations 50 --run_name human_pitched13 --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > logs/train_human13.txt 2>&1
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched13 | head -1); I1=$(last $D)
read HR FALL <<< "$(diag $D/model_$I1.pt $I1)"; echo "$(date +%H:%M) $I1 diag1024 $HR fall $FALL% (stop if diag < 0.66)"
LIST="$D/model_$I1.pt"
if [ -z "$HR" ] || $P -c "import sys; sys.exit(0 if float('$HR') < 0.66 else 1)"; then echo "early stop at $I1"; else
    $P -u scripts/train.py --task AIB-PitchedBatting-Human13-v0 --headless --num_envs 2048 --max_iterations 250 --run_name human_pitched13b --resume True --load_run $(basename $D) --checkpoint model_$I1.pt >> logs/train_human13.txt 2>&1
    D2=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched13b | head -1); I2=$(last $D2)
    for it in 12650 12700; do [ -f $D2/model_$it.pt ] && LIST="$LIST $D2/model_$it.pt"; done
    LIST="$LIST $D2/model_$I2.pt"
fi
grep -E "bat/(hit_rate|fall_rate|success_rate)" logs/train_human13.txt | tail -3 | tr -s ' '
date +"%H:%M Human13 training ended: $LIST"
for M in $LIST; do
    it=$(basename $M .pt); it=${it#model_}
    if [ -f logs/diag_bat_contact_h13_$it.json ]; then read HR FALL <<< "$($P -c "import json; s=json.load(open('logs/diag_bat_contact_h13_$it.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))")"
    else read HR FALL <<< "$(diag $M $it)"; fi
    if [ "${FALL:-100}" -gt 37 ] || $P -c "import sys; sys.exit(0 if float('${HR:-0}') < 0.68 else 1)"; then echo "$(date +%H:%M) model_$it diag1024 ${HR:-none} | fall ${FALL:-?}% (fails, no showcase)"; continue; fi
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $M --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 $PLAYFLAGS > logs/b_h13_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_h13_$it.txt logs/b_table_h13_$it.json > /dev/null
    F=$($P -c "import json; s=json.load(open('logs/b_table_h13_$it.json'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_h13_$it.json --hit_rate $HR --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$it diag1024 $HR | fall $FALL% | fair >= 29 m $F% | $W"
done
echo "gate: fall <= 37%, diag1024 >= 0.68, ONE OUTS >= 17.0%, fair >= 29 m >= 14.8%; lowest fall, within 5 points the higher win"
date +"%H:%M done"
