#!/bin/bash
# Human11 (Human9 + follow_through_s 0.4 -> 1.2 s) from Human9b model_12500: 300 its, checkpoint every 50. NOT started -
# needs the decision session's go-ahead and a slot. Eval at the delivery aim -0.08 with the 1024-pitch diag (seed 0).
# Falsifier: at 12600 (+100) the diag fall-after-hit share must be <= 47% (delivery 57% - 10), else stop.
# Gate per checkpoint 12600 / 12700 / 12800 (decision session): fall-after-hit <= 37% (57% - 20) and no regression on the
# other three vs the delivery: diag1024 >= 0.684, ONE OUTS >= 18.5%, fair and >= 29 m >= 15.9%.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
date +"%H:%M start Human11"
$P -u scripts/train.py --task AIB-PitchedBatting-Human11-v0 --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched11 --resume True --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > logs/train_human11.txt 2>&1 &
TP=$!
D=""; until [ -n "$D" ]; do sleep 30; D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched11 2>/dev/null | head -1); done
diag() {  # $1 iteration -> "hit_rate fall%"
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$1.pt --envs 1024 --seed 0 --tag h11_$1 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 > logs/h11_$1_contact.txt 2>&1
    $P -c "import json; s=json.load(open('logs/diag_bat_contact_h11_$1.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))"
}
until [ -f $D/model_12600.pt ] || ! kill -0 $TP 2>/dev/null; do sleep 30; done
if kill -0 $TP 2>/dev/null; then
    sleep 20; read HR FALL <<< "$(diag 12600)"; echo "$(date +%H:%M) 12600 diag $HR fall $FALL% (stop above 47%)"
    if [ -z "$FALL" ] || [ "$FALL" -gt 47 ]; then echo "falsifier: stop at 12600"; kill $TP; fi
fi
wait $TP
date +"%H:%M Human11 ended: $(ls $D | grep model | tr '\n' ' ')"
for it in 12600 12700 12800; do
    [ -f $D/model_$it.pt ] || continue
    if [ -f logs/diag_bat_contact_h11_$it.json ]; then read HR FALL <<< "$($P -c "import json; s=json.load(open('logs/diag_bat_contact_h11_$it.json'))['summary']; h=s['hits']; print(round(s['hit_rate'],3), round(h['fell']/max(h['n'],1)*100))")"; else read HR FALL <<< "$(diag $it)"; fi
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_$it.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 > logs/b_h11_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_h11_$it.txt logs/b_table_h11_$it.json > /dev/null
    F=$($P -c "import json; s=json.load(open('logs/b_table_h11_$it.json'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_h11_$it.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$it diag1024 ${HR:-none} | fall $FALL% | fair >= 29 m $F% | $W"
done
echo "gate: fall <= 37%, diag1024 >= 0.684, ONE OUTS >= 18.5%, fair >= 29 m >= 15.9%"
date +"%H:%M done"
