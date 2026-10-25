#!/bin/bash
# Human10 (Human9 + w_miss 20 -> 40) from Human9 model_12300: 400 its, checkpoint every 50. NOT started automatically -
# start after pitcher ho2 ends (decision session approved 10-25; no batter GPU while a pitcher trainer runs).
# Falsifiers (decision session): diag hit rate (+-0.03 s, aim -0.12) at 12350 (+50) >= 0.66 and at 12400 (+100) >= 0.65,
# else stop (Human9: 0.629 / 0.484). Gate per checkpoint 12400 / 12500 / 12600 / 12700: diag >= 0.68, ONE OUTS (1024-ball
# table, measured hit rate) >= 13.1% and fair and >= 29 m >= 11.3% of pitches (12300; guards a retreat to safe grounders);
# best win among the passing ones replaces model_12300.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
date +"%H:%M start Human10"
$P -u scripts/train.py --task AIB-PitchedBatting-Human10-v0 --headless --num_envs 2048 --max_iterations 400 --run_name human_pitched10 --resume True --load_run 2026-10-25_12-31-03_human_pitched9 --checkpoint model_12300.pt > logs/train_human10.txt 2>&1 &
TP=$!
D=""; until [ -n "$D" ]; do sleep 30; D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched10 2>/dev/null | head -1); done
diag() {  # $1 iteration -> logs/h10_$1_contact.txt, prints the hit rate
    timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$1.pt --tag human10_$1 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h10_$1_contact.txt 2>&1
    grep -o "hit rate [0-9.]*" logs/h10_$1_contact.txt | grep -o "[0-9.]*$"
}
check() {  # $1 iteration, $2 minimum diag hit rate
    until [ -f $D/model_$1.pt ] || ! kill -0 $TP 2>/dev/null; do sleep 30; done
    kill -0 $TP 2>/dev/null || return
    sleep 20
    HR=$(diag $1); echo "$(date +%H:%M) $1 diag ${HR:-none} (stop below $2)"
    if [ -z "$HR" ] || $P -c "import sys; sys.exit(0 if float('$HR') < $2 else 1)"; then echo "falsifier: stop at $1"; kill $TP; fi
}
check 12350 0.66
check 12400 0.65
wait $TP
date +"%H:%M Human10 ended: $(ls $D | grep model | tr '\n' ' ')"
for it in 12400 12500 12600 12700; do
    [ -f $D/model_$it.pt ] || continue
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_$it.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h10_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_aimy_h10_$it.txt logs/b_aimy_table_h10_$it.json
    if [ -f logs/h10_${it}_contact.txt ]; then HR=$(grep -o "hit rate [0-9.]*" logs/h10_${it}_contact.txt | grep -o "[0-9.]*$"); else HR=$(diag $it); fi
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h10_$it.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')
    F=$($P -c "import json; s=json.load(open('logs/b_aimy_table_h10_$it.json'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
    echo "$(date +%H:%M) model_$it diag ${HR:-none} | fair >= 29 m $F% | $W"
done
echo "gate: diag >= 0.68, ONE OUTS >= 13.1% and fair >= 29 m >= 11.3%; best win replaces model_12300"
date +"%H:%M done"
