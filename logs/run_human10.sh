#!/bin/bash
# Human10 (Human9 + w_miss 20 -> 40) from Human9 model_12300: 400 its, checkpoint every 50. NOT started automatically -
# needs the decision session's go-ahead, after pitcher ho2 (no batter GPU while a pitcher trainer runs, decision session 15:48).
# Falsifier: at 12400 (+100) diag hit rate (+-0.03 s, aim -0.12) must be >= 0.65 (Human9 at 12400: 0.484), else stop.
# Gate per checkpoint 12400 / 12500 / 12600 / 12700: diag >= 0.68 and ONE OUTS (1024-ball table, measured hit rate) >= 13.1%;
# best win among the passing ones replaces model_12300.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
date +"%H:%M start Human10"
$P -u scripts/train.py --task AIB-PitchedBatting-Human10-v0 --headless --num_envs 2048 --max_iterations 400 --run_name human_pitched10 --resume True --load_run 2026-10-25_12-31-03_human_pitched9 --checkpoint model_12300.pt > logs/train_human10.txt 2>&1 &
TP=$!
D=""; until [ -n "$D" ]; do sleep 30; D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched10 2>/dev/null | head -1); done
until [ -f $D/model_12400.pt ] || ! kill -0 $TP 2>/dev/null; do sleep 30; done
sleep 20
$P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_12400.pt --tag human10_12400 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h10_12400_contact.txt 2>&1
HR=$(grep -o "hit rate [0-9.]*" logs/h10_12400_contact.txt | grep -o "[0-9.]*$"); echo "$(date +%H:%M) 12400 diag $HR"
if [ -z "$HR" ] || $P -c "import sys; sys.exit(0 if float('$HR') < 0.65 else 1)"; then echo "falsifier: stop (12400 diag ${HR:-none} < 0.65)"; kill $TP; fi
wait $TP
date +"%H:%M Human10 ended: $(ls $D | grep model | tr '\n' ' ')"
for it in 12400 12500 12600 12700; do
    [ -f $D/model_$it.pt ] || continue
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_$it.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h10_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_aimy_h10_$it.txt logs/b_aimy_table_h10_$it.json
    [ -f logs/h10_${it}_contact.txt ] || timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$it.pt --tag human10_$it --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h10_${it}_contact.txt 2>&1
    HR=$(grep -o "hit rate [0-9.]*" logs/h10_${it}_contact.txt | grep -o "[0-9.]*$")
    echo "$(date +%H:%M) model_$it diag ${HR:-none} | $($P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h10_$it.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')"
done
echo "gate: diag >= 0.68 and ONE OUTS >= 13.1%; best win replaces model_12300"
date +"%H:%M done"
