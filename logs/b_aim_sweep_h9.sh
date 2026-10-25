#!/bin/bash
# Batter aim sweep (decision session 10-25 16:2x, after pitcher ho3): Human9 learned to lift at training aim 0, so the game's
# aim -0.12 cuts under twice (12500: diag 0.539 at -0.12, 0.793 at 0). Checkpoints 12300 / 12500 / 12694 x aim 0 / -0.04 / -0.08:
# diag first; diag >= 0.68 -> showcase 1024 + ONE OUTS at the measured rate. Gate: diag >= 0.68, ONE OUTS >= 13.1%,
# fair and >= 29 m >= 11.3% of pitches; best win replaces model_12300 + aim -0.12 (parameter change only) unless its fall-after-hit
# share is > 10 points above 12300's 54% (11-26 match video) - then report to the decision session first.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
declare -A CK=([12300]=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_12-31-03_human_pitched9/model_12300.pt
               [12500]=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
               [12694]=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12694.pt)
for it in 12300 12500 12694; do
    for a in 0 -0.04 -0.08; do
        tag=h9_${it}_aim$a
        if [ "$tag" = "h9_12500_aim0" ]; then cp logs/h9b_12500_aim0_contact.txt logs/${tag}_contact.txt
        else timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint ${CK[$it]} --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a > logs/${tag}_contact.txt 2>&1; fi
        HR=$(grep -o "hit rate [0-9.]*" logs/${tag}_contact.txt | grep -o "[0-9.]*$")
        J=logs/diag_bat_contact_${tag}.json; [ "$tag" = "h9_12500_aim0" ] && J=logs/diag_bat_contact_human9b_12500_aim0.json
        FALL=$($P -c "import json; h=json.load(open('$J'))['summary']['hits']; print(f\"{h['fell']}/{h['n']} = {h['fell']/max(h['n'],1)*100:.0f}%\")" 2>/dev/null)
        line="$(date +%H:%M) model_$it aim $a: diag ${HR:-none} | fell after hit $FALL (12300 at -0.12: 95/177 = 54%)"
        if [ -n "$HR" ] && $P -c "import sys; sys.exit(0 if float('$HR') >= 0.68 else 1)"; then
            timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint ${CK[$it]} --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a > logs/b_${tag}.txt 2>&1
            $P scripts/make_batter_table.py logs/b_${tag}.txt logs/b_table_${tag}.json > /dev/null
            F=$($P -c "import json; s=json.load(open('logs/b_table_$tag.json'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
            W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_$tag.json --hit_rate $HR --show 0 2>&1 | tail -2 | tr '\n' ' ')
            line="$line | fair >= 29 m $F% | $W"
        fi
        echo "$line"
    done
done
echo "gate: diag >= 0.68, ONE OUTS >= 13.1%, fair >= 29 m >= 11.3%; fall after hit > 64% (12300 + 10 points) -> report before replacing"
date +"%H:%M done"
