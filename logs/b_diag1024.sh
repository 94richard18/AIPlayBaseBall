#!/bin/bash
# Batter re-measure (decision session 10-25 18:0x): diag with 1024 pitches (seed 0), gate 0.68 unchanged. Runs in slot B
# once pitcher P2 (zero-shot eval, 1 process) has started, one batter process at a time. Combos passing 0.68 -> ONE OUTS
# 1000 matches on a 1024-ball showcase table; gate diag >= 0.68, ONE OUTS >= 13.1%, fair and >= 29 m >= 11.3%; best win
# = delivery. None passes -> the highest diag is the delivery, report to the decision session.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/eval_stage.py')" -ge 1 ]; do sleep 30; done
date +"%H:%M pitcher P2 running, batter starts in slot B"
R9=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_12-31-03_human_pitched9
R9b=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b
slot() { while [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -ge 2 ] || [ "$(cnt 'scripts/train.py')" != "0" ]; do sleep 30; done; }
for combo in "12300 $R9 -0.12" "12500 $R9b -0.08" "12500 $R9b -0.04" "12350 $R9b -0.08"; do
    set -- $combo; it=$1; C=$2/model_$1.pt; a=$3; tag=d1024_${it}_aim$a
    slot
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a > logs/${tag}_contact.txt 2>&1
    HR=$(grep -o "hit rate [0-9.]*" logs/${tag}_contact.txt | grep -o "[0-9.]*$")
    FALL=$($P -c "import json; h=json.load(open('logs/diag_bat_contact_$tag.json'))['summary']['hits']; print(f\"{h['fell']/max(h['n'],1)*100:.0f}%\")" 2>/dev/null)
    line="$(date +%H:%M) model_$it aim $a: diag1024 ${HR:-none} | fell after hit ${FALL:-?}"
    if [ -n "$HR" ] && $P -c "import sys; sys.exit(0 if float('$HR') >= 0.68 else 1)"; then
        T=logs/b_table_h9_${it}_aim$a.json
        if [ ! -f $T ] || [ "$($P -c "import json; print(len(json.load(open('$T'))['swings']) > 300)")" != "True" ]; then
            slot
            timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $C --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a > logs/b_h9_${it}_aim$a.txt 2>&1
            $P scripts/make_batter_table.py logs/b_h9_${it}_aim$a.txt $T > /dev/null
        fi
        F=$($P -c "import json; s=json.load(open('$T'))['swings']; print(round(sum(1 for b in s if abs(b['spray_deg']) <= 45 and b['carry_m'] >= 29) / 1024 * 100, 1))")
        W=$($P scripts/one_outs.py --matches 1000 --batter_table $T --hit_rate $HR --show 0 2>&1 | tail -2 | tr '\n' ' ')
        line="$line | fair >= 29 m $F% | $W"
    fi
    echo "$line"
done
echo "gate: diag1024 >= 0.68, ONE OUTS >= 13.1%, fair >= 29 m >= 11.3%; none -> highest diag1024 is the delivery"
date +"%H:%M done"
