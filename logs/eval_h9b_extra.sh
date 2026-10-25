#!/bin/bash
# Human9b extra checkpoints (decision session 10-25 15:2x): 12350 and 12694 (12650 only if both fail), slot B, one Isaac at a time
# via isaac() (waits for ho2's eval gaps), hard cap 15 min per checkpoint. Gate: diag >= 0.68 and ONE OUTS >= 13.1%; best win replaces 12300.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
D=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b
isaac() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/(train|showcase|eval_stage|diag_)' }).Count" | tr -d '\r'; }
passed=0
one() {
    while [ "$(isaac)" -ge 2 ]; do sleep 30; done
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_$1.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h9b_$1.txt 2>&1
    $P scripts/make_batter_table.py logs/b_aimy_h9b_$1.txt logs/b_aimy_table_h9b_$1.json
    while [ "$(isaac)" -ge 2 ]; do sleep 30; done
    timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$1.pt --tag human9b_$1 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h9b_$1_contact.txt 2>&1
    HR=$(grep -o "hit rate [0-9.]*" logs/h9b_$1_contact.txt | grep -o "[0-9.]*$")
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9b_$1.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$1 diag ${HR:-none} | $W"
    if [ -n "$HR" ] && $P -c "import sys; sys.exit(0 if float('$HR') >= 0.68 else 1)"; then
        w=$(echo "$W" | grep -o "batter [0-9.]*%" | grep -o "[0-9.]*")
        $P -c "import sys; sys.exit(0 if float('$w') >= 13.1 else 1)" && { echo "model_$1 PASSES (diag $HR, win $w%)"; passed=1; }
    fi
}
one 12350
one 12694
[ $passed = 0 ] && one 12650
date +"%H:%M done"
