#!/bin/bash
# Delivery batter vs the RL pitcher's P3c pitches (decision session 10-25 19:3x; before Human13 continues):
# Human9b model_12500 + aim -0.08, release_p3c_identical*_t.csv (own release state per pitch): diag 1024 (hit rate, falls)
# + showcase 1024 (batted-ball table) -> CPU ONE OUTS 1000 matches with the same run's pitch_csv as the pitcher.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
C=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/(train|showcase|eval_stage|diag_)' }).Count" | tr -d '\r'; }
until R=$(ls assets/game/release_p3c_identical*_t*.csv 2>/dev/null | head -1) && [ -n "$R" ]; do git pull -q --rebase >/dev/null 2>&1; sleep 60; done
T=$(ls assets/game/pitches_p3c_identical*_t*.csv 2>/dev/null | head -1)
echo "$(date +%H:%M) release $R | pitch table ${T:-missing}"
for step in diag showcase; do
    until [ "$(cnt)" -lt 2 ]; do sleep 20; done
    if [ $step = diag ]; then
        timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --envs 1024 --seed 0 --tag p3ct_12500 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 --release_csv $R > logs/p3ct_12500_contact.txt 2>&1
        HR=$(grep -o "hit rate [0-9.]*" logs/p3ct_12500_contact.txt | grep -o "[0-9.]*$")
        echo "$(date +%H:%M) diag1024 vs P3c: hit rate ${HR:-none} | $($P -c "import json; s=json.load(open('logs/diag_bat_contact_p3ct_12500.json'))['summary']; h=s['hits']; m=s['misses']; print(f\"fell {h['fell']}/{h['n']}, miss dominant {s['miss_dominant']}, ball above bat {s['miss_ball_above']}/{m.get('n',0)}\")" 2>&1 | tail -1)"
    else
        timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $C --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 --release_csv $R > logs/b_p3ct_12500.txt 2>&1
        $P scripts/make_batter_table.py logs/b_p3ct_12500.txt logs/b_table_p3ct_12500.json
    fi
done
[ -n "$T" ] && PT="--pitches $T" || PT=""
echo "ONE OUTS vs P3c ($T): $($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_p3ct_12500.json --hit_rate ${HR:-0} $PT --show 0 2>&1 | tail -2 | tr '\n' ' ')"
echo "ONE OUTS same table vs the fallback pitcher: $($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_p3ct_12500.json --hit_rate ${HR:-0} --show 0 2>&1 | tail -2 | tr '\n' ' ')"
date +"%H:%M done"
