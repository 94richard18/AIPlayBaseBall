#!/bin/bash
# Fall fix, zero-shot (no training; proposal 10-26 17:4x): after the reference clip ends (~0.4 s after contact), blend the
# joint targets to a balanced pose and fade the residual / feed-forward out. Pre-contact behaviour is unchanged, so the
# hit rate must stay (falsifier: |hit - baseline| <= 0.02). Delivery 12500, diag1024 seed 0.
# Baselines: fast 88-112 aim -0.08: hit 0.684, fall 57%; P3c aim 0: in-zone 0.327.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
for v in "contact 0.6" "contact 0.3" "start 0.6"; do
    set -- $v; tag=fall_${1}_$2
    until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; do sleep 30; done
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $DEL --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 --post_ref_pose $1 --post_ref_blend $2 > logs/${tag}_contact.txt 2>&1
    echo "$(date +%H:%M) $tag fast aim -0.08: $(grep -o 'hit rate [0-9.]*' logs/${tag}_contact.txt) | $($P -c "import json; h=json.load(open('logs/diag_bat_contact_$tag.json'))['summary']['hits']; print(f\"fell {h['fell']}/{h['n']} = {h['fell']/max(h['n'],1)*100:.0f}%  fall after hit s {h.get('fall_after_hit_s')}\")")"
done
echo "FALLZS_END $(date +%H:%M)"
