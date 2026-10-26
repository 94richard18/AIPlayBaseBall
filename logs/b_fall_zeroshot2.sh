#!/bin/bash
# Fall fix control (decision session 10-26): contact pose, 0.6 s, policy residual kept (not faded) - separates "PD alone
# cannot hold the pose" from "the pose itself is the problem". Runs after b_fall_zeroshot.sh (FALLZS_END).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
until grep -q "^FALLZS_END" logs/b_fall_zeroshot.txt 2>/dev/null; do sleep 20; done
until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; do sleep 30; done
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
tag=fall_contact_0.6_keepres
timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $DEL --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 --post_ref_pose contact --post_ref_blend 0.6 --post_ref_keep_residual > logs/${tag}_contact.txt 2>&1
echo "$(date +%H:%M) $tag fast aim -0.08: $(grep -o 'hit rate [0-9.]*' logs/${tag}_contact.txt) | $($P -c "import json; h=json.load(open('logs/diag_bat_contact_$tag.json'))['summary']['hits']; print(f\"fell {h['fell']}/{h['n']} = {h['fell']/max(h['n'],1)*100:.0f}%  fall after hit s {h.get('fall_after_hit_s')}\")")"
echo "FALLZS2_END $(date +%H:%M)"
