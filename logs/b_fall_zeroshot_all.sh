#!/bin/bash
# Fall fix zero-shot (decision session 10-26), requeued after the 17:26 overlap: 4 runs, delivery 12500, fast aim -0.08,
# diag1024 seed 0. Baseline hit 0.684, fall 57%. Pass: |hit - 0.684| <= 0.02 and fall <= 37%.
# Starts only when no train.py / eval_stage.py / showcase.py (pitcher trainer or filming) is seen at two checks 60 s apart.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
free() { [ "$(cnt 'scripts/(train|eval_stage|showcase).py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; }
slot() { while true; do if free; then sleep 60; free && return; fi; sleep 30; done; }
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
for v in "fall_contact_0.6 --post_ref_pose contact --post_ref_blend 0.6" "fall_contact_0.3 --post_ref_pose contact --post_ref_blend 0.3" \
         "fall_start_0.6 --post_ref_pose start --post_ref_blend 0.6" "fall_contact_0.6_keepres --post_ref_pose contact --post_ref_blend 0.6 --post_ref_keep_residual"; do
    set -- $v; tag=$1; shift
    slot
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $DEL --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 "$@" > logs/${tag}_contact.txt 2>&1
    echo "$(date +%H:%M) $tag: $(grep -o 'hit rate [0-9.]*' logs/${tag}_contact.txt) | $($P -c "import json; h=json.load(open('logs/diag_bat_contact_$tag.json'))['summary']['hits']; print(f\"fell {h['fell']}/{h['n']} = {h['fell']/max(h['n'],1)*100:.0f}%  fall after hit s {h.get('fall_after_hit_s')}\")" 2>&1 | tail -1)"
done
echo "FALLZSALL_END $(date +%H:%M)"
