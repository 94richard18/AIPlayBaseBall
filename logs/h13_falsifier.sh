#!/bin/bash
# Human13 falsifier + play-side clamp check (decision session 10-25 19:1x), slot B, one batter process at a time
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
C=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
slot() { while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/(train|showcase|eval_stage|diag_)' }).Count" | tr -d '\r')" -ge 2 ]; do sleep 30; done; }
for a in 0 -0.08; do
    slot; timeout 600 $P -u scripts/diag_episode_end.py --headless --checkpoint $C --task AIB-PitchedBatting-Human13-v0 --envs 1024 --seed 0 --aim_offset_z $a --timing_noise 0.03 --tag h13_aim$a > logs/diag_episode_end_h13_aim$a.txt 2>&1
    echo "$(date +%H:%M) train env Human13 aim $a: $(grep '\[end\]' logs/diag_episode_end_h13_aim$a.txt | cut -c1-400)"
done
for v in "clamp --clamp_ref_phase" "clampff --clamp_ref_phase --zero_ff_after_ref_end"; do
    set -- $v; tag=play_$1_12500; shift
    slot; timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 "$@" > logs/${tag}_contact.txt 2>&1
    echo "$(date +%H:%M) play $tag: $(grep -o 'hit rate [0-9.]*' logs/${tag}_contact.txt) | $($P -c "import json; h=json.load(open('logs/diag_bat_contact_$tag.json'))['summary']['hits']; print(f\"fell {h['fell']}/{h['n']} = {h['fell']/max(h['n'],1)*100:.0f}%  fall after hit s {h.get('fall_after_hit_s')}\")")"
done
echo "falsifier: Human13 train-env fell + lost >= 40% of hit episodes; play baseline 12500 -0.08: hit 0.684, fell 401/700 = 57%"
date +"%H:%M done"
