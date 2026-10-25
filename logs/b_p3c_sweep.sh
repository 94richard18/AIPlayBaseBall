#!/bin/bash
# Zero-shot timing x aim sweep vs P3c t18 (decision session 10-25 19:4x), delivery 12500, diag 1024 each, slot B, cap 20 min.
# "Swing 0.03 / 0.06 / 0.09 s later": timing_bias_s + = the pitch arrives later (the batter early), so later swing = smaller
# bias: 0.055 - {0.03, 0.06, 0.09} = 0.025 / -0.005 / -0.035. Order: aim 0 first (misses are under the ball), then -0.04, -0.08.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
C=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/(train|showcase|eval_stage|diag_)' }).Count" | tr -d '\r'; }
until [ "$(cnt)" -lt 2 ]; do sleep 15; done
T0=$(date +%s); date +"%H:%M sweep starts (cap 20 min)"
echo "baseline (bias 0.055, aim -0.08): hit 0.140, ball above bat 692/881, misses early (timing < 0) see p3ct_12500"
for a in 0 -0.04 -0.08; do
    for b in 0.025 -0.005 -0.035; do
        [ $(( $(date +%s) - T0 )) -ge 1020 ] && { echo "cap: stop before aim $a bias $b"; break 2; }
        until [ "$(cnt)" -lt 2 ]; do sleep 15; done
        tag=p3cs_a${a}_b$b
        timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias $b --aim_offset_z $a --release_csv $R > logs/${tag}_contact.txt 2>&1
        echo "$(date +%H:%M) aim $a bias $b: $($P -c "
import json, numpy as np
d = json.load(open('logs/diag_bat_contact_$tag.json')); s = d['summary']; pp = d['per_pitch']
hit = np.array(pp['hit'], bool); t = np.array(pp['timing'] if 'timing' in pp else np.array(pp['t_square']) - np.array(pp['t_plane']), float)
m = ~hit; early = int(((t < 0) & m).sum()); late = int(((t > 0) & m).sum())
print(f\"hit {s['hit_rate']:.3f} | misses {int(m.sum())}: early {early} late {late} ball above bat {s['miss_ball_above']} | fell after hit {s['hits'].get('fell', 0)}/{s['hits'].get('n', 0)}\")" 2>&1 | tail -1)"
    done
done
date +"%H:%M done"
