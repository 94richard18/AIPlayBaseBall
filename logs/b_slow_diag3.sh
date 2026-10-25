#!/bin/bash
# Slow-pitch diagnosis 5: delivery vs P3c t18 split by the pitch's plate location (in / out of the StrikeZone). The
# synthetic P3c-shaped slow pitches (near the zone) hit 0.50 like the fast ones, the real P3c table only 0.140 - and
# ~half of P3c's pitches are balls, which diag makes the batter swing at. Runs after b_slow_diag2.sh (SLOWDIAG2_END).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
until grep -q "^SLOWDIAG2_END" logs/b_slow_diag2.txt 2>/dev/null; do sleep 20; done
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
for a in -0.08 0; do
    until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; do sleep 30; done
    tag=slowdiag_del_p3c_zone_a$a
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $DEL --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a --release_csv assets/game/p3c_identical_t18_release.csv > logs/${tag}_contact.txt 2>&1
    echo "== $(date +%H:%M) $tag"; grep -E "hit rate|\[bat\] slow" logs/${tag}_contact.txt
done
echo "SLOWDIAG3_END $(date +%H:%M)"
