#!/bin/bash
# Slow-pitch diagnosis (decision session 10-25 20:3x), GPU part after pitcher C: per pitch-speed group hit rate,
# early / late misses, tempo action in flight, reference clock at the plate (diag_bat_contact.py by_speed).
#  1. Human14 12599 in its training mix (half P3c-shaped 55-75 km/h, half 88-112), aim 0 (training aim)
#  2. delivery 12500, same mix, aim 0
#  3. delivery 12500 vs P3c t18 releases, aim -0.08 (game setting)
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
R=assets/game/p3c_identical_t18_release.csv
H14=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_20-10-51_human_pitched14/model_12599.pt
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
for job in "slowdiag_h14_mix $H14 0 --mix_release_csv $R" "slowdiag_del_mix $DEL 0 --mix_release_csv $R" "slowdiag_del_p3c $DEL -0.08 --release_csv $R"; do
    set -- $job; tag=$1; C=$2; a=$3; shift 3
    until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; do sleep 30; done
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a "$@" > logs/${tag}_contact.txt 2>&1
    echo "== $(date +%H:%M) $tag (aim $a $*)"; grep -E "hit rate|by_speed|\[bat\] (slow|fast)" logs/${tag}_contact.txt
done
echo "SLOWDIAG_END $(date +%H:%M)"
