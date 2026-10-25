#!/bin/bash
# batter B-slot check (10-25, zero-shot, < 30 min): lateral aim offset aim_offset_y on Human7 model_12095 + aim_offset_z -0.12
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
C=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-22_22-46-49_human_pitched7/model_12095.pt
for y in 0.0 -0.08 -0.04 0.04 0.08; do
    timeout 420 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $C --eval_envs 256 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 --aim_offset_y $y > logs/b_aimy_$y.txt 2>&1
    echo "y=$y $(sed -n '/^\[eval\]/,/^}/p' logs/b_aimy_$y.txt | grep -E '"landed"|exit_velo_mps_median|ge28|fair_frac|fair_carry|spray_deg_quartiles' -A3 | tr -d ' \n')"
done
date +"%H:%M done"
