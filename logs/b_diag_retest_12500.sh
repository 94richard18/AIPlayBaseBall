#!/bin/bash
# Retest (decision session 10-25): Human9b model_12500 + aim -0.08 diag with two more seeds, after pitcher ho3b.
# Keep the delivery if the mean of the three (0.684 + two) >= 0.68, else back to Human9 model_12300 + aim -0.12.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
C=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
for s in 1 2; do
    timeout 300 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C --tag h9b_12500_aim-0.08_seed$s --seed $s --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 > logs/h9b_12500_aim-0.08_seed${s}_contact.txt 2>&1
    echo "seed $s: $(grep -o 'hit rate [0-9.]*' logs/h9b_12500_aim-0.08_seed${s}_contact.txt)"
done
$P -c "
import re
v=[0.684]+[float(re.search(r'hit rate ([0-9.]+)',open(f'logs/h9b_12500_aim-0.08_seed{s}_contact.txt').read()).group(1)) for s in (1,2)]
m=sum(v)/3; print('hit rates',v,'mean',round(m,3),'-> keep 12500 + aim -0.08' if m>=0.68 else '-> back to 12300 + aim -0.12')"
date +"%H:%M done"
