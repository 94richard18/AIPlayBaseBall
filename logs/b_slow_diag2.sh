#!/bin/bash
# Slow-pitch diagnosis 4 (decision session 10-25 21:0x): delivery vs P3c with the tempo held at 0 once the ball is in the air.
# Tempo hypothesis holds if the hit rate rises from 0.140 to >= 0.30. Runs after b_slow_diag.sh (marker SLOWDIAG_END).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '$1' }).Count" | tr -d '\r'; }
until grep -q "^SLOWDIAG_END" logs/b_slow_diag.txt 2>/dev/null; do sleep 30; done
until [ "$(cnt 'scripts/train.py')" = "0" ] && [ "$(cnt 'scripts/(train|showcase|eval_stage|diag_)')" -lt 2 ]; do sleep 30; done
DEL=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
tag=slowdiag_del_p3c_tempo0
timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $DEL --envs 1024 --seed 0 --tag $tag --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.08 --release_csv assets/game/p3c_identical_t18_release.csv --tempo_zero_after_release > logs/${tag}_contact.txt 2>&1
echo "== $(date +%H:%M) $tag (tempo 0 after release; hypothesis holds if hit >= 0.30, was 0.140)"; grep -E "hit rate|\[bat\] (slow|fast)|miss dominant" logs/${tag}_contact.txt
echo "SLOWDIAG2_END $(date +%H:%M)"
