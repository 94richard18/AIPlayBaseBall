#!/bin/bash
# Link chain of the delivery 12500 (decision session 10-29): diag_bat_contact --chain_log --phase_log --link_log, hits only
# (closed loop, AIB_EFFORT_SCALE 1.5, P3c t18, jitter 0, bias 0.055, aim 0, 1024 envs): per-joint contribution to the
# sweet-spot speed at contact, upper-body joint speed peaks + peak times vs contact (robot / reference), torque saturation
# over the last 0.1 s. Which is the lost speed: wrist torque, waist / shoulder torque, or timing?
# Gate: Isaac 0 twice 60 s apart. End marker: the file logs/BSPEED5_END.txt ("BSPEED5_END HH:MM ..."), also on failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED5_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED5_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open (Isaac 0 twice 60 s apart)"
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv

AIB_EFFORT_SCALE=1.5 timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C0 --envs 1024 --seed 0 --tag bspeed5_12500 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log --chain_log --phase_log --link_log > logs/bspeed5_12500_contact.txt 2>&1
B=$(grep "\[bat\] link\|\[bat\] chain_log\|\[bat\] contact_log\|\[bat\] [lrw]" logs/bspeed5_12500_contact.txt)
if [ -n "$B" ]; then echo "$B"; RES="ok"; else echo "failed:"; tail -8 logs/bspeed5_12500_contact.txt; RES="failed"; fi
date +"%H:%M bspeed5 done, isaac after: $(cnt)"
