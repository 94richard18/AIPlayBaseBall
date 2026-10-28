#!/bin/bash
# Open loop vs trained policy (decision session 10-29): is the pelvis yaw ratio ~0.55 real or an open-loop artifact?
# (a) diag_bat_speed F0 (human 1.5x, open loop, 16 envs) with foot lift + sole tilt;
# (b) diag_bat_contact --body_log on the delivery 12500 (closed loop, AIB_EFFORT_SCALE 1.5, P3c t18, timing jitter 0,
#     bias 0.055, aim 0, 1024 envs): pelvis yaw / foot translation / foot yaw / lift / tilt vs the reference over the last
#     0.3 s of reference time before contact, split by hit / miss and fell / stood.
# Policy ratio ~0.55 -> real whole-body cap (next lever: a training reward); >= 0.8 -> the open-loop 0.56 was an artifact.
# Gate: Isaac 0 twice 60 s apart. End marker: the file logs/BSPEED4_END.txt ("BSPEED4_END HH:MM ..."), also on failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED4_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED4_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open (Isaac 0 twice 60 s apart)"
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv

timeout 600 $P -u scripts/diag_bat_speed.py --headless --envs 16 --tail 1.0 --strength human --effort_scale 1.5 > logs/diag_bat_speed_f0f_h15.txt 2>&1
O=$(grep -A28 "^== diag_bat_speed" logs/diag_bat_speed_f0f_h15.txt)
if [ -n "$O" ]; then echo "--- (a) open loop F0f"; echo "$O" | grep -v "INFO"; RES="(a) ok"; else echo "(a) failed:"; tail -5 logs/diag_bat_speed_f0f_h15.txt; RES="(a) failed"; fi

[ "$(cnt)" -lt 1 ] || { echo "Isaac busy before (b): stop"; exit 0; }
AIB_EFFORT_SCALE=1.5 timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C0 --envs 1024 --seed 0 --tag bspeed4_12500 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log --body_log > logs/bspeed4_12500_contact.txt 2>&1
B=$(grep "\[bat\] body_log\|\[bat\] contact_log\|slow in zone" logs/bspeed4_12500_contact.txt)
if [ -n "$B" ]; then echo "--- (b) closed loop 12500"; echo "$B"; RES="$RES | (b) ok"; else echo "(b) failed:"; tail -5 logs/bspeed4_12500_contact.txt; RES="$RES | (b) failed"; fi
date +"%H:%M bspeed4 done, isaac after: $(cnt)"
