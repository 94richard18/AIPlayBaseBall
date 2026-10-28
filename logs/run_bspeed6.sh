#!/bin/bash
# Swing-start bias sweep on the delivery 12500 (decision session 10-29, zero training): is the early waist peak just the
# engine starting the swing 0.055 s early? diag_bat_contact --contact_log --chain_log --phase_log --link_log at
# timing_bias 0.055 / 0.030 / 0.000 / -0.020 (jitter 0, P3c t18, aim 0, 1024 envs, AIB_EFFORT_SCALE 1.5).
# Gate: Isaac 0 twice 60 s apart. End marker: the file logs/BSPEED6_END.txt ("BSPEED6_END HH:MM ..."), also on failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED6_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED6_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open (Isaac 0 twice 60 s apart)"
C0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv

st() { $P -c "
import json, math; S = json.load(open('logs/diag_bat_contact_bspeed6_$1.json'))['summary']; h = S['hits']; c = S['contact_log']
L = S['link_log']['rows']; Q = S['chain_log']['qd_vs_ref']['waist_yaw']
print(f\"bias $1: in-zone {S['by_speed']['slow in zone']['hit_rate']} | hits {h['n']} | EV med {c['ev'][1]} | ss contact {c['ss_contact'][1]} | \"
      f\"waist_yaw qd at contact {Q['qd']} (ref {Q['qd_ref']}), peak t {L['waist_yaw']['t_pk_sim']} (ref {L['waist_yaw']['t_pk_ref']}) | \"
      f\"l_shoulder_pitch peak t {L['l_shoulder_pitch']['t_pk_sim']} | l_wrist_pitch peak t {L['l_wrist_pitch']['t_pk_sim']} | \"
      f\"falls after hit {round(100 * h['fell'] / max(h['n'], 1))}%\")"; }
for b in 0.055 0.030 0.000 -0.020; do
    [ "$(cnt)" -lt 1 ] || { echo "Isaac busy before bias $b: stop"; exit 0; }
    AIB_EFFORT_SCALE=1.5 timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $C0 --envs 1024 --seed 0 --tag bspeed6_$b --timing_noise 0 --timing_bias $b --aim_offset_z 0 --release_csv $R --contact_log --chain_log --phase_log --link_log > logs/bspeed6_$b.txt 2>&1
    O=$(st $b 2>&1)
    echo "$(date +%H:%M) $O"; RES="$RES | ${O:0:60}"
done
date +"%H:%M bspeed6 done, isaac after: $(cnt)"
