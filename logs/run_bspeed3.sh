#!/bin/bash
# Foot-friction check (decision session 10-29, diag only): open loop, 16 envs, no training.
# F0c human 1.5x / F1c strong gain x3 with foot-ground friction 2.0 / 2.0 (scene default material, combine max);
# F0d human 1.5x with --friction 1.0 0.9 explicitly (today's values: the robot has no material of its own, the scene
# default is 1.0 / 0.9) -> must reproduce F0b; it checks the option itself.
# Gate: 20's sky re-film end (grep -q M3SKY_ALL_END logs/m3sky_all_chain.txt) + Isaac 0 twice 60 s apart.
# End marker: the file logs/BSPEED3_END.txt (one line "BSPEED3_END HH:MM ..."), also on any failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED3_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED3_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until grep -q M3SKY_ALL_END logs/m3sky_all_chain.txt 2>/dev/null; do sleep 30; done
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open (Isaac 0 twice 60 s apart)"
run() {  # $1 tag, rest = args
    t=$1; shift
    [ "$(cnt)" -lt 1 ] || { echo "Isaac busy before $t: stop"; exit 0; }
    timeout 600 $P -u scripts/diag_bat_speed.py --headless --envs 16 --tail 1.0 "$@" > logs/diag_bat_speed_$t.txt 2>&1
    O=$(grep -A24 "^== diag_bat_speed" logs/diag_bat_speed_$t.txt)
    [ -n "$O" ] || { echo "$t failed:"; tail -5 logs/diag_bat_speed_$t.txt; RES="$RES | $t failed"; return; }
    echo "--- $t ($*)"; echo "$O"
    Y=$(echo "$O" | grep -o "(ratio [0-9.]*)" | grep -o "[0-9.]*")
    SS=$(echo "$O" | grep -o "sweet spot peak before contact [0-9.]*" | grep -o "[0-9.]*$")
    RES="$RES | $t pelvis yaw ratio $Y ss $SS"
}
run f0c_h15_mu2 --strength human --effort_scale 1.5 --friction 2.0 2.0
run f1c_strong_g3_mu2 --strength strong --gain_scale 3 --friction 2.0 2.0
run f0d_h15_mu1 --strength human --effort_scale 1.5 --friction 1.0 0.9
date +"%H:%M bspeed3 done, isaac after: $(cnt)"
