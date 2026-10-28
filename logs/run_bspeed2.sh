#!/bin/bash
# Bat-speed third-cap check (decision session 10-29): diag_bat_speed with the lower-body lines (pelvis yaw rate, ankle /
# knee / hip_yaw speed peaks, foot horizontal speed vs the reference) for F1 (strong, gain x3) and F0 (human 1.5x).
# Open loop, 16 envs, no training. Started by hand on 14's "go"; checks Isaac 0 twice 60 s apart first.
# End marker: the file logs/BSPEED2_END.txt (one line "BSPEED2_END HH:MM ..."), also on any failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED2_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED2_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
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
    RES="$RES | $t pelvis yaw ratio $Y"
}
run f1b_strong_g3 --strength strong --gain_scale 3
run f0b_h15 --strength human --effort_scale 1.5
date +"%H:%M bspeed2 done, isaac after: $(cnt)"
