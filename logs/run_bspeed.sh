#!/bin/bash
# Bat-speed cap fork (decision session 10-29 00:0x): open-loop reference swing (zero residual), 16 envs, no training.
# F0 human 1.5x (today's USD baseline) / F1 strong gain x3 / F2 human 1.5x gain x3 / F3 human 2.0x gain x3.
# F1 >= 28 m/s sweet spot -> PD bandwidth is the non-torque cap; ~20 -> look elsewhere.
# Gate: 20's p5 re-film end (grep -q M3PA_END logs/m3pa_chain.txt) + Isaac 0 twice 60 s apart.
# End marker: the file logs/BSPEED_END.txt (one line "BSPEED_END HH:MM ..."), also on any failure.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
M=logs/BSPEED_END.txt
rm -f $M
RES=""
trap 'echo "BSPEED_END $(date +%H:%M) ${RES:-no result}" > $M; cat $M' EXIT
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
date +"%H:%M bspeed: waiting for M3PA_END in logs/m3pa_chain.txt"
until grep -q M3PA_END logs/m3pa_chain.txt 2>/dev/null; do sleep 30; done
until [ "$(cnt)" -lt 1 ] && sleep 60 && [ "$(cnt)" -lt 1 ]; do sleep 20; done
date +"%H:%M gate open"
run() {  # $1 tag, rest = args
    t=$1; shift
    [ "$(cnt)" -lt 1 ] || { echo "Isaac busy before $t: stop"; exit 0; }
    timeout 600 $P -u scripts/diag_bat_speed.py --headless --envs 16 --tail 1.0 "$@" > logs/diag_bat_speed_$t.txt 2>&1
    O=$(grep -A20 "^== diag_bat_speed" logs/diag_bat_speed_$t.txt)
    [ -n "$O" ] || { echo "$t failed:"; tail -5 logs/diag_bat_speed_$t.txt; RES="$RES | $t failed"; return; }
    echo "--- $t ($*)"; echo "$O"
    SS=$(echo "$O" | grep -o "sweet spot peak before contact [0-9.]*" | grep -o "[0-9.]*$")
    RES="$RES | $t ss $SS"
}
run f0_h15 --strength human --effort_scale 1.5
run f1_strong_g3 --strength strong --gain_scale 3
run f2_h15_g3 --strength human --effort_scale 1.5 --gain_scale 3
run f3_h20_g3 --strength human --effort_scale 2.0 --gain_scale 3
date +"%H:%M bspeed done, isaac after: $(cnt)"
