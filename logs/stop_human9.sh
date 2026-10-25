#!/bin/bash
# yield to the pitcher (decision session 10-25): stop the Human9 trainer once model_12300 is written
cd /d/MyProject/AIPlayBaseBall
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched9 | head -1)
until [ -f $D/model_12300.pt ]; do sleep 10; done
sleep 20
powershell -NoProfile -Command "Stop-Process -Id 71532 -Force"
echo "saved: $(ls $D | grep model | sort -t_ -k2 -n | tail -1)"
# training stats at the iteration-12295 check and at the stop
awk '/Learning iteration 12295\//{f=1} f&&/bat\/(hit_rate|success_rate|fair_rate|carry:)/{print "it12295 " $0} /Learning iteration 12296\//{exit}' logs/train_human9.txt | tr -s ' '
grep -o "Learning iteration [0-9/]*" logs/train_human9.txt | tail -1
grep -E "bat/(hit_rate|success_rate|fair_rate|carry:)" logs/train_human9.txt | tail -4 | tr -s ' '
date +"%H:%M trainer stopped"
