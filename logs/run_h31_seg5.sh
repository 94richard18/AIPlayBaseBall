#!/bin/bash
# 14 (11-01 11:4x): segment 4 already runs with ramp 16-28 / hinge cap 14, so the new bounds start at segment 5: logs/h31/STOP
# (content "relaunch-seg5") ends the chain after the segment-4 eval; then, if STOP still has that content (14 did not turn it
# into a real stop) and H31_2000_END passes 14's continue rule (exit >= 25 and hit >= 0.50), relaunch run_h31.sh at K0=5 from the
# newest segment-4 checkpoint with ramp 18-32 and exit-hinge cap 20 (w 40 / w 3 / ref 16 / win 0.01 unchanged).
cd /d/MyProject/AIPlayBaseBall
O=logs/h31; M=logs/H31_SEG5_RELAUNCH.txt; rm -f $M; MSG="aborted"
trap 'echo "H31_SEG5_RELAUNCH $(date +%H:%M) $MSG" > $M; cat $M' EXIT
echo relaunch-seg5 > $O/STOP
until [ -f logs/H31_2000_END.txt ] && [ -f logs/H31_END.txt ] && grep -q "before segment 5" logs/H31_END.txt; do
  [ -f logs/H31_END.txt ] && ! grep -q "before segment 5" logs/H31_END.txt && { MSG="chain ended otherwise: $(cut -c1-200 logs/H31_END.txt)"; exit 1; }
  sleep 60
done
[ "$(tr -d '\r\n' < $O/STOP)" = "relaunch-seg5" ] || { MSG="STOP changed by 14, not relaunched"; exit 0; }
EX=$(sed -n 's/.*(800 Hz, 4 clips): ss [0-9.]* exit \([0-9.]*\) .*/\1/p' logs/H31_2000_END.txt)
HT=$(sed -n 's/.*(800 Hz, 4 clips): ss [0-9.]* exit [0-9.]* launch [-0-9.]* hit \([0-9.]*\) .*/\1/p' logs/H31_2000_END.txt)
awk -v e="$EX" -v h="$HT" 'BEGIN{exit !(e ~ /^[0-9.]+$/ && h ~ /^[0-9.]+$/ && e + 0 >= 25 && h + 0 >= 0.50)}' \
  || { MSG="STOP rule (exit '$EX' hit '$HT'): chain left stopped"; exit 0; }
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched_h31_4 | head -1)
L=$(ls $D | grep model | sort -t_ -k2 -n | tail -1)
rm -f $O/STOP
MSG="relaunched segment 5 from $(basename $D)/$L (exit $EX hit $HT), ramp 18-32, hinge cap 20 -> logs/h31_chain5.txt"
trap - EXIT; echo "H31_SEG5_RELAUNCH $(date +%H:%M) $MSG" > $M
K0=5 LR0=$(basename $D) LC0=$L AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=18.0 AIB_SS_CA_HI=32.0 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=16.0 \
  AIB_EXIT_HINGE_CAP=20.0 exec bash logs/run_h31.sh > logs/h31_chain5.txt 2>&1
