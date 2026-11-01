#!/bin/bash
# 14 (11-01 16:3x): seg 6 = the follow-through option IF seg 5's joint row still has >= 10/64 falls before done, else the carry
# line goes on unchanged. Once segment 5 trains (its run dir exists, so the chain is past its STOP checks for k=5), STOP
# "relaunch-seg6" ends the chain after the segment-5 eval; then, if STOP still has that content and H31_2500_END passes the
# continue rule (exit >= 24 = 14's STOP bar, hit >= 0.50): joint s6 (64 envs, saim + R4 recover, PH 2.6) of the segment-5
# checkpoint -> falls before done; >= 10 -> AIB_FOLLOW_THROUGH_S=2.0 AIB_EPISODE_S=4.0 AIB_SETTLE_TAIL_S=1.5 (cfg: ref_hold_end
# + lost-after-hit-is-fall, phase 2.6) after a 64-env 20-it smoke of that variant (crash -> the carry line instead); relaunch
# run_h31c.sh (= run_h31b.sh, joints at 6.5 s) at K0=6 with the segment-5 bounds; boundary joint at --seconds 6.5 (14, 18:1x) -> logs/H31_SEG6_RELAUNCH.txt, logs/h31_chain6.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
export AIB_H29_SIGMA_AX=0.20 AIB_H29_SIGMA_GAP=0.10 AIB_SAIM_BOUND=0.4
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S
SEG5="AIB_SS_CA_WIN=0.01 AIB_SS_CA_LO=18.0 AIB_SS_CA_HI=32.0 AIB_EXIT_HINGE_W=3.0 AIB_EXIT_HINGE_REF=16.0 AIB_EXIT_HINGE_CAP=20.0 AIB_CARRY_REF=0.0 AIB_CARRY_SPAN=30.0"
FT="AIB_FOLLOW_THROUGH_S=2.0 AIB_EPISODE_S=4.0 AIB_SETTLE_TAIL_S=1.5"
LHI="AIB_LAUNCH_HI_W=10.0 AIB_LAUNCH_HI_REF=40.0 AIB_LAUNCH_HI_SPAN=15.0"
B=logs/rsl_rl/aib1_tee_batting_mimic
O=logs/h31; M=logs/H31_SEG6_RELAUNCH.txt; rm -f $M; MSG="aborted"
trap 'echo "H31_SEG6_RELAUNCH $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
MSG="waiting for segment 5 to start"
until ls -d $B/*human_pitched_h31_5 > /dev/null 2>&1; do [ -f logs/H31_END.txt ] && { MSG="chain ended before segment 5: $(cut -c1-200 logs/H31_END.txt)"; exit 1; }; sleep 60; done
echo relaunch-seg6 > $O/STOP
MSG="waiting for H31_2500_END"
until [ -f logs/H31_2500_END.txt ] && [ -f logs/H31_END.txt ] && grep -q "before segment 6" logs/H31_END.txt; do
  [ -f logs/H31_END.txt ] && ! grep -q "before segment 6" logs/H31_END.txt && { MSG="chain ended otherwise: $(cut -c1-200 logs/H31_END.txt)"; exit 1; }
  sleep 60
done
[ "$(tr -d '\r\n' < $O/STOP)" = "relaunch-seg6" ] || { MSG="STOP changed by 14, not relaunched"; exit 0; }
EX=$(sed -n 's/.*(800 Hz, 4 clips): ss [0-9.]* exit \([0-9.]*\) .*/\1/p' logs/H31_2500_END.txt)
HT=$(sed -n 's/.*(800 Hz, 4 clips): ss [0-9.]* exit [0-9.]* launch [-0-9.]* hit \([0-9.]*\) .*/\1/p' logs/H31_2500_END.txt)
awk -v e="$EX" -v h="$HT" 'BEGIN{exit !(e ~ /^[0-9.]+$/ && h ~ /^[0-9.]+$/ && e + 0 >= 24 && h + 0 >= 0.50)}' \
  || { MSG="STOP rule (exit '$EX' hit '$HT'): chain left stopped"; exit 0; }
D=$(ls -td $B/*human_pitched_h31_5 | head -1); L=$(ls $D | grep model | sort -t_ -k2 -n | tail -1); I=${L#model_}; I=${I%.pt}
MSG="joint s6 of $(basename $D)/$L"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov4/recover_ckpt.txt)
JX="--saim logs/armaim_auth/scripted_gain.json --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $D/$L --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --phase_horizon 2.6 --seconds 6.5 --csv $O/joint_s6_$I.csv \
  > $O/joint_s6_$I.txt 2>&1
[ -f $O/joint_s6_$I.csv ] && ! grep -q Traceback $O/joint_s6_$I.txt || { MSG="joint s6 $I crashed: $(grep -A14 Traceback $O/joint_s6_$I.txt | tail -1)"; exit 1; }
touch $O/joint_s6_$I.s65
JR=$($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_scratch.txt | head -1)
BD=$($P -c "import csv; print(sum(r['batter_fell_before_done'] == 'True' for r in csv.DictReader(open('$O/joint_s6_$I.csv'))))")
# 14 (18:5x): pop-up rule first - diag fair-hit share >= 45 deg above 14496's 0.193 or fair launch q75 > 50 -> the launch ceiling
# (AIB_LAUNCH_HI_W 10, ref 40, span 15) instead; otherwise the follow-through / carry rule on the joint falls before done
read PS PQ PN <<< "$($P logs/popup_share.py logs/diag_bat_contact_h31_$I.json 2>/dev/null)"
POP=$(awk -v s="$PS" -v q="$PQ" 'BEGIN{print (s ~ /^[0-9.]+$/ && (s + 0 > 0.193 || q + 0 > 50)) ? 1 : 0}')
PNOTE="diag fair pop-ups >= 45 deg $PS (14496 0.193), fair launch q75 $PQ (n $PN)"
smoke() {  # $1 variant env, $2 tag -> 0 if a 64-env 20-it smoke of the variant runs clean
  MSG="$2 smoke"; idle
  env $SEG5 $1 AIB_PHYSICS_DT=0.00125 timeout 1500 $P -u scripts/train.py --task AIB-PitchedBatting-Human31-v0 --headless --num_envs 64     --max_iterations 20 --run_name h31_${2}_smoke --resume True --load_run $(basename $D) --checkpoint $L > $O/${2}_smoke.txt 2>&1
  ! grep -q Traceback $O/${2}_smoke.txt && grep -q "AIB_PHYSICS_DT: physics dt 0.001250" $O/${2}_smoke.txt
}
V=""
if [ "$POP" = 1 ]; then
  if smoke "$LHI" lhi; then V="$LHI"; NOTE="$PNOTE -> launch ceiling ($LHI; smoke $(grep -o 'Iteration time: [0-9.]*s' $O/lhi_smoke.txt | tail -1))"
  else NOTE="$PNOTE -> launch ceiling, but its smoke failed ($(grep -A14 Traceback $O/lhi_smoke.txt | tail -1)) -> carry line"; fi
elif [ "$BD" -ge 10 ]; then
  if smoke "$FT" ft; then V="$FT"; NOTE="$PNOTE; falls before done $BD >= 10 -> follow-through ($FT; smoke $(grep -o 'Iteration time: [0-9.]*s' $O/ft_smoke.txt | tail -1))"
  else NOTE="$PNOTE; falls before done $BD >= 10 but the follow-through smoke failed ($(grep -A14 Traceback $O/ft_smoke.txt | tail -1)) -> carry line"; fi
else NOTE="$PNOTE; falls before done $BD < 10 -> carry line"; fi
rm -f $O/STOP
MSG="relaunched segment 6 from $(basename $D)/$L (exit $EX hit $HT); joint s6: $JR; $NOTE -> run_h31c.sh, logs/h31_chain6.txt"
trap - EXIT; echo "H31_SEG6_RELAUNCH $(date +%H:%M) $MSG" > $M
exec env K0=6 LR0=$(basename $D) LC0=$L $SEG5 $V bash logs/run_h31c.sh > logs/h31_chain6.txt 2>&1
