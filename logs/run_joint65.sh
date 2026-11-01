#!/bin/bash
# 14 (11-01 18:1x): joint rows move to --seconds 6.5 (20's cut test: at 6.0 s 5/64 envs are still in play). Rerun the 14496 +
# R4 1895 joint s6 at 6.5 s after FILMCAM_END (Isaac <= 1 = beside the seg-5 trainer, RAM >= 20 GB, no FILM* / PDT / CUTTEST GO
# without a newer END): the 6.0 s csv / txt -> *_60, new csv in place, logs/h31/joint_s6_14496.s65 tag (R5c's verify waits for
# it), H31_JOINT_END refreshed (same format + "6.5 s") -> logs/JOINT65_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/h31; I=14496; JC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-01_11-42-57_human_pitched_h31_4/model_$I.pt
RC=logs/rsl_rl/aib1_recover/2026-10-31_18-56-40_h25r4/model_1895.pt
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
JX="--saim logs/armaim_auth/scripted_gain.json --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
M=logs/JOINT65_END.txt; rm -f $M; MSG="aborted"
trap 'echo "JOINT65_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt logs/CUTTEST_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
MSG="waiting for FILMCAM_END"
until [ -f logs/FILMCAM_END.txt ] && [ logs/FILMCAM_END.txt -nt logs/FILMCAM_GO.txt ]; do sleep 30; done
until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
[ -f $O/joint_s6_${I}_60.csv ] || { cp $O/joint_s6_$I.csv $O/joint_s6_${I}_60.csv; cp $O/joint_s6_$I.txt $O/joint_s6_${I}_60.txt; }
MSG="joint"
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $JC --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --phase_horizon 2.6 --seconds 6.5 \
  --csv $O/joint_s6_${I}_65tmp.csv > $O/joint_s6_${I}_65.txt 2>&1
grep -q Traceback $O/joint_s6_${I}_65.txt || [ ! -f $O/joint_s6_${I}_65tmp.csv ] && { MSG="CRASH: $(grep -A14 Traceback $O/joint_s6_${I}_65.txt | tail -1)"; exit 1; }
mv $O/joint_s6_${I}_65tmp.csv $O/joint_s6_$I.csv; cp $O/joint_s6_${I}_65.txt $O/joint_s6_$I.txt; touch $O/joint_s6_$I.s65
R=$($P $SC/h31_joint_row.py $O/joint_s6_$I.csv)
echo "H31_JOINT_END $(date +%H:%M) BC=$JC ROWS=$O/joint_s6_$I.csv (recover 2026-10-31_18-56-40_h25r4, 6.5 s; the 6.0 s row is joint_s6_${I}_60.csv): $R" > logs/H31_JOINT_END.txt
MSG="6.5 s: $R | $($P logs/film_pick.py $O/joint_s6_$I.csv $O/film_pick_65_scratch.txt | head -1) | 6.0 s: $($P logs/film_pick.py $O/joint_s6_${I}_60.csv $O/film_pick_60_scratch.txt | head -1)"
