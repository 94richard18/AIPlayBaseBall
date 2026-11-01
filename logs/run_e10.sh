#!/bin/bash
# E10 (planning 14, 11-01 20:0x; user: batter strength back to human level + no axis-alignment signal): zero-shot rows at
# AIB_EFFORT_SCALE 1.0, 800 Hz, phase 2.6, saim OFF everywhere. Isaac 0 for each step.
#  (a) diag in-zone 4 clips (as the H31 rows) + jitter (14, 20:2x): 14496 saim ON (reference: the before), 14496 + percept_filter, 14496 raw (no filter), plain 12500 -> logs/E10_DIAG_END.txt
#  (b) joint s6 14496 + percept_filter + R5c recover (1.5x-trained), 6.5 s, --effort_probe 1.0 -> logs/E10_JOINT_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EFFORT_PROBE
B=logs/rsl_rl/aib1_tee_batting_mimic
C14=$B/2026-11-01_11-42-57_human_pitched_h31_4/model_14496.pt
C12=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
O=logs/e10; mkdir -p $O
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
M=logs/E10_DIAG_END.txt; rm -f $M logs/E10_JOINT_END.txt; MSG="aborted"
trap 'echo "E10_DIAG_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
diag() {  # $1 ckpt, $2 tag, $3 extra flags
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 $3 --envs 1024 --seed 0 --tag $2 \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log > $O/diag_$2.txt 2>&1
  [ -f logs/diag_bat_contact_$2.json ] && ! grep -q Traceback $O/diag_$2.txt || { MSG="$MSG | $2 CRASH $(grep -A14 Traceback $O/diag_$2.txt | tail -1)"; return; }
  MSG="$MSG | $2: $($P logs/e10_row.py logs/diag_bat_contact_$2.json); all-pitch fair pop-up share / q75 / n $($P logs/popup_share.py logs/diag_bat_contact_$2.json)"
}
MSG="1.0x, 800 Hz, PH 2.6, 4 clips"
GY=logs/armaim_auth/scripted_gain.json; RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
diag $C14 e10_14496_saimON "--armaim $RES --aim_scripted $GY --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter"
diag $C14 e10_14496_pf --percept_filter
diag $C14 e10_14496_raw ""
diag $C12 e10_12500 ""
trap - EXIT; echo "E10_DIAG_END $(date +%H:%M) $MSG" > $M; cat $M
M=logs/E10_JOINT_END.txt; MSG="aborted"
trap 'echo "E10_JOINT_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
RC=$(tr -d '\r\n' < logs/recov5c/recover_ckpt.txt)
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $C14 --envs 64 --seed 6 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
  --effort_probe 1.0 --csv $O/joint_s6_14496_e10.csv > $O/joint_s6_14496_e10.txt 2>&1
[ -f $O/joint_s6_14496_e10.csv ] && ! grep -q Traceback $O/joint_s6_14496_e10.txt || { MSG="CRASH $(grep -A14 Traceback $O/joint_s6_14496_e10.txt | tail -1)"; exit 1; }
MSG="14496 + percept_filter, saim off, R5c $(basename $RC), 1.0x ($(grep -c 'EFFORT PROBE: batter effort scale 1.0' $O/joint_s6_14496_e10.txt) probe line), 6.5 s: $($P logs/film_pick.py $O/joint_s6_14496_e10.csv $O/film_pick_e10.txt | head -1) | R5c @1.5x ref: ground 1, fell 6 (before done 0)"
