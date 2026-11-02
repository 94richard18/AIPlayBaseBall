#!/bin/bash
# H32 FORM diag (planning 14, 11-02 16:4x; user approved, "跑吧"): no training. 12500 (plain h9b), 14496, 14995, 15494, 15993 at
# 1.0x, percept_filter, no saim, 800 Hz, PH 2.6, the 4 joint clips, real heights, P3c releases, seed 0, 1024 balls, --form_log:
# reference-tracking pose_err / key_err per episode over the swing (clip start -> contact) and after contact, legs / waist / arms
# split. Does "looks human" trade against "hits far" across the line? csv logs/h32/form_<ckpt>.csv; marker logs/H32_FORM_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_FOLLOW_THROUGH_S AIB_EPISODE_S AIB_EFFORT_PROBE AIB_WAIT_TORSO_W AIB_FALL_AFTER_HIT_W
O=logs/h32; B=logs/rsl_rl/aib1_tee_batting_mimic; R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
M=logs/H32_FORM_END.txt; rm -f $M; MSG="aborted"; TB=""
trap 'echo -e "H32_FORM_END $(date +%H:%M) $MSG\nckpt | hit | exit | carry | launch | pose_err mean/q90 | key_err mean/q90 | after contact | split$TB" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
S0=$(date +%s)
for X in 12500:2026-10-25_13-56-51_human_pitched9b 14496:2026-11-01_11-42-57_human_pitched_h31_4 14995:2026-11-01_20-56-20_human_pitched_h32_1 \
         15494:2026-11-02_00-24-42_human_pitched_h32_2 15993:2026-11-02_04-02-58_human_pitched_h32_3; do
  I=${X%%:*}; D=${X#*:}; T=form_$I
  MSG="FAILED: waiting for idle before $I"
  until [ "$(isaac)" -le 1 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $B/$D/model_$I.pt --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ] && { MSG="FAILED: diag $I $(grep -A14 Traceback $O/diag_$T.txt | tail -1)"; exit 1; }
  TB="$TB\n$($P logs/h32_form.py logs/diag_bat_contact_$T.json $I 2>&1 | tail -1)"
done
MSG="5 ckpts, $((($(date +%s) - S0) / 60)) min ($(grep -m1 '\[form\] joints' $O/diag_form_15993.txt))"
