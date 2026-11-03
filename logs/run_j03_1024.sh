#!/bin/bash
# run_j03_1024.sh (14, 11-03 14:5x): sign-flip falsifier. The 256-env jitter-0.03 diag of 13498 raised launch (+26.5 all-fair,
# pop-ups 0.33) while the j0.03 tables lower it (+10.6). 1024-env j0.03 diag of 13498, after TEMPO_SWEEP_END, lowest priority:
# flip gone = sample; flip stays = fair mask (landing vs launch spray) next, the diag's jitter rows not comparable to the tables.
# Marker logs/J03_1024_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
M=logs/J03_1024_END.txt; rm -f $M; MSG="aborted"
trap 'echo "J03_1024_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
binl() { $P logs/h33_bins.py $1 2>/dev/null | grep "all fair" | head -1 | sed -n 's/.*launch med *\([-+0-9.]*\) .*/\1/p'; }
MSG="waiting for TEMPO_SWEEP_END"; until [ -f logs/TEMPO_SWEEP_END.txt ] && [ logs/TEMPO_SWEEP_END.txt -nt logs/run_j03_1024.sh ]; do sleep 60; done
MSG="diag"; idle
T=h33j_base_13498_j03_1024
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint logs/rsl_rl/aib1_tee_batting_mimic/2026-11-03_06-50-43_human_pitched_h33g_1/model_13498.pt \
  --percept_filter --envs 1024 --seed 0 --tag $T --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 \
  --real_height --clips $C4 --physics_dt 0.00125 --contact_log --form_log > logs/h33/diag_$T.txt 2>&1
grep -q Traceback logs/h33/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ] && { MSG="diag crash or no json"; exit 1; }
MSG="13498 diag j0.03 1024 envs: bins all-fair launch $(binl $T) (256 envs: +26.5; diag j0 1024: $(binl h33g_13498); tables j0.03 +10.6 / j0 +15.9) | pop-ups $($P logs/popup_share.py logs/diag_bat_contact_$T.json) | $($P logs/tempo_row.py $T 2>/dev/null | tail -1)"
