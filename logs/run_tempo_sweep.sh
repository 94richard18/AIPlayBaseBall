#!/bin/bash
# run_tempo_sweep.sh (14, 11-03 14:2x): is the reference the lever? Diags of 12500 (h9b) and 13498 (h33g_1) with the reference
# played x tempo_script 1.0 / 1.15 / 1.30 (re-timed to contact), 256 envs, jitter 0, 11-02 protocol, --form_log --sat_log.
# Pose / key are measured against the reference AT that tempo_script (same ref_t). No training. Runs after H33JT_CHAIN_END (run_h33j_t.sh)
# (GPU gap), idle() + nofilm before each diag. Rows -> logs/h33/tempo_sweep.txt, marker logs/TEMPO_SWEEP_END.txt.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE
R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"; B=logs/rsl_rl/aib1_tee_batting_mimic
O=logs/h33/tempo_sweep.txt; M=logs/TEMPO_SWEEP_END.txt; rm -f $M; MSG="aborted"; ROWS=""
trap 'echo "TEMPO_SWEEP_END $(date +%H:%M) $MSG |$ROWS" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
MSG="waiting for H33JT_CHAIN_END"; until [ -f logs/H33JT_CHAIN_END.txt ] && [ logs/H33JT_CHAIN_END.txt -nt logs/run_tempo_sweep.sh ]; do sleep 60; done
for ck in "12500 $B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt" "13498 $B/2026-11-03_06-50-43_human_pitched_h33g_1/model_13498.pt"; do
  set -- $ck
  for ts in 1.0 1.15 1.30; do
    T=tempo${ts}_$1; MSG="diag $T"; idle
    timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $2 --percept_filter --envs 256 --seed 0 --tag $T \
      --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
      --physics_dt 0.00125 --contact_log --form_log --sat_log --tempo_script $ts > logs/h33/diag_$T.txt 2>&1
    if grep -q Traceback logs/h33/diag_$T.txt || [ ! -f logs/diag_bat_contact_$T.json ]; then ROW="$1 x$ts: CRASH $(grep -A14 Traceback logs/h33/diag_$T.txt | tail -1)"
    else ROW="$1 x$ts: $($P logs/tempo_row.py $T 2>/dev/null | tail -1) | da $($P logs/e10_row.py logs/diag_bat_contact_$T.json | sed -n 's/.* da \([0-9.]* \/ q90 [0-9.]*\).*/\1/p')"; fi
    echo "$(date +%H:%M) $ROW" >> $O; ROWS="$ROWS $ROW ||"
  done
done
MSG="done"
