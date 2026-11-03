#!/bin/bash
# run_ema.sh (14, 11-03 22:xx): eval only, after PIN_CMP_END. (1) obs-noise-off diag of 15993 (--percept_clean: ball percept noise /
# bias off, delay kept) -> NOISE_OFF_END; (2) AIB_ACT_EMA 0.3 / 0.5 (pre-commit residual low-pass): diag j0 1024 envs + joint s6
# (R6b, seed 6) each -> EMA_03_END / EMA_05_END (logs/ema_gate.py vs form_15993 + joint_s6_15993_h32.csv); (3) the first passing
# alpha (0.3 first) gets the game-scene tables (both columns) -> H33_TABLES_15993_ema<alpha>_END. Isaac 0 + RAM + nofilm per launch.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.0 AIB_ARMAIM_NOBOOK=1 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT AIB_EFFORT_PROBE AIB_TIMING_NOISE AIB_TEMPO_SCRIPT AIB_ACT_EMA AIB_PRESTART_PIN
O=logs/h33; R=assets/game/p3c_identical_t18_release.csv; C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=logs/rsl_rl/aib1_recover/2026-11-02_02-41-30_h25r6b/model_3091.pt
BC=logs/rsl_rl/aib1_tee_batting_mimic/2026-11-02_04-02-58_human_pitched_h32_3/model_15993.pt
M=logs/EMA_CHAIN_END.txt; rm -f $M logs/NOISE_OFF_END.txt logs/EMA_03_END.txt logs/EMA_05_END.txt; MSG="aborted"
trap 'echo "EMA_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
nofilm() { local g e; for g in logs/FILM*_GO.txt logs/PDT_GO.txt; do [ -f $g ] || continue; e=${g%_GO.txt}_END.txt; [ -f $e ] && [ $e -nt $g ] || return 1; done; return 0; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ] && nofilm; do sleep 30; done; }
diag() {  # $1 tag, extra flags...
  local T=$1; shift
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $BC --percept_filter --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 \
    --physics_dt 0.00125 --contact_log --form_log "$@" > $O/diag_$T.txt 2>&1
  ! grep -q Traceback $O/diag_$T.txt && [ -f logs/diag_bat_contact_$T.json ]
}
MSG="waiting for PIN_CMP_END"; until [ -f logs/PIN_CMP_END.txt ] && [ logs/PIN_CMP_END.txt -nt logs/run_ema.sh ]; do sleep 60; done
MSG="noise-off diag"; idle
diag form_15993_clean --percept_clean || { MSG="noise-off diag crash"; exit 1; }
echo "NOISE_OFF_END $(date +%H:%M) 15993 ball percept noise / bias off: $($P logs/ema_gate.py form_15993 form_15993_clean $O/joint_s6_15993_h32.csv $O/joint_s6_15993_h32.csv | cut -d'|' -f2-3)" > logs/NOISE_OFF_END.txt
PASSA=""
for AL in 0.3 0.5; do
  K=${AL/./}; MSG="EMA $AL diag"; idle
  AIB_ACT_EMA=$AL diag form_15993_ema$AL || { MSG="EMA $AL diag crash"; exit 1; }
  MSG="EMA $AL joint s6"; idle
  AIB_ACT_EMA=$AL timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $BC --envs 64 --seed 6 --after_done recover_rl \
    --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 --percept_filter --phase_horizon 2.6 --seconds 6.5 \
    --csv $O/joint_s6_15993_ema$AL.csv > $O/joint_s6_15993_ema$AL.txt 2>&1
  [ -f $O/joint_s6_15993_ema$AL.csv ] && ! grep -q Traceback $O/joint_s6_15993_ema$AL.txt || { MSG="EMA $AL joint crash"; exit 1; }
  G=$($P logs/ema_gate.py form_15993 form_15993_ema$AL $O/joint_s6_15993_h32.csv $O/joint_s6_15993_ema$AL.csv)
  echo "EMA_${K}_END $(date +%H:%M) AIB_ACT_EMA=$AL: $G" > logs/EMA_${K}_END.txt
  [ -z "$PASSA" ] && [ "${G%% *}" = PASS ] && PASSA=$AL
done
[ -n "$PASSA" ] || { MSG="no alpha passed the gate (EMA_03_END / EMA_05_END)"; exit 0; }
MSG="tables ema $PASSA"; idle
AIB_ACT_EMA=$PASSA bash logs/run_h33_tables.sh $BC 15993_ema$PASSA form_15993_ema$PASSA > logs/h33_tables_15993_ema$PASSA.out 2>&1
MSG="alpha $PASSA tables: $(cut -c1-300 logs/H33_TABLES_15993_ema${PASSA}_END.txt) | 15993 ref 23.25 / 27.69"
