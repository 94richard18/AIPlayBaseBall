#!/bin/bash
# G4 + ES (planning 14, 11-01 00:1x). Starts now (idle). Plain 12500, 4 joint clips, real heights, phase 2.6, j0, P3c releases.
#  G4_END: the compliant grip at 400 Hz - G4_k20000 (K 2e4 / C 200), then G4_k80000 (K 8e4 / C 400); --grip_log --chain_log.
#    ss ~21 with a ~50 mm gap -> the 400 Hz extra is not loop error-correction energy; ss toward 16-17 -> 400 Hz was inflated.
#  ES_END: zero-shot effort probes, hard grip (measurement only, not a spec change; AIB_EFFORT_SCALE overrides the 1.5 spec
#    for these diag processes only): ES8_e20 800 Hz x2.0, ES8_e30 800 Hz x3.0, ES4_e20 400 Hz x2.0 (control); --sat_log.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5 PYTHONPATH=.
unset AIB_FIX_STD AIB_STD_FROM AIB_SWING_STOCHASTIC AIB_PHYSICS_DT
R=assets/game/p3c_identical_t18_release.csv
H0=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
C4="obp_167-7 obp_282-1 obp_42-1 obp_42-2"
SC=C:/Users/User/AppData/Local/Temp/claude/D--MyProject-AIPlayBaseBall/05cd3340-2202-4b0c-80bc-c708b2ddb810/scratchpad
O=logs/g4es; mkdir -p $O
rm -f logs/G4_END.txt logs/ES_END.txt
GMSG="aborted"; GOUT=""; EMSG="aborted (G4 did not finish)"; EOUT=""
trap 'echo "G4_END $(date +%H:%M) $GMSG$GOUT" > logs/G4_END.txt; cat logs/G4_END.txt; echo "ES_END $(date +%H:%M) $EMSG$EOUT" > logs/ES_END.txt; cat logs/ES_END.txt' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
row() {  # $1 tag, $2 row helper, rest: extra flags -> prints the row
  local TG=$1 HP=$2; shift 2
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $H0 --envs 1024 --seed 0 --tag $TG --timing_noise 0 \
    --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --phase_horizon 2.6 --real_height --clips $C4 --contact_log \
    "$@" > $O/diag_$TG.txt 2>&1
  if grep -q Traceback $O/diag_$TG.txt || [ ! -f logs/diag_bat_contact_$TG.json ]; then
    echo "$TG: crash or no json ($(grep -A14 Traceback $O/diag_$TG.txt | tail -1))"; return
  fi
  echo "$TG: $($P $SC/$HP logs/diag_bat_contact_$TG.json)"
}
GMSG="400 Hz, compliant top-hand grip, plain 12500 (vs CH_W4 hard 400 Hz ss 21.3 gap 5.9 mm / CH_W8 hard 800 Hz ss 16.6)"
GOUT="$GOUT | $(row G4_k20000 grip_row.py --grip_soft 20000 200 --grip_log --chain_log)"
GOUT="$GOUT | $(row G4_k80000 grip_row.py --grip_soft 80000 400 --grip_log --chain_log)"
GMSG="done; $GMSG"
EMSG="zero-shot effort probes, hard grip, plain 12500 (vs P4 400 Hz x1.5: ss 21.3, exit 17.9, launch 24.2, hit 0.483)"
EOUT="$EOUT | $(AIB_EFFORT_SCALE=2.0 row ES8_e20 dtnum_row.py --physics_dt 0.00125 --sat_log)"
EOUT="$EOUT | $(AIB_EFFORT_SCALE=3.0 row ES8_e30 dtnum_row.py --physics_dt 0.00125 --sat_log)"
EOUT="$EOUT | $(AIB_EFFORT_SCALE=2.0 row ES4_e20 dtnum_row.py --sat_log)"
EMSG="done; $EMSG"
