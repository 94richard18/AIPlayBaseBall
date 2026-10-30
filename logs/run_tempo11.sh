#!/bin/bash
# Line C falsifier (planning 14, 10-31 06:01): row 9 (12500 @ 1.5x + saim bound 0.4 + full comp + filtered percept) with the
# whole swing reference played x1.1 (cfg.tempo_script, re-timed to contact), diag1024 s0 -> TEMPO_FALSIFIER_END.
# Rule: exit median up >= 1 m/s vs row 9 (18.0) with in-zone hit >= 0.40 -> line C candidate, else dead.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
O=logs/tempo11; mkdir -p $O; M=logs/TEMPO_FALSIFIER_END.txt; rm -f $M; MSG="aborted"
trap 'echo "TEMPO_FALSIFIER_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted logs/armaim_auth/scripted_gain.json \
  --aim_bound 0.4 --aim_window 0.60 --aim_obs_comp --aim_obs_comp_full --percept_filter --tempo_script 1.1 --envs 1024 --seed 0 --tag tempo11 \
  --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv assets/game/p3c_identical_t18_release.csv --contact_log > $O/diag.txt 2>&1
grep -q Traceback $O/diag.txt && { MSG="crash: $(grep -A14 Traceback $O/diag.txt | tail -2 | tr '\n' ' ')"; exit 1; }
J=logs/diag_bat_contact_tempo11.json; [ -f $J ] || { MSG="no json"; exit 1; }
G=$($P scripts/armaim_gate.py --diag $J --stage 300 --label tempo11 | head -1); C=$($P scripts/carry_cols.py --diag $J --label tempo11)
MSG="$($P - <<PY
import json, numpy as np, subprocess
P = {k: np.array(v, dtype=float) for k, v in json.load(open("$J"))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
hz = P["hit"].astype(bool) & z
Z = hz.sum() / z.sum(); EV = np.nanmedian(P["ev_any"][hz])
print(("CANDIDATE" if EV >= 19.0 and Z >= 0.40 else "DEAD") + f" line C: in-zone hit {Z:.3f}, exit median {EV:.1f} (row 9 18.0; rule >= 19.0 and hit >= 0.40)")
PY
) | $G | $C"
