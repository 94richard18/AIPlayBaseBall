#!/bin/bash
# Selection route (planning 14, 10-30; measurement for the user's decision): waits for logs/HUMAN27B_400_END.txt (the GPU),
# re-runs the 12500 diag1024 (seed 0, P3c t18, jitter 0, bias 0.055, aim 0, --contact_log) with the new *_any / spray /
# backspin keys, checks it reproduces the old dump's hits, then scripts/band_select.py -> logs/band_select/. Marker BAND_END.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
M=logs/BAND_END.txt; rm -f $M; MSG="aborted"
trap 'echo "BAND_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
W=0; until [ -f logs/HUMAN27B_400_END.txt ]; do sleep 60; W=$((W + 1)); [ $W -ge 240 ] && { MSG="no HUMAN27B_400_END within 4 h"; exit 1; }; done
idle
mkdir -p logs/band_select
timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt --envs 1024 --seed 0 --tag band12500 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv assets/game/p3c_identical_t18_release.csv --contact_log > logs/band_select/diag.txt 2>&1
grep -q Traceback logs/band_select/diag.txt && { grep -A12 Traceback logs/band_select/diag.txt | tail -8; MSG="diag crash"; exit 1; }
CHK=$($P -c "
import json, numpy as np
a = json.load(open('logs/diag_bat_contact_h26_base.json'))['per_pitch']; b = json.load(open('logs/diag_bat_contact_band12500.json'))['per_pitch']
h0, h1 = np.array(a['hit'], bool), np.array(b['hit'], bool); e = np.array(b['ev_any'], float)
print(f'reproduction: hits {h1.sum()} vs old {h0.sum()}, same envs {(h0 == h1).mean():.3f}; hits without ev_any {int(np.isnan(e[h1]).sum())}')")
echo "$CHK"
$P scripts/band_select.py --diag logs/diag_bat_contact_band12500.json --out logs/band_select > logs/band_select/run.txt 2>&1 || { tail -5 logs/band_select/run.txt; MSG="band_select crash | $CHK"; exit 1; }
MSG="done | $CHK | logs/band_select/summary.txt"
