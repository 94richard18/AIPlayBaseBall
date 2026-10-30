#!/bin/bash
# Human27b resume (planning 14, 10-30): model_12599 (2026-10-30_08-22-59_human_pitched27b_1) + 300 its = +400 total, same
# terms. Gate ONLY at +400 -> logs/HUMAN27B_400_END.txt: in-zone |axis_off| median on hits <= 0.087 m AND plate_y slope
# on hits > -0.70 AND in-zone hit >= 0.31 (diag1024 seed 0, P3c t18, jitter 0, bias 0.055, aim 0, --contact_log).
# Marker line also carries the trainer bat/aim_abs_ax every 50 its. Fail = the accuracy training route is closed.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human27b-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
TL=logs/train_human27b_400.txt; O=logs/h27b
M=logs/HUMAN27B_400_END.txt
rm -f $M; MSG="aborted"
trap 'echo "HUMAN27B_400_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
mkdir -p $O
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched27b_2 --resume True --load_run 2026-10-30_08-22-59_human_pitched27b_1 --checkpoint model_12599.pt > $TL 2>&1
D=$(ls -td $B/*human_pitched27b_2 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="no checkpoint"; exit 1; }
AX=$(cat logs/train_human27b.txt $TL | grep "bat/aim_abs_ax:" | grep -o "[-0-9.]*$" | $P -c "
import sys; v = [float(x) for x in sys.stdin.read().split()]
print('aim_abs_ax per 50 its (mean of each block, its 1-400): ' + ' '.join(f'{sum(v[i:i + 50]) / len(v[i:i + 50]):.3f}' for i in range(0, len(v), 50)))")
idle
timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $D/model_$I.pt --envs 1024 --seed 0 --tag h27b_$I --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$I.txt 2>&1
grep -q Traceback $O/diag_$I.txt && fail $O/diag_$I.txt diag
OUT=$($P -c "
import json, numpy as np
P = {k: np.array(v, dtype=float) for k, v in json.load(open('logs/diag_bat_contact_h27b_$I.json'))['per_pitch'].items()}
z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10); h = P['hit'].astype(bool)
ax = P['x_on_bat'] - 0.7112; D = P['D_contact'][h & z]
s = []
for m in (h & z, ~h & z):
    y, a = P['plate_y'][m], ax[m]; k = np.isfinite(y) & np.isfinite(a); b = np.polyfit(y[k], a[k], 1)
    s.append((b[0], np.corrcoef(y[k], a[k])[0, 1], np.std(a[k] - np.polyval(b, y[k]))))
Z, AH = (h & z).sum() / z.sum(), np.nanmedian(np.abs(ax[h & z]))
ok = AH <= 0.087 and s[0][0] > -0.70 and Z >= 0.31
print(('PASS' if ok else 'FAIL') + f' +400 model_$I: in-zone hit {Z:.3f} |axis_off| hits {AH:.4f} misses {np.nanmedian(np.abs(ax[~h & z])):.4f} |dx| {np.nanmedian(np.abs(P[\"dx_contact\"][h & z])):.4f} D share {((D >= 0) & (D <= 0.025)).mean():.3f} ev {np.nanmedian(P[\"ev_contact\"][h & z]):.1f} launch {np.nanmedian(P[\"launch_contact\"][h & z]):.1f} | slope / r / resid sd: hits {s[0][0]:+.2f} / {s[0][1]:+.2f} / {s[0][2]:.3f}, misses {s[1][0]:+.2f} / {s[1][1]:+.2f} / {s[1][2]:.3f}')")
MSG="$OUT ($D) | $AX"
