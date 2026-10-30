#!/bin/bash
# Human29 (planning 14, 10-30 conditional GO): fine-tune 12500 @ 1.5x on the filtered percept (AIB-PitchedBatting-Human29-v0 =
# Human27b + percept_filter). Waits for PFILTER_END; launches only if the filter-on zero-shot row has in-zone hit >= 0.40 AND
# the percept's last-step crossing |err| (mean |perc_y_last - act_y|) <= 0.03 m; else writes both markers "HOLD" and stops.
# Aim sigmas = max(0.20 / 0.10, 1.2 x the zero-shot start medians: in-zone |axis_off|, gap clamped >= 0).
# +300: slope (axis_off vs plate_y, in-zone hits) flatter than -0.50, |axis_off| hits <= 0.08, hit >= 0.36 -> H29_300_END;
# FAIL stops. +1000 (700 more): slope flatter than -0.30, |dx| <= 0.06, hit >= 0.40, exit >= 20, D share >= 0.25 -> H29_1000_END.
# Diag: diag1024 seed 0, P3c t18 release, jitter 0, bias 0.055, aim 0, --contact_log, --percept_filter.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human29-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
O=logs/h29; mkdir -p $O; TL=$O/train.txt
M3=logs/H29_300_END.txt; M=logs/H29_1000_END.txt
rm -f $M3 $M; MSG="aborted"; M3DONE=0
trap '[ $M3DONE -eq 1 ] || echo "H29_300_END $(date +%H:%M) $MSG" > $M3; echo "H29_1000_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
diag() {  # $1 ckpt $2 tag
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag $2 --timing_noise 0 --timing_bias 0.055 \
    --aim_offset_z 0 --release_csv $R --contact_log --percept_filter > $O/diag_$2.txt 2>&1
  grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt "diag $2"
  [ -f logs/diag_bat_contact_$2.json ] || { MSG="diag $2: no json"; exit 1; }
}
until [ -f logs/PFILTER_END.txt ]; do sleep 30; done
# 10-30 21:2x: the straight-line filter was biased (+3.7 cm last step, HOLD); re-run the zero-shot with the quadratic filter (pf2)
idle
timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt   --armaim $(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt --aim_scripted logs/armaim_auth/scripted_gain.json   --aim_scripted_true --aim_scripted_dry --aim_window 0.60 --percept_filter --aim_scripted_dump logs/armaim/pair_pf2.npz --envs 1024 --seed 0   --tag pf2 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > logs/armaim/diag_pf2.txt 2>&1
grep -q Traceback logs/armaim/diag_pf2.txt && fail logs/armaim/diag_pf2.txt "diag pf2"
[ -f logs/diag_bat_contact_pf2.json ] && [ -f logs/armaim/pair_pf2.npz ] || { MSG="HOLD: no filter-on zero-shot output"; exit 0; }
echo "$(date +%H:%M) pf2 $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_pf2.json --stage 300 --label pf2 | tr '
' ' ') | $($P scripts/armaim_pair.py --wet logs/armaim/pair_pf2.npz --label pf2 | tr '
' ' ')" > $O/pf2.txt
CHK=$($P -c "
import json, numpy as np
P = {k: np.array(v, dtype=float) for k, v in json.load(open('logs/diag_bat_contact_pf2.json'))['per_pitch'].items()}
z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10); h = P['hit'].astype(bool)
hit = (h & z).sum() / z.sum()
ax = np.where(h & np.isfinite(P['dx_any']), P['dx_any'], P['x_on_bat'] - 0.7112)
mA = np.nanmedian(np.abs(ax[z])); mG = np.nanmedian(np.clip(P['gap'][z], 0, None))
W = np.load('logs/armaim/pair_pf2.npz'); c = {n: W['rows'][:, i] for i, n in enumerate(W['names'])}
err = np.mean(np.abs(c['perc_y_last'] - c['act_y']))
sa, sg = max(0.20, 1.2 * mA), max(0.10, 1.2 * mG)
ok = hit >= 0.40 and err <= 0.03
print(f\"{'GO' if ok else 'HOLD'} {sa:.3f} {sg:.3f} zero-shot hit {hit:.3f}, last-step |err| {err:.4f} m, start medians |axis_off| {mA:.3f} gap {mG:.3f} -> sigma_ax {sa:.3f} sigma_gap {sg:.3f}\")")
echo "$(date +%H:%M) $CHK" | tee $O/check.txt
read V SA SG REST <<< "$CHK"
[ "$V" = GO ] || { MSG="HOLD: $CHK"; exit 0; }
export AIB_H29_SIGMA_AX=$SA AIB_H29_SIGMA_GAP=$SG
BASE=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_pf2.json --label "12500 filter zero-shot" | tr '\n' ' ')
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 300 --run_name human_pitched29_1 --resume True \
  --load_run 2026-10-25_13-56-51_human_pitched9b --checkpoint model_12500.pt > $TL 2>&1
grep -q Traceback $TL && fail $TL train_1
D=$(ls -td $B/*human_pitched29_1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
[ -n "$I" ] || { MSG="train_1: no checkpoint"; exit 1; }
diag $D/model_$I.pt h29_$I
G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h29_$I.json --stage 300 --label "model_$I ($D)" | tr '\n' ' ')
echo "H29_300_END $(date +%H:%M) $G | sigma $SA / $SG | base: $BASE" > $M3; M3DONE=1; cat $M3
case "$G" in PASS*) ;; *) MSG="stopped at +300: $G"; exit 0;; esac
idle
$P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations 700 --run_name human_pitched29_2 --resume True \
  --load_run $(basename $D) --checkpoint model_$I.pt > $O/train_2.txt 2>&1
grep -q Traceback $O/train_2.txt && fail $O/train_2.txt train_2
D2=$(ls -td $B/*human_pitched29_2 2>/dev/null | head -1); I2=$([ -n "$D2" ] && last $D2)
[ -n "$I2" ] || { MSG="train_2: no checkpoint"; exit 1; }
diag $D2/model_$I2.pt h29_$I2
G2=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_h29_$I2.json --stage 1000 --label "model_$I2 ($D2)" | tr '\n' ' ')
MSG="$G2 | +300: $G"
