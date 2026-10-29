#!/bin/bash
# Human27 (planning 14, 10-30, contact accuracy): AIB-PitchedBatting-Human27-v0 = Human15 pitches + aim term on every swing
# 6 x exp(-(axis_off/0.08)^2) x exp(-(gap/0.05)^2) at the closest approach (gap clamped >= 0) + hit reward 20 x
# exp(-(dx/0.08)^2) x exp(-((D - 0.010)/0.015)^2); distance / success / exit / backspin / mimic w_contact off; launch,
# spray, w_miss 20 kept. From the delivery 12500 at AIB_EFFORT_SCALE 1.5, 2048 envs. Obs unchanged, recover untouched.
# Diag: diag1024 seed 0, P3c t18, jitter 0, bias 0.055, aim 0, --contact_log. In-zone stats from per_pitch:
#   axis_off |.| median = |x_on_bat - 0.7112| (closest approach; the contact station on hits), hits / misses separately.
# +100 gate: axis_off |.| med <= 0.75 x baseline on in-zone hits AND in-zone misses (baseline 10.2 / 12.4 cm);
#   in-zone hit |dx| med <= 0.080 m; in-zone hit >= 0.31 -> logs/HUMAN27_100_END.txt
# +300 (200 more) gate: |dx| med <= 0.060; D share in [0, 2.5 cm] >= 0.25; in-zone hit >= 0.40; exit med >= 19 m/s
#   -> logs/HUMAN27_300_END.txt
# Gate: Isaac 0 twice 60 s apart before every launch. Markers by EXIT trap (also on failure).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human27-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
R=assets/game/p3c_identical_t18_release.csv
TL=logs/train_human27.txt; O=logs/h27
M1=logs/HUMAN27_100_END.txt; M3=logs/HUMAN27_300_END.txt
rm -f $M1 $M3; STAGE=100; MSG="aborted"
trap 'if [ "$STAGE" = 100 ]; then echo "HUMAN27_100_END $(date +%H:%M) $MSG" > $M1; cat $M1; echo "HUMAN27_300_END $(date +%H:%M) not started ($MSG)" > $M3; else echo "HUMAN27_300_END $(date +%H:%M) $MSG" > $M3; fi; cat $M3' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
mkdir -p $O; touch $O/start
diag() {  # $1 ckpt $2 tag
    idle
    timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h27_$2 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$2.txt 2>&1
    grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt diag_$2
}
st() { $P -c "
import json, numpy as np
P = {k: np.array(v, dtype=float) for k, v in json.load(open('$1'))['per_pitch'].items()}
z = (np.abs(P['plate_y']) <= 0.2525) & (P['plate_z'] >= 0.50) & (P['plate_z'] <= 1.10); h = P['hit'].astype(bool)
ax = np.abs(P['x_on_bat'] - 0.7112); D = P['D_contact'][h & z]
print(f\"{(h & z).sum() / z.sum():.3f} {np.nanmedian(ax[h & z]):.4f} {np.nanmedian(ax[~h & z]):.4f} {np.nanmedian(np.abs(P['dx_contact'][h & z])):.4f} {((D >= 0) & (D <= 0.025)).mean():.3f} {np.nanmedian(P['ev_contact'][h & z]):.1f} {np.nanmedian(P['launch_contact'][h & z]):.1f}\")"; }
train() {  # $1 chunk tag $2 its $3 load_run $4 ckpt
    idle
    $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations $2 --run_name human_pitched27_$1 --resume True --load_run $3 --checkpoint $4 >> $TL 2>&1
    D=$(ls -td $B/*human_pitched27_$1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
    [ -n "$I" ] || { MSG="chunk $1: no checkpoint"; exit 1; }
}
read Z0 AH0 AM0 X0 S0 E0 L0 <<< "$(st logs/diag_bat_contact_h26_base.json)"
echo "$(date +%H:%M) baseline 12500: in-zone $Z0 axis_off hits $AH0 misses $AM0 |dx| $X0 D share $S0 ev $E0 launch $L0"
train 1 100 2026-10-25_13-56-51_human_pitched9b model_12500.pt; D1=$D; I1=$I
diag $D1/model_$I1.pt $I1; read Z AH AM X S E L <<< "$(st logs/diag_bat_contact_h27_$I1.json)"
LINE="model_$I1 ($D1): in-zone $Z (base $Z0) axis_off hits $AH (base $AH0) misses $AM (base $AM0) |dx| $X (base $X0) D share $S (base $S0) ev $E launch $L"
echo "$(date +%H:%M) +100 $LINE"
if ! $P -c "import sys; sys.exit(0 if $AH <= 0.75 * $AH0 and $AM <= 0.75 * $AM0 and $X <= 0.080 and $Z >= 0.31 else 1)"; then MSG="FAIL +100: $LINE"; exit 0; fi
MSG="PASS +100: $LINE"; echo "HUMAN27_100_END $(date +%H:%M) $MSG" > $M1; cat $M1
STAGE=300; MSG="aborted after +100"
train 2 200 $(basename $D1) model_$I1.pt; D3=$D; I3=$I
diag $D3/model_$I3.pt $I3; read Z AH AM X S E L <<< "$(st logs/diag_bat_contact_h27_$I3.json)"
LINE="model_$I3 ($D3): in-zone $Z axis_off hits $AH misses $AM |dx| $X D share $S ev $E launch $L"
echo "$(date +%H:%M) +300 $LINE"
V=$($P -c "print('PASS' if $X <= 0.060 and $S >= 0.25 and $Z >= 0.40 and $E >= 19 else 'FAIL')")
MSG="$V +300: $LINE"
