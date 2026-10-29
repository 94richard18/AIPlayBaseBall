#!/bin/bash
# Human26b (planning 14, 10-30: the one retry after Human26's +100 fail; sigma_dx 0.04 had no gradient at ~10 cm):
# AIB-PitchedBatting-Human26b-v0 = Human15 + impact-step 4 x g_dx + 3 x g_dx x g_D, g_dx = exp(-(dx/0.12)^2),
# g_D = exp(-((D - 0.015)/0.012)^2), from
# the delivery 12500 at AIB_EFFORT_SCALE 1.5, 2048 envs. Swing plane folded into D (logs/contact_offset/d15_check.py).
# 0. baseline: diag1024 (seed 0, P3c t18, jitter 0, bias 0.055, aim 0, --contact_log) on 12500.
# 1. +100 -> diag1024 -> falsifier: in-zone hit >= 0.7 x baseline; in-zone |dx| median <= 0.75 x baseline; in-zone share
#    of D in [0, 2.5 cm] > baseline (and > the joint 35-hit 0.20) -> marker logs/HUMAN26B_100_END.txt (20 films then).
# 2. (20 filmed after HUMAN26_100_END; the chain waits for logs/M3ROUTINE2_END.txt before +100) +200 more -> diag1024 + one showcase table (1024, the 3-table
#    settings) -> falsifier: in-zone hit >= 0.40, launch median 15-30 deg, exit median >= 18 m/s, >= 1 fair ball with
#    carry >= 38.9 m -> marker logs/HUMAN26B_300_END.txt.
# Gate: Isaac 0 twice 60 s apart before every launch. Markers by EXIT trap (also on failure).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
export AIB_EFFORT_SCALE=1.5
TASK=AIB-PitchedBatting-Human26b-v0
B=logs/rsl_rl/aib1_tee_batting_mimic
C0=$B/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
R=assets/game/p3c_identical_t18_release.csv
TL=logs/train_human26b.txt; O=logs/h26b
M1=logs/HUMAN26B_100_END.txt; M3=logs/HUMAN26B_300_END.txt
rm -f $M1 $M3; STAGE=100; MSG="aborted"
trap 'if [ "$STAGE" = 100 ]; then echo "HUMAN26B_100_END $(date +%H:%M) $MSG" > $M1; cat $M1; echo "HUMAN26B_300_END $(date +%H:%M) not started ($MSG)" > $M3; else echo "HUMAN26B_300_END $(date +%H:%M) $MSG" > $M3; fi; cat $M3' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
fail() { grep -A14 Traceback $1 | tail -10; MSG="crash $2"; exit 1; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
mkdir -p $O; touch $O/start
diag() {  # $1 ckpt $2 tag
    idle
    timeout 900 $P -u scripts/diag_bat_contact.py --headless --checkpoint $1 --envs 1024 --seed 0 --tag h26b_$2 --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$2.txt 2>&1
    grep -q Traceback $O/diag_$2.txt && fail $O/diag_$2.txt diag_$2
    grep -E '^\[bat\] (slow in zone|contact_geom)' $O/diag_$2.txt
}
st() { $P -c "
import json; S = json.load(open('logs/diag_bat_contact_h26b_$1.json'))['summary']; g = S['contact_geom']['in-zone hits']
print(S['by_speed']['slow in zone']['hit_rate'], g['abs_dx_med'], g['D_in_0_25'], g['launch'][1], g['ev'][1])"; }
train() {  # $1 chunk tag $2 its $3 load_run $4 ckpt
    idle
    $P -u scripts/train.py --task $TASK --headless --num_envs 2048 --max_iterations $2 --run_name human_pitched26b_$1 --resume True --load_run $3 --checkpoint $4 >> $TL 2>&1
    D=$(ls -td $B/*human_pitched26b_$1 2>/dev/null | head -1); I=$([ -n "$D" ] && last $D)
    [ -n "$I" ] || { MSG="chunk $1: no checkpoint"; exit 1; }
}
cp logs/diag_bat_contact_h26_base.json logs/diag_bat_contact_h26b_base.json  # Human26's 12500 baseline, same diag
read Z0 X0 S0 L0 E0 <<< "$(st base)"
W=0  # 20's routine refilm (the 06:57 one rendered washed out): its marker line says "refilm"
until grep -qi refilm logs/M3ROUTINE2_END.txt 2>/dev/null; do sleep 60; W=$((W + 1)); [ $W -ge 180 ] && { MSG="no logs/M3ROUTINE2_END.txt within 3 h"; exit 1; }; done
echo "$(date +%H:%M) film done: $(cat logs/M3ROUTINE2_END.txt)"
echo "$(date +%H:%M) baseline 12500: in-zone $Z0 |dx| med $X0 D in [0, 2.5 cm] $S0 launch med $L0 ev med $E0"
train 1 100 2026-10-25_13-56-51_human_pitched9b model_12500.pt; D1=$D; I1=$I
TRD=$(grep "bat/contact_abs_dx:" $TL | grep -o "[0-9.]*$" | $P -c "
import sys; v = [float(x) for x in sys.stdin.read().split()]
print(f'trainer bat/contact_abs_dx it 1 {v[0]:.4f} (EMA from 0), its 11-20 {sum(v[10:20]) / 10:.4f}, last 10 {sum(v[-10:]) / 10:.4f} (n {len(v)})')")
echo "$TRD"
diag $D1/model_$I1.pt $I1; read Z X S L E <<< "$(st $I1)"
LINE="model_$I1 ($D1): in-zone $Z (base $Z0) |dx| med $X (base $X0) D in [0, 2.5 cm] $S (base $S0, joint 0.20) launch $L ev $E | $TRD"
echo "$(date +%H:%M) +100 $LINE"
if ! $P -c "import sys; sys.exit(0 if $Z >= 0.7 * $Z0 and $X <= 0.75 * $X0 and $S > $S0 and $S > 0.20 else 1)"; then MSG="FAIL +100: $LINE - line closed"; exit 0; fi
MSG="PASS +100: $LINE"; echo "HUMAN26B_100_END $(date +%H:%M) $MSG" > $M1; cat $M1
STAGE=300; MSG="aborted after +100"
train 2 200 $(basename $D1) model_$I1.pt; D3=$D; I3=$I
diag $D3/model_$I3.pt $I3; read Z X S L E <<< "$(st $I3)"
idle
timeout 900 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D3/model_$I3.pt --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > $O/table_$I3.txt 2>&1
$P scripts/make_batter_table.py $O/table_$I3.txt $O/table_$I3.json || fail $O/table_$I3.txt table
CM=$($P -c "
import json; s = json.load(open('$O/table_$I3.json'))['swings']; f = [w['carry_m'] for w in s if abs(w['spray_deg']) <= 45]
print(f'{max(f, default=0):.1f} {sum(c >= 38.9 for c in f)}')")
read CMAX N39 <<< "$CM"
LINE="model_$I3 ($D3): in-zone $Z (base $Z0) |dx| med $X D in [0, 2.5 cm] $S launch med $L ev med $E | table fair carry max $CMAX m, >= 38.9 m: $N39"
echo "$(date +%H:%M) +300 $LINE"
V=$($P -c "print('PASS' if $Z >= 0.40 and 15 <= $L <= 30 and $E >= 18 and $N39 >= 1 else 'FAIL')")
MSG="$V +300: $LINE"
