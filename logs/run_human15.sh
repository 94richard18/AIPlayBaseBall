#!/bin/bash
# Human15 (Human9 + all pitches P3c-shaped 55-75 km/h in the StrikeZone) from the delivery 12500, 300 its, checkpoint
# every 50, one batter Isaac at a time. Start only after the pitcher's pause eval and logs/film_1026.sh are done.
# Falsifier at +100: diag1024 vs P3c t18 (aim 0) in-zone hit rate >= 0.45 (delivery 0.327), else stop.
# Gate (decision session 10-26): in-zone vs P3c >= 0.5; ONE OUTS vs p3c_identical_t18 (in-zone hit rate, aim-0 table) >
# 18.1% with outfield hits >= 5% (if this is the only miss: report first). Fast 88-112 km/h diag recorded, not gated.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
cnt() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'scripts/(train|showcase|eval_stage|diag_)' }).Count" | tr -d '\r'; }
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
inzone() { $P -c "import json; print(json.load(open('logs/diag_bat_contact_h15_$1.json'))['summary']['by_speed']['slow in zone']['hit_rate'])"; }
d() {  # $1 ckpt $2 tag $3 aim, rest extra
    local c=$1 t=$2 a=$3; shift 3
    until [ "$(cnt)" -lt 2 ]; do sleep 20; done
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $c --envs 1024 --seed 0 --tag h15_$t --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a "$@" > logs/h15_${t}_contact.txt 2>&1
}
train() { $P -u scripts/train.py --task AIB-PitchedBatting-Human15-v0 --headless --num_envs 2048 --max_iterations $1 --run_name $2 --resume True --load_run $3 --checkpoint $4 >> logs/train_human15.txt 2>&1; }
date +"%H:%M start Human15 from human_pitched9b/model_12500"
train 100 human_pitched15 2026-10-25_13-56-51_human_pitched9b model_12500.pt
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched15 | head -1); I1=$(last $D)
d $D/model_$I1.pt ${I1}_p3c 0 --release_csv $R; Z1=$(inzone ${I1}_p3c)
echo "$(date +%H:%M) $I1 vs P3c in-zone ${Z1:-none} (stop below 0.45); train ref_err_plate_s $(grep 'mimic/ref_err_plate_s' logs/train_human15.txt | tail -1 | tr -s ' ')"
LIST="$D/model_$I1.pt"
if [ -z "$Z1" ] || $P -c "import sys; sys.exit(0 if float('$Z1') < 0.45 else 1)"; then echo "falsifier: stop at $I1"; else
    train 200 human_pitched15b $(basename $D) model_$I1.pt
    D2=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched15b | head -1); I2=$(last $D2)
    [ -f $D2/model_12700.pt ] && LIST="$LIST $D2/model_12700.pt"; LIST="$LIST $D2/model_$I2.pt"
fi
date +"%H:%M Human15 training ended: $LIST"
grep -E "mimic/ref_err_plate_s|bat/hit_rate" logs/train_human15.txt | tail -2 | tr -s ' '
for M in $LIST; do
    it=$(basename $M .pt); it=${it#model_}
    [ -f logs/diag_bat_contact_h15_${it}_p3c.json ] || d $M ${it}_p3c 0 --release_csv $R
    Z=$(inzone ${it}_p3c); A=$(grep -o "hit rate [0-9.]*" logs/h15_${it}_p3c_contact.txt | grep -o "[0-9.]*$")
    d $M ${it}_fast -0.08; HF=$(grep -o "hit rate [0-9.]*" logs/h15_${it}_fast_contact.txt | grep -o "[0-9.]*$")
    until [ "$(cnt)" -lt 2 ]; do sleep 20; done
    timeout 900 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $M --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R > logs/b_h15_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_h15_$it.txt logs/b_table_h15_$it.json
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_h15_$it.json --hit_rate ${Z:-0} --pitches $T --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$it vs P3c: in-zone ${Z:-none}, all ${A:-none} | fast 88-112 (aim -0.08) ${HF:-none} | ONE OUTS vs P3c: $W"
done
echo "gate: in-zone vs P3c >= 0.5; ONE OUTS vs P3c > 18.1% with outfield_hit >= 5% (only miss -> report)"
echo "HUMAN15_END $(date +%H:%M)"
