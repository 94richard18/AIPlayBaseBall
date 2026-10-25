#!/bin/bash
# Human14 (Human13 + half the pitches shaped like P3c t18 at 55-75 km/h) - NOT started; after pitcher C (decision session).
# START_RUN / START_CKPT: Human13's best checkpoint (default: the delivery Human9b 12500). 400 its, checkpoint every 50,
# one batter Isaac at a time. Falsifier at +100: diag1024 vs P3c >= 0.25 (delivery 0.140, best zero-shot 0.212) and
# diag1024 at 88-112 km/h >= 0.60, else stop. Gate per checkpoint (decision session): ONE OUTS vs p3c_identical_t18 >
# 17.5% with outfield hits >= 5%; diag1024 vs P3c >= 0.5; diag1024 at 88-112 km/h >= 0.60. Eval aim: the better of 0 / -0.08.
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
START_RUN=${START_RUN:-2026-10-25_13-56-51_human_pitched9b}; START_CKPT=${START_CKPT:-model_12500.pt}
R=assets/game/p3c_identical_t18_release.csv; T=assets/game/p3c_identical_t18_pitches.csv
PF="--clamp_ref_phase"
last() { L=$(ls $1 | grep model | sort -t_ -k2 -n | tail -1); L=${L#model_}; echo ${L%.pt}; }
hr() { grep -o "hit rate [0-9.]*" $1 | grep -o "[0-9.]*$"; }
d() {  # $1 ckpt $2 tag $3 aim $4.. extra -> hit rate
    local c=$1 t=$2 a=$3; shift 3
    timeout 600 $P -u scripts/diag_bat_contact.py --headless --checkpoint $c --envs 1024 --seed 0 --tag h14_$t --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a $PF "$@" > logs/h14_${t}_contact.txt 2>&1
    hr logs/h14_${t}_contact.txt
}
train() {  # $1 its $2 run name $3 load run $4 ckpt
    $P -u scripts/train.py --task AIB-PitchedBatting-Human14-v0 --headless --num_envs 2048 --max_iterations $1 --run_name $2 --resume True --load_run $3 --checkpoint $4 >> logs/train_human14.txt 2>&1
}
date +"%H:%M start Human14 from $START_RUN/$START_CKPT"
train 100 human_pitched14 $START_RUN $START_CKPT
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched14 | head -1); I1=$(last $D)
H1=$(d $D/model_$I1.pt ${I1}_p3c 0 --release_csv $R); H2=$(d $D/model_$I1.pt ${I1}_fast -0.08)
echo "$(date +%H:%M) $I1 diag1024 vs P3c (aim 0) ${H1:-none}, 88-112 km/h (aim -0.08) ${H2:-none} (stop if < 0.25 / < 0.60)"
LIST="$D/model_$I1.pt"
if [ -z "$H1" ] || [ -z "$H2" ] || $P -c "import sys; sys.exit(0 if float('$H1') < 0.25 or float('$H2') < 0.60 else 1)"; then echo "falsifier: stop at $I1"; else
    train 300 human_pitched14b $(basename $D) model_$I1.pt
    D2=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched14b | head -1)
    for it in $(ls $D2 | grep model | sed 's/model_//; s/.pt//' | sort -n | awk 'NR%2==0'); do LIST="$LIST $D2/model_$it.pt"; done
    I2=$(last $D2); echo "$LIST" | grep -q "model_$I2.pt" || LIST="$LIST $D2/model_$I2.pt"
fi
date +"%H:%M Human14 training ended: $LIST"
for M in $LIST; do
    it=$(basename $M .pt); it=${it#model_}
    [ -f logs/h14_${it}_p3c_contact.txt ] && A0=$(hr logs/h14_${it}_p3c_contact.txt) || A0=$(d $M ${it}_p3c 0 --release_csv $R)
    A8=$(d $M ${it}_p3c8 -0.08 --release_csv $R)
    [ -f logs/h14_${it}_fast_contact.txt ] && HF=$(hr logs/h14_${it}_fast_contact.txt) || HF=$(d $M ${it}_fast -0.08)
    a=0; HP=$A0; $P -c "import sys; sys.exit(0 if float('${A8:-0}') > float('${A0:-0}') else 1)" && { a=-0.08; HP=$A8; }
    timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $M --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z $a $PF --release_csv $R > logs/b_h14_$it.txt 2>&1
    $P scripts/make_batter_table.py logs/b_h14_$it.txt logs/b_table_h14_$it.json > /dev/null
    W=$($P scripts/one_outs.py --matches 1000 --batter_table logs/b_table_h14_$it.json --hit_rate ${HP:-0} --pitches $T --show 0 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M) model_$it vs P3c: diag aim 0 ${A0:-none} / aim -0.08 ${A8:-none} (eval aim $a) | 88-112 km/h ${HF:-none} | ONE OUTS vs P3c: $W"
done
echo "gate: ONE OUTS vs P3c > 17.5% with outfield_hit >= 5%; diag vs P3c >= 0.5; diag 88-112 >= 0.60"
date +"%H:%M done"
