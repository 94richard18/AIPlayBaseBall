#!/usr/bin/env bash
# Keep-best search on the start-noise eval: every live run so far peaked within ~100-150 iterations of a good
# checkpoint and then drifted / collapsed (fo, fp, fq, ft, fu, fv, fw). So run short live rounds (train_live.sh with
# NOISY_EVAL=1), each from the best checkpoint so far by the start-noise eval (start_joint_noise 0.02), with a new seed
# per round, until that eval reaches <goal> or the rounds run out.
#   bash scripts/train_select.sh <task> <checkpoint.pt> <best_noisy_so_far> <rounds> <run_prefix> [its=100] [every=25] [goal=30]
# Lines: "[select] round r from <ckpt> (best k)", "[select] new best k (<ckpt>)", "[select] goal reached: k (<ckpt>)".
set -u
cd "$(dirname "$0")/.."
TASK=$1
BEST_CK=$2
BEST=$3
ROUNDS=$4
PRE=$5
ITS=${6:-100}
EVERY=${7:-25}
GOAL=${8:-30}
for r in $(seq 1 "$ROUNDS"); do
    RUN="${PRE}_r$r"
    echo "[select] round $r from $BEST_CK (best $BEST)"
    # identical-start goal 99: never stop a round early, the selection is on the noisy count
    NOISY_EVAL=1 SEED=$((r + ${SEED0:-0})) bash scripts/train_live.sh "$TASK" "$BEST_CK" "$ITS" "$RUN" "$EVERY" 99 > "logs/chain_$RUN.txt" 2>&1
    grep -E "^\[cycles\] [0-9]+(:| with start noise)" "logs/chain_$RUN.txt" | sed 's/^/    /'
    CK=""
    while read -r line; do
        case "$line" in
            "[cycles] "*": "*" up ("*")") CK=$(echo "$line" | sed 's/.*(\(.*\))$/\1/') ;;
            "[cycles] "*" with start noise 0.02: "*" up")
                K=$(echo "$line" | sed 's/.*: \([0-9?]*\) up$/\1/')
                if [ "$K" != "?" ] && [ -n "$CK" ] && [ "$K" -gt "$BEST" ]; then
                    BEST=$K; BEST_CK=$CK; echo "[select] new best $BEST ($BEST_CK)"
                fi ;;
        esac
    done < <(grep -E "^\[cycles\] [0-9]+(:| with start noise)" "logs/chain_$RUN.txt")
    if [ "$BEST" -ge "$GOAL" ]; then echo "[select] goal reached: $BEST ($BEST_CK)"; exit 0; fi
done
echo "[select] rounds done, best $BEST ($BEST_CK)"
exit 1
