#!/bin/bash
# run_h33g_chain.sh (14, 11-03 06:0x, all three steps without waiting): (1) run_h33f3.sh = diag + form + joint s6 on h33f_1/model_13100
# (H33F_13100_END); (2) game-scene tables (run_h33_tables.sh, both jitter columns) on 13100 if it passes pose <= 0.20 / key <= 0.11 /
# hit >= 0.887 / in-zone fair launch median >= 5, else on 13200 (H33_TABLES_<I>_END); (3) run_h33g.sh (2 segments) from 13100 if it
# passed everything, else from H33E 12999 (gates' previous = that ckpt's eval). Final marker logs/H33G_CHAIN_END.txt.
cd /d/MyProject/AIPlayBaseBall
M=logs/H33G_CHAIN_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33G_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
F1=2026-11-03_04-50-46_human_pitched_h33f_1; B=logs/rsl_rl/aib1_tee_batting_mimic
MSG="step 1: 13100 eval"; bash logs/run_h33f3.sh > logs/h33f3.out 2>&1
[ -f logs/H33F_13100_END.txt ] && [ logs/H33F_13100_END.txt -nt logs/run_h33f3.sh ] || { MSG="step 1 failed: $(cut -c1-300 logs/H33F3_END.txt)"; exit 1; }
E=$(cat logs/H33F_13100_END.txt); LZ=$(echo "$E" | sed -n 's/.*in-zone fair launch median \([-0-9.na]*\) .*/\1/p')
fld() { echo "$2" | sed -n "s/.* $1 \([0-9.]*\) .*/\1/p"; }
if grep -q "PASS" logs/H33F3_END.txt && awk -v l="$LZ" 'BEGIN{exit !(l ~ /^-?[0-9.]+$/ && l + 0 >= 5)}'; then
  PICK=13100; export G_LR=$F1 G_LC=model_13100.pt G_H=$(fld hit "$E") G_E=$(fld exit "$E") G_L=$LZ G_F=$(fld fair "$E")
else
  PICK=13200; export G_LR=2026-11-03_00-36-33_human_pitched_h33e_1 G_LC=model_12999.pt G_H=0.907 G_E=18.2 G_L=0.3 G_F=0.78
fi
S1="13100: $(cut -c1-160 logs/H33F3_END.txt | sed 's/^H33F3_END //'), launch $LZ -> tables on $PICK, H33G from $G_LC"
MSG="step 2: tables on $PICK | $S1"; bash logs/run_h33_tables.sh $B/$F1/model_$PICK.pt $PICK > logs/h33_tables_$PICK.out 2>&1
S2=$(cut -c1-300 logs/H33_TABLES_${PICK}_END.txt)
MSG="step 3: H33G from $G_LC | $S1 | $S2"; bash logs/run_h33g.sh > logs/h33g.out 2>&1
MSG="done | $S1 | $S2 | $(cut -c1-400 logs/H33G_END.txt)"
