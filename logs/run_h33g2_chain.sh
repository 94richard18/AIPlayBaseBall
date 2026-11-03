#!/bin/bash
# run_h33g2_chain.sh (14, 11-03 09:0x): game-scene tables on h33g_1/model_13498 (both columns, H33_TABLES_13498_END) -> wait for
# 14's FILM13_GO (written after the tables) and its FILM13_END -> H33G segment 2 (run_h33g2.sh). STOP (logs/h33/STOP) is honoured
# by run_h33g2.sh. Final marker logs/H33G2_CHAIN_END.txt.
cd /d/MyProject/AIPlayBaseBall
M=logs/H33G2_CHAIN_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33G2_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
MSG="tables 13498"; bash logs/run_h33_tables.sh logs/rsl_rl/aib1_tee_batting_mimic/2026-11-03_06-50-43_human_pitched_h33g_1/model_13498.pt 13498 h33g_13498 > logs/h33_tables_13498.out 2>&1
S2=$(cut -c1-300 logs/H33_TABLES_13498_END.txt)
MSG="waiting for FILM13_GO | $S2"; until [ -f logs/FILM13_GO.txt ] && [ logs/FILM13_GO.txt -nt logs/H33_TABLES_13498_END.txt ]; do sleep 60; done
MSG="waiting for FILM13_END | $S2"; until [ -f logs/FILM13_END.txt ] && [ logs/FILM13_END.txt -nt logs/FILM13_GO.txt ]; do sleep 60; done
[ -f logs/h33/STOP ] && { MSG="STOP before segment 2 | $S2"; exit 0; }
MSG="segment 2 | $S2"; bash logs/run_h33g2.sh > logs/h33g2.out 2>&1
MSG="done | $S2 | $(cut -c1-400 logs/H33G2_END.txt)"
