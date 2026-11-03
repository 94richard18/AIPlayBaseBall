#!/bin/bash
# run_h33h_chain.sh (14, 11-03 11:5x): wait for the 13997 game-scene tables (launched on their own, H33_TABLES_13997_END) ->
# 14's FILM14_GO (written after the tables) and its FILM14_END -> H33H (run_h33h.sh, form hinge, from 13997). STOP
# (logs/h33/STOP) is honoured by run_h33h.sh. Final marker logs/H33H_CHAIN_END.txt.
cd /d/MyProject/AIPlayBaseBall
M=logs/H33H_CHAIN_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33H_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
MSG="waiting for tables 13997"; until [ -f logs/H33_TABLES_13997_END.txt ] && [ logs/H33_TABLES_13997_END.txt -nt logs/run_h33h_chain.sh ]; do sleep 60; done
S2=$(cut -c1-300 logs/H33_TABLES_13997_END.txt)
MSG="waiting for FILM14_GO | $S2"; until [ -f logs/FILM14_GO.txt ] && [ logs/FILM14_GO.txt -nt logs/H33_TABLES_13997_END.txt ]; do sleep 60; done
MSG="waiting for FILM14_END | $S2"; until [ -f logs/FILM14_END.txt ] && [ logs/FILM14_END.txt -nt logs/FILM14_GO.txt ]; do sleep 60; done
[ -f logs/h33/STOP ] && { MSG="STOP before H33H | $S2"; exit 0; }
MSG="H33H | $S2"; bash logs/run_h33h.sh > logs/h33h.out 2>&1
MSG="done | $S2 | $(cut -c1-400 logs/H33H_END.txt)"
