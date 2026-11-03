#!/bin/bash
# run_h33j_chain.sh (14, 11-03 12:4x): waits for H33H_END (newer than this script), then H33J (run_h33j.sh; start ckpt from
# the H33H result). STOP (logs/h33/STOP) is honoured by run_h33j.sh. Final marker logs/H33J_CHAIN_END.txt.
cd /d/MyProject/AIPlayBaseBall
M=logs/H33J_CHAIN_END.txt; rm -f $M; MSG="aborted"
trap 'echo "H33J_CHAIN_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
MSG="waiting for H33H_END"; until [ -f logs/H33H_END.txt ] && [ logs/H33H_END.txt -nt logs/run_h33j_chain.sh ]; do sleep 60; done
sleep 120  # let H33H's own chain finish writing H33H_CHAIN_END
MSG="H33J"; bash logs/run_h33j.sh > logs/h33j.out 2>&1
MSG="done | $(cut -c1-500 logs/H33J_END.txt)"
