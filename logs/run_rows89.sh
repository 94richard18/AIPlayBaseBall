#!/bin/bash
# Planning 14 (10-30 21:0x): after PFILTER_END - row 8 = row 6 + full compensation (bat pose seen by the swing policy moved back
# by the Jacobian-mapped offset), true dy; row 9 = row 6 (q comp, + full if row 8's hit slope is flatter) on the FILTERED percept
# (dy = filtered predicted crossing y - nominal, every step) = the deployable scripted aim / ArmAim2 init; then the Human29 chain
# (logs/run_human29.sh: its own GO check). Window 0.60, diag1024 seed 0, dumps + realised vs r5dry (= the 12500 swing alone).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
R=assets/game/p3c_identical_t18_release.csv; GY=logs/armaim_auth/scripted_gain.json
O=logs/armaim; M=logs/ROWS89_END.txt
rm -f $M; MSG="aborted"; OUT=""
trap 'echo "ROWS89_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
run() {  # $1 tag, rest extra args
  T=$1; shift; idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted $GY --aim_window 0.60 \
    --aim_scripted_dump $O/pair_$T.npz "$@" --envs 1024 --seed 0 --tag $T --timing_noise 0 --timing_bias 0.055 \
    --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt && { grep -A14 Traceback $O/diag_$T.txt | tail -10; MSG="$OUT || crash $T"; exit 1; }
  [ -f logs/diag_bat_contact_$T.json ] && [ -f $O/pair_$T.npz ] || { MSG="$OUT || $T: no json / npz"; exit 1; }
}
slope() { $P scripts/armaim_gate.py --diag logs/diag_bat_contact_$1.json | head -1 | sed -E 's/.*slope hits ([-+0-9.]+).*/\1/'; }
until [ -f logs/PFILTER_END.txt ]; do sleep 30; done
run r8wet --aim_scripted_true --aim_obs_comp --aim_obs_comp_full
OUT="$OUT || $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_r8wet.json --stage 300 --label 'row8 full comp, true dy' | tr '\n' ' ') | $($P scripts/armaim_pair.py --wet $O/pair_r8wet.npz --dry $O/pair_r5dry.npz --label row8 | tr '\n' ' ') | $(grep 'window torque' $O/diag_r8wet.txt | tail -1)"
S6=$(slope r6wet); S8=$(slope r8wet)
FULL=$($P -c "print('--aim_obs_comp_full' if $S8 > $S6 else '')")
echo "$(date +%H:%M) row 6 slope $S6, row 8 slope $S8 -> row 9 comp: q ${FULL:++ full}" | tee $O/rows89_choice.txt
run r9wet --percept_filter --aim_obs_comp $FULL
OUT="$OUT || $($P scripts/armaim_gate.py --diag logs/diag_bat_contact_r9wet.json --stage 300 --label "row9 filtered percept, comp q ${FULL:++ full}" | tr '\n' ' ') | $($P scripts/armaim_pair.py --wet $O/pair_r9wet.npz --label row9 | tr '\n' ' ') | $(grep 'armaim scripted' $O/diag_r9wet.txt | tail -3 | tr '\n' ' ')"
MSG="$OUT"
echo "ROWS89_END $(date +%H:%M) $MSG" > $M; cat $M; trap - EXIT
bash logs/run_human29.sh > logs/h29_chain.txt 2>&1
