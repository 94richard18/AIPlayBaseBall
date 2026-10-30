#!/bin/bash
# Percept filter zero-shot (planning 14, 10-30): frozen 12500 @ 1.5x with cfg.percept_filter (ballistic LSQ crossing
# prediction), no aim offset (ArmAim scripted DRY: offset logged, not applied = the 12500 swing alone, percept columns dumped).
# Rows: filter on; filter on + percept noise/bias 0. After ARMAIM_ROW7_END. -> logs/PFILTER_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim; M=logs/PFILTER_END.txt
rm -f $M; MSG="aborted"; OUT=""
trap 'echo "PFILTER_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
until [ -f logs/ARMAIM_ROW7_END.txt ]; do sleep 30; done
for RW in "pf --percept_filter" "pfclean --percept_filter --percept_clean"; do
  set -- $RW; T=$1; shift
  until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted logs/armaim_auth/scripted_gain.json \
    --aim_scripted_true --aim_scripted_dry --aim_window 0.60 "$@" --aim_scripted_dump $O/pair_$T.npz --envs 1024 --seed 0 --tag $T \
    --timing_noise 0 --timing_bias 0.055 --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$T.txt 2>&1
  grep -q Traceback $O/diag_$T.txt && { grep -A14 Traceback $O/diag_$T.txt | tail -10; MSG="$OUT || crash $T"; exit 1; }
  [ -f $O/pair_$T.npz ] || { MSG="$OUT || $T: no npz"; exit 1; }
  G=$($P scripts/armaim_gate.py --diag logs/diag_bat_contact_$T.json --stage 1000 --label "12500 $T" | tr '\n' ' ')
  OUT="$OUT || $G | $($P scripts/armaim_pair.py --wet $O/pair_$T.npz --label $T | tr '\n' ' ')"
done
MSG="$OUT"
