#!/bin/bash
# ArmAim scripted rows 2 (planning 14, 10-30 20:2x): window 0.60, true dy, diag1024 seed 0, every row dumps per-pitch sums
#  (i) row 4 wet + dry pair (realised share per group, ball crossing / percept columns)
#  (ii) row 5 y + z wet + dry pair (realised on z)
#  (iii) row 6 = row 4 + observation compensation (the swing policy sees q - offset); paired with the row 4 dry
# -> scripts/armaim_gate.py + scripts/armaim_pair.py per row -> logs/ARMAIM_ROWS2_END.txt
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
SW=logs/rsl_rl/aib1_tee_batting_mimic/2026-10-25_13-56-51_human_pitched9b/model_12500.pt
RES=$(ls -td logs/rsl_rl/aib1_armaim/*armaim_smoke | head -1)/model_1.pt
R=assets/game/p3c_identical_t18_release.csv
O=logs/armaim; M=logs/ARMAIM_ROWS2_END.txt
rm -f $M; MSG="aborted"
trap 'echo "ARMAIM_ROWS2_END $(date +%H:%M) $MSG" > $M; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ]; do sleep 30; done; }
run() {  # $1 tag, $2 gain json, $3 extra args
  idle
  timeout 1500 $P -u scripts/diag_bat_contact.py --headless --checkpoint $SW --armaim $RES --aim_scripted $2 --aim_scripted_true \
    --aim_window 0.60 --aim_scripted_dump $O/pair_$1.npz $3 --envs 1024 --seed 0 --tag $1 --timing_noise 0 --timing_bias 0.055 \
    --aim_offset_z 0 --release_csv $R --contact_log > $O/diag_$1.txt 2>&1
  grep -q Traceback $O/diag_$1.txt && { grep -A14 Traceback $O/diag_$1.txt | tail -10; MSG="$OUT || crash $1"; exit 1; }
  [ -f logs/diag_bat_contact_$1.json ] && [ -f $O/pair_$1.npz ] || { MSG="$OUT || $1: no json / npz"; exit 1; }
}
gate() { $P scripts/armaim_gate.py --diag logs/diag_bat_contact_$1.json --stage 1000 --label "$2" | tr '\n' ' '; }
pair() { $P scripts/armaim_pair.py --wet $O/pair_$1.npz --dry $O/pair_$2.npz --label "$3" | tr '\n' ' '; }
GY=logs/armaim_auth/scripted_gain.json; GZ=logs/armaim_auth/scripted_gain_yz.json
OUT=""
run r4wet $GY; run r4dry $GY --aim_scripted_dry
OUT="$OUT || $(gate r4wet 'row4 y w0.60') | $(pair r4wet r4dry row4)"; echo "$OUT" > $O/rows2_partial.txt
run r5wet $GZ --aim_scripted_z; run r5dry $GZ "--aim_scripted_z --aim_scripted_dry"
OUT="$OUT || $(gate r5wet 'row5 y+z w0.60') | $(pair r5wet r5dry row5)"; echo "$OUT" > $O/rows2_partial.txt
run r6wet $GY --aim_obs_comp
OUT="$OUT || $(gate r6wet 'row6 y w0.60 obs comp') | $(pair r6wet r4dry row6)"
MSG="$OUT"
