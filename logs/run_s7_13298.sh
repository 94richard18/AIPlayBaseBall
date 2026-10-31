#!/bin/bash
# Joint seed s7 for the designated candidate model_13298 (planning 14, 10-31 16:33): s6 put 3/64 on the ground (bar 2).
# After STOCH_DIAG_END (newer than this chain's start), before Human30f: joint_smoke s7, R3 recover model_1596 + saim, same
# columns as s6 -> logs/S7_13298_END.txt (ground / 64, hits, torso tilt, the pelvis z of each fall). Then releases Human30f:
# 14's logs/h30f/GO was moved to GO.held at 16:3x so run_h30f.sh waits for this; it is moved back here (also from the EXIT
# trap, so a crash never blocks 30f).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export AIB_EFFORT_SCALE=1.5
PC=logs/rsl_rl/aib1_stage/2026-10-25_14-41-27_ho2_tw2plw100/model_7600.pt
RC=$(cat logs/recov3/recover_ckpt.txt)
GY=logs/armaim_auth/scripted_gain.json
JX="--saim $GY --saim_bound 0.4 --saim_window 0.60 --saim_comp full --percept_filter"
H=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched30c_long1 | head -1)/model_13298.pt
O=logs/h30cadopt2; M=logs/S7_13298_END.txt; rm -f $M; MSG="aborted"
touch $O/s7.stamp
trap 'echo "S7_13298_END $(date +%H:%M) $MSG" > $M; [ -f logs/h30f/GO.held ] && mv logs/h30f/GO.held logs/h30f/GO; cat $M' EXIT
isaac() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'isaaclab_env' -and \$_.CommandLine -match 'scripts/' }).Count" | tr -d '\r'; }
freegb() { powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'; }
idle() { until [ "$(isaac)" -eq 0 ] && sleep 60 && [ "$(isaac)" -eq 0 ] && [ "$(freegb)" -ge 20 ]; do sleep 30; done; }
until [ -f logs/STOCH_DIAG_END.txt ] && [ logs/STOCH_DIAG_END.txt -nt $O/s7.stamp ]; do sleep 30; done
idle
timeout 1500 $P -u scripts/joint_smoke.py --headless --pitcher $PC --batter $H --envs 64 --seed 7 --after_done recover_rl \
  --recover_ckpt $RC --timing_csv logs/m2lite/timing.csv --timing_bias 0.055 $JX --trig_dump $O/trig_joint_s7.npz \
  --csv $O/joint_s7.csv > $O/joint_s7.txt 2>&1
grep -q Traceback $O/joint_s7.txt && { MSG="crash joint_s7: $(grep -A14 Traceback $O/joint_s7.txt | tail -2 | tr '\n' ' ')"; exit 1; }
MSG=$($P - <<PY
import csv
f = lambda x: float(x) if x not in ('', 'nan') else float('nan')
b = list(csv.DictReader(open('$O/joint_s7.csv')))
falls = [(i, x['hit'], x['clip'], x['kmh'], round(f(x['min_pelvis_z_2s']), 2)) for i, x in enumerate(b) if f(x['min_pelvis_z_2s']) < 0.35]
tt = sorted(f(x['torso_tilt_1p5_deg']) for x in b if f(x['torso_tilt_1p5_deg']) == f(x['torso_tilt_1p5_deg']))
s6 = list(csv.DictReader(open('$O/joint_s6.csv'))); g6 = sum(f(x['min_pelvis_z_2s']) < 0.35 for x in s6)
bf = [i for i, x in enumerate(b) if x['batter_fell'] == 'True']; bf6 = sum(x['batter_fell'] == 'True' for x in s6)
print(f"joint s7 model_13298 (recover {'$RC'.split('/')[-2]}): on the ground (min pelvis z 2 s < 0.35) {len(falls)}/{len(b)}, "
      f"batter_fell flag {len(bf)}/{len(b)} (envs {bf}; s6 {bf6}/64), hits {sum(x['hit'] == 'True' for x in b)}/{len(b)}, "
      f"torso tilt +1.5 s med {tt[len(tt) // 2] if tt else float('nan'):.1f} deg | falls (env, hit, clip, km/h, min pelvis z): {falls} | "
      f"s6 + s7 ground {g6} + {len(falls)} = {g6 + len(falls)} (R4 needed if > 4)")
PY
)
