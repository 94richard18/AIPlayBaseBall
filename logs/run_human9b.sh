#!/bin/bash
# Human9 continuation (decision session 10-25): model_12300 -> 12695 (395 its), checkpoint every 50 (yield at the next one).
# Gate vs the delivery model_12300: replace only if ONE OUTS 1000 matches >= 13.1% and diag hit rate >= 0.68. Table from 1024 balls.
# NOT started automatically - start when the decision session gives the slot (after pitcher ho, ~15:30).
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
export SAVE_EVERY=50
R=2026-10-25_12-31-03_human_pitched9
date +"%H:%M start Human9b"
$P -u scripts/train.py --task AIB-PitchedBatting-Human9-v0 --headless --num_envs 2048 --max_iterations 395 --run_name human_pitched9b --resume True --load_run $R --checkpoint model_12300.pt > logs/train_human9b.txt 2>&1
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched9b | head -1); C=$D/$(ls $D | grep model | sort -t_ -k2 -n | tail -1)
date +"%H:%M Human9b ended $C"
grep -E "bat/(hit_rate|success_rate|fair_rate|carry:)" logs/train_human9b.txt | tail -4 | tr -s ' '
$P -u scripts/diag_bat_contact.py --headless --checkpoint $C --tag human9b --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/h9b_contact.txt 2>&1
HR=$(grep -o "hit rate [0-9.]*" logs/h9b_contact.txt | grep -o "[0-9.]*$"); echo "hit rate $HR"
timeout 900 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $C --eval_envs 1024 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h9b.txt 2>&1
$P - <<'PY'
import re, json, numpy as np
PAT = re.compile(r"\[play\] env (\d+): carry\s+([\d.]+) m \| EV\s+([\d.]+) m/s.*?\| LA\s+([-\d.]+) deg \| spray\s+([-+\d.]+) deg \| spin\s+([\d.]+) rpm \(backspin\s+([-+\d.]+)\)")
seen, sw = set(), []
for m in PAT.finditer(open("logs/b_aimy_h9b.txt", errors="ignore").read()):
    g = m.groups(); e = int(g[0])
    if e in seen: continue
    seen.add(e); sw.append(dict(env=e, carry_m=float(g[1]), exit_velo_mps=float(g[2]), launch_deg=float(g[3]), spray_deg=float(g[4]), spin_rpm=float(g[5]), backspin_rpm=float(g[6])))
json.dump(dict(source="logs/b_aimy_h9b.txt", swings=sw), open("logs/b_aimy_table_h9b.json", "w"))
a = np.array([[s["carry_m"], s["spray_deg"], s["exit_velo_mps"]] for s in sw]); fair = np.abs(a[:, 1]) <= 45
print(f"balls {len(sw)}/1024  fair {fair.mean()*100:.1f}%  fair and >= 29 m {(fair & (a[:, 0] >= 29)).sum()/1024*100:.1f}% (12300: 11.3%)  ev med {np.median(a[:, 2]):.1f}")
PY
$P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9b.json --hit_rate $HR --show 0 2>&1 | tail -2
echo "gate: replace 12300 if win >= 13.1% and hit rate $HR >= 0.68"
date +"%H:%M done"
