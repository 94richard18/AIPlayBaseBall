#!/bin/bash
# Human9 model_12300 batted-ball table (slot B, stats only, hard cap 10 min) -> CPU ONE OUTS 1000 matches at hit_rate 0.73
cd /d/MyProject/AIPlayBaseBall
P=/c/Users/User/anaconda3/envs/isaaclab_env/python.exe
D=$(ls -td logs/rsl_rl/aib1_tee_batting_mimic/*human_pitched9 | head -1)
timeout 600 $P -u scripts/showcase.py --headless --task human_pitched --checkpoint $D/model_12300.pt --eval_envs 256 --timing_noise 0.03 --timing_bias 0.055 --aim_offset_z -0.12 > logs/b_aimy_h9_12300.txt 2>&1
$P scripts/sum_aimy.py h9_12300
$P - <<'PY'
import re, json
PAT = re.compile(r"\[play\] env (\d+): carry\s+([\d.]+) m \| EV\s+([\d.]+) m/s.*?\| LA\s+([-\d.]+) deg \| spray\s+([-+\d.]+) deg \| spin\s+([\d.]+) rpm \(backspin\s+([-+\d.]+)\)")
seen, sw = set(), []
for m in PAT.finditer(open("logs/b_aimy_h9_12300.txt", errors="ignore").read()):
    g = m.groups(); e = int(g[0])
    if e in seen: continue
    seen.add(e); sw.append(dict(env=e, carry_m=float(g[1]), exit_velo_mps=float(g[2]), launch_deg=float(g[3]), spray_deg=float(g[4]), spin_rpm=float(g[5]), backspin_rpm=float(g[6])))
json.dump(dict(source="logs/b_aimy_h9_12300.txt", swings=sw), open("logs/b_aimy_table_h9_12300.json", "w"))
import numpy as np
a = np.array([[s["carry_m"], s["spray_deg"]] for s in sw]); of = ((np.abs(a[:, 1]) <= 45) & (a[:, 0] >= 29)).sum()
print(f"fair and >= 29 m: {of}/256 = {of/256*100:.1f}% (Human7 4.7%)")
PY
$P scripts/one_outs.py --matches 1000 --batter_table logs/b_aimy_table_h9_12300.json --hit_rate 0.73 --show 0 2>&1 | tail -2
date +"%H:%M done"
