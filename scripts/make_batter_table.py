"""Batter table for the game engine (TableBatter / one_outs.py --batter_table) from a showcase --eval_envs log: first
batted ball per env from the [play] lines. Prints the batted-ball summary (fair and >= 29 m is per pitch).

    python scripts/make_batter_table.py logs/<showcase log>.txt <out>.json
"""

import json
import re
import sys

import numpy as np

PAT = re.compile(r"\[play\] env (\d+): carry\s+([\d.]+) m \| EV\s+([\d.]+) m/s.*?\| LA\s+([-\d.]+) deg \| spray\s+([-+\d.]+) deg"
                 r" \| spin\s+([\d.]+) rpm \(backspin\s+([-+\d.]+)\)")
src, out = sys.argv[1], sys.argv[2]
seen, sw = set(), []
for m in PAT.finditer(open(src, encoding="utf-8", errors="ignore").read()):
    g = m.groups()
    e = int(g[0])
    if e in seen:
        continue
    seen.add(e)
    sw.append(dict(env=e, carry_m=float(g[1]), exit_velo_mps=float(g[2]), launch_deg=float(g[3]), spray_deg=float(g[4]),
                   spin_rpm=float(g[5]), backspin_rpm=float(g[6])))
json.dump(dict(source=src, swings=sw), open(out, "w"))
n = 1024  # pitches per showcase run (--eval_envs)
a = np.array([[s["carry_m"], s["spray_deg"], s["exit_velo_mps"]] for s in sw]) if sw else np.zeros((0, 3))
fair = np.abs(a[:, 1]) <= 45
print(f"{out}: balls {len(sw)}  fair {fair.mean() * 100 if len(sw) else 0:.1f}%  "
      f"fair and >= 29 m {(fair & (a[:, 0] >= 29)).sum() / n * 100:.1f}% of {n}  ev med {np.median(a[:, 2]) if len(sw) else 0:.1f}")
