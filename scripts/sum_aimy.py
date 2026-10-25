"""Summarize the batter lateral-aim sweep (logs/b_aimy_<y>.txt, showcase --eval_envs 256): hit rate (balls off the bat
/ 256), fair % (of batted balls and of all pitches), fair carry median, share >= 28 m/s, spray quartiles (- = first-base side)."""

import re
import sys

import numpy as np

PAT = re.compile(r"\[play\] env (\d+): carry\s+([\d.]+) m \| EV\s+([\d.]+) m/s.*?\| LA\s+([-\d.]+) deg \| spray\s+([-+\d.]+) deg")
for y in sys.argv[1:]:
    try:
        rows = [tuple(map(float, m.groups())) for m in PAT.finditer(open(f"logs/b_aimy_{y}.txt", encoding="utf-8", errors="ignore").read())]
    except FileNotFoundError:
        print(f"y={y}: not run"); continue
    seen, uniq = set(), []
    for r in rows:  # first ball per env (the eval window is one episode)
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    if not uniq:
        print(f"y={y}: no balls"); continue
    a = np.array(uniq)
    carry, ev, spray = a[:, 1], a[:, 2], a[:, 4]
    fair = np.abs(spray) <= 45.0
    print(f"y={y:>6}: hit {len(a)/256:.3f}  fair {fair.mean()*100:4.1f}% of batted ({fair.sum()/256*100:4.1f}% of pitches)  "
          f"fair carry med {np.median(carry[fair]) if fair.any() else 0:5.1f} m  ev>=28 {np.mean(ev>=28)*100:4.1f}%  "
          f"ev med {np.median(ev):4.1f}  spray q {np.percentile(spray,[25,50,75]).round(0).tolist()}  1B-side foul {np.mean(spray<-45)*100:4.1f}%")
