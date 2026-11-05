"""h34q_gain.py <ref tag> <test tag>: H34Q gain check (14, 11-05) - app-window torso / legs / feet jitter each <= 0.90 x the ref
(16300). Prints 'GAIN ...' or 'NO GAIN ...'."""
import json
import sys

import numpy as np


def load(tag):
    P = json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"]
    return lambda k: np.nanmean(np.array(P[k], float))


r, t = load(sys.argv[1]), load(sys.argv[2])
c = {k: (r(f"jit_g_app_{k}"), t(f"jit_g_app_{k}")) for k in ("torso", "legs", "feet")}
ok = all(b <= 0.90 * a for a, b in c.values())
print(("GAIN" if ok else "NO GAIN") + " (app <= -10%: " + " ".join(f"{k} {a:.2f}->{b:.2f} ({100 * (b / a - 1):+.0f}%)" for k, (a, b) in c.items()) + ")")
