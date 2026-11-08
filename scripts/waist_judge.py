"""Plan docs/plans/fastball_restart.md 5-16: screening verdict of the waist-effort what-if (diag_chain --waist_effort_scale).

    python scripts/waist_judge.py logs/waist     # reads k1.0.npz, k1.25.npz, k1.5.npz

Order (first hit stops): (1) any k with n_valid < 96 (speed or ratio) -> W3 evidence insufficient; (2) intervention not
effective (waist yaw and pitch motor-torque peak medians both < 1.10 x k = 1.0) -> W3; (3) falls before the release
+> 10 vs k = 1.0 -> W3; (4) W1: speed median >= +0.5 m/s and trunk / pelvis median >= +0.1; W2: speed diff in
(-0.5, +0.5); else W3. Thresholds are uncalibrated empirical screening thresholds (one launch per k).
"""

import os
import sys

import numpy as np

d = sys.argv[1]
K = ["1.0", "1.25", "1.5"]
L = {}
for k in K:
    z = np.load(os.path.join(d, f"k{k}.npz"), allow_pickle=True)
    ok = z["ok"]
    rd = ok[:, 0] & ok[:, 1]
    L[k] = dict(speed=np.asarray(z["speed"]), ratio=z["pk"][rd, 1, 1] / z["pk"][rd, 0, 1], tau=np.median(z["tau_pk"], 0),
                fell=int(z["fell_pre"]), notrel=int(z["not_rel"]))
b = L["1.0"]
for k in K:
    x = L[k]
    print(f"[judge] k {k}: n speed {len(x['speed'])}, n ratio {len(x['ratio'])}, speed median {np.median(x['speed']):.2f} "
          f"({np.median(x['speed']) - np.median(b['speed']):+.2f}), trunk/pelvis {np.median(x['ratio']):.3f} "
          f"({np.median(x['ratio']) - np.median(b['ratio']):+.3f}), motor torque peak yaw / pitch {x['tau'][0]:.1f} / "
          f"{x['tau'][1]:.1f} Nm (x{x['tau'][0] / b['tau'][0]:.2f} / x{x['tau'][1] / b['tau'][1]:.2f}), fell before release "
          f"{x['fell']} ({x['fell'] - b['fell']:+d}), not released {x['notrel']}")
x = L["1.5"]
ds = np.median(x["speed"]) - np.median(b["speed"])
dr = np.median(x["ratio"]) - np.median(b["ratio"])
if any(len(L[k]["speed"]) < 96 or len(L[k]["ratio"]) < 96 for k in K):
    v = "W3 evidence insufficient (n_valid < 96)"
elif not (x["tau"][0] >= 1.10 * b["tau"][0] or x["tau"][1] >= 1.10 * b["tau"][1]):
    v = "W3 intervention not effectively applied (motor torque peak < +10%)"
elif x["fell"] - b["fell"] > 10:
    v = "W3 falls before the release +> 10"
elif ds >= 0.5 and dr >= 0.1:
    v = "W1 positive change meeting the screening thresholds observed this once, to be confirmed"
elif -0.5 < ds < 0.5:
    v = "W2 no positive response reaching the thresholds observed this once (does not exclude H-W)"
else:
    v = "W3 other"
print(f"[judge] k 1.5 vs 1.0: speed {ds:+.2f} m/s, trunk/pelvis {dr:+.3f} -> {v}")
