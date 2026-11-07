"""S1D mechanism falsifier (docs/plans/fastball_restart.md 5-7, Codex round 13): over actor updates 7661-7670 of a
DIAG_JSON file, PASS when all three hold: (1) learning rate after the update > 1e-6 in >= 8 of 10; (2) median absolute
KL of column 2 (waist_yaw) <= 1.5e-3; (3) median total KL <= 0.025. Column 2's KL share is printed as a side note.

python scripts/s1d_mech.py DIAG_JSON
"""
import json
import sys

import numpy as np

R = {r["update"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8")) if "kl_col" in r}
W = [R[u] for u in range(7661, 7671) if u in R]
if len(W) < 10:
    print(f"s1d_mech: only {len(W)} of 10 updates -> WAIT")
    sys.exit(0)
lr = np.array([r["lr_after"] for r in W])
k2 = np.array([r["kl_col"][2] for r in W])
kt = np.array([r["kl"] for r in W])
c1, c2, c3 = (lr > 1e-6).sum() >= 8, np.median(k2) <= 1.5e-3, np.median(kt) <= 0.025
share = np.median(k2 / np.maximum(kt, 1e-12))
down = sum(r.get("sched_down", 0) for r in W); up = sum(r.get("sched_up", 0) for r in W)
print(f"s1d_mech: lr>1e-6 {int((lr > 1e-6).sum())}/10 (median {np.median(lr):.1e}), col2 KL median {np.median(k2):.2e}, "
      f"total KL median {np.median(kt):.2e}, col2 share {share:.2f}, schedule down/up {down}/{up} -> "
      f"{'PASS' if c1 and c2 and c3 else 'FAIL'}")
