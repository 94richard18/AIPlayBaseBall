"""KL state per 100-iteration block of a DIAG_JSON file (docs/plans/fastball_restart.md 5-8, Codex round 15): median
post-update total KL, column 2 (waist_yaw) KL, learning rate and schedule down / up counts per block; a block is "near
target" when the median total KL is in [0.0015, 0.006] (desired_kl 0.003: [d/2, 2d]). Warm-up updates (KL 0) are skipped.

python scripts/kl_blocks.py DIAG_JSON
"""
import json
import sys

import numpy as np

R = [r for r in map(json.loads, open(sys.argv[1], encoding="utf-8")) if r.get("kl", 0) > 0]
B = {}
for r in R:
    B.setdefault(r["update"] // 100 * 100, []).append(r)
out = []
for b in sorted(B):
    rs = B[b]
    kt = np.median([r["kl"] for r in rs])
    out.append(f"{b}-{b + 99}: n {len(rs)} KL {kt:.2e} col2 {np.median([r['kl_col'][2] for r in rs]):.1e} "
               f"lr {np.median([r['lr_after'] for r in rs]):.1e} down/up {sum(r.get('sched_down', 0) for r in rs)}/"
               f"{sum(r.get('sched_up', 0) for r in rs)} {'near' if 0.0015 <= kt <= 0.006 else 'OUT'}")
print("kl_blocks: " + " | ".join(out) + (" -> all near target" if out and all(o.endswith("near") for o in out) else " -> not all near"))
