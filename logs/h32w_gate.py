"""h32w_gate.py <ref tag> <test tag>: H32W wrist gate on two --jit_groups diags (14, 11-05): test wait wrists RMS <= 8.0 and no
other group (arms / feet / legs / torso) worse than the ref by > 5% in any window (all / wait / app). Prints 'PASS ...' / 'FAIL ...'."""
import json
import sys

import numpy as np


def load(tag):
    P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"].items()}
    return lambda k: np.nanmean(P[k][np.isfinite(P[k])])


r, t = load(sys.argv[1]), load(sys.argv[2])
bad, rows = [], []
ww = t("jit_g_wait_wrists")
if not ww <= 8.0:
    bad.append(f"wait wrists {ww:.2f} > 8.0")
for w in ("all", "wait", "app"):
    cells = []
    for g in ("wrists", "arms", "feet", "legs", "torso"):
        a, b = r(f"jit_g_{w}_{g}"), t(f"jit_g_{w}_{g}")
        cells.append(f"{g} {a:.2f}->{b:.2f} ({100 * (b / a - 1):+.0f}%)")
        if g != "wrists" and not b <= 1.05 * a:
            bad.append(f"{w} {g} {b:.2f} > 1.05 x {a:.2f}")
    rows.append(f"{w}: " + " ".join(cells))
print(("FAIL " + "; ".join(bad) if bad else "PASS") + " | " + " | ".join(rows))
