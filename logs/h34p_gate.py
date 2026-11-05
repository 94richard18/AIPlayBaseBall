"""h34p_gate.py <ref tag> <test tag> <test pose>: H34P pin-until-release gate (14, 11-05) on two --jit_groups diags. PASS = hit >= 0.88,
fair exit median >= 29.0, fair launch median >= +14, pose (h32_form swing-window pose error) <= 0.30 and every app-window jitter
group <= 1.05 x the ref; wait-window groups are reported only (0-ish by construction while pinned). One line."""
import json
import sys

import numpy as np


def load(tag):
    P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"].items()
         if isinstance(v, list) and v and not isinstance(v[0], (list, dict, str))}
    h = P["hit"] > 0
    fa = h & (np.abs(P["spray_contact"]) <= 45)
    g = lambda k: np.nanmean(P[k][np.isfinite(P[k])])
    return dict(hit=h.mean(), exit=np.nanmedian(P["ev_contact"][fa]), launch=np.nanmedian(P["launch_contact"][fa])), g


(r, gr), (t, gt) = load(sys.argv[1]), load(sys.argv[2])
pose = float(sys.argv[3]) if sys.argv[3] not in ("", "nan") else float("nan")
bad = [s for s, ok in (("hit", t["hit"] >= 0.88), ("exit", t["exit"] >= 29.0), ("launch", t["launch"] >= 14.0), ("pose", pose <= 0.30)) if not ok]
J = []
for w in ("app", "wait"):
    cells = []
    for k in ("wrists", "arms", "feet", "legs", "torso"):
        a, b = gr(f"jit_g_{w}_{k}"), gt(f"jit_g_{w}_{k}")
        cells.append(f"{k} {a:.2f}->{b:.2f} ({100 * (b / a - 1):+.0f}%)")
        if w == "app" and not b <= 1.05 * a:
            bad.append(f"app {k}")
    J.append(f"{w}: " + " ".join(cells))
print(("PASS" if not bad else "FAIL " + ",".join(bad))
      + f" | hit {r['hit']:.3f}->{t['hit']:.3f} exit {r['exit']:.1f}->{t['exit']:.1f} launch {r['launch']:+.1f}->{t['launch']:+.1f} pose {pose:.3f} | "
      + " | ".join(J))
