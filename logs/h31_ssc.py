"""In-zone hit medians from a diag json: ss at contact / swing max / dx at contact, fair share of hits, exit-hinge median
on fair hits (w x clamp(exit - ref, 0, cap)). Usage: ssc.py diag.json [w ref cap]"""
import json
import sys

import numpy as np

P = {k: np.array(v, float) for k, v in json.load(open(sys.argv[1]))["per_pitch"].items()
     if k in ("plate_y", "plate_z", "hit", "ss_contact", "ss_speed_max", "dx_contact", "spray_contact", "ev_contact")}
h = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10) & (P["hit"] > 0)
w, ref, cap = (list(map(float, sys.argv[2:5])) + [3.0, 16.0, 14.0][len(sys.argv[2:5]):])
f = h & (np.abs(P["spray_contact"]) <= 45)
print(f"ss_contact {np.nanmedian(P['ss_contact'][h]):.1f} (swing max {np.nanmedian(P['ss_speed_max'][h]):.1f}), "
      f"dx_contact {np.nanmedian(P['dx_contact'][h]):+.3f}, fair share {f.sum() / max(h.sum(), 1):.2f}, "
      f"exit-hinge med fair {np.nanmedian(w * np.clip(P['ev_contact'][f] - ref, 0, cap)) if f.any() else float('nan'):.1f}")
