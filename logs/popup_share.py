"""diag json -> fair-hit pop-up share (launch at contact >= 45 deg) and launch q75 on fair hits (|spray| <= 45), the seg-6
launch-ceiling rule (14, 11-01 18:5x; 14496 baseline 0.193 / 37.9). Usage: popup_share.py diag.json -> "share q75 n"."""
import json
import sys

import numpy as np

P = {k: np.array(v, float) for k, v in json.load(open(sys.argv[1]))["per_pitch"].items()}
L = P["launch_contact"]
f = (P["hit"] > 0) & (np.abs(P["spray_contact"]) <= 45) & np.isfinite(L)
print(f"{np.mean(L[f] >= 45):.3f} {np.percentile(L[f], 75):.1f} {f.sum()}" if f.sum() > 1 else "nan nan 0")
