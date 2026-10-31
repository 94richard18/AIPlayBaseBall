"""In-zone hit medians from a diag json: ss at contact / swing max / dx at contact. Usage: ssc.py diag.json"""
import json
import sys

import numpy as np

P = {k: np.array(v, float) for k, v in json.load(open(sys.argv[1]))["per_pitch"].items()
     if k in ("plate_y", "plate_z", "hit", "ss_contact", "ss_speed_max", "dx_contact")}
h = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10) & (P["hit"] > 0)
print(f"ss_contact {np.nanmedian(P['ss_contact'][h]):.1f} (swing max {np.nanmedian(P['ss_speed_max'][h]):.1f}), "
      f"dx_contact {np.nanmedian(P['dx_contact'][h]):+.3f}")
