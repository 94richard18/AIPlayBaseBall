"""E10 side by side (14, 20:3x): diag jsons -> one line per row: hit / exit / launch / ss_contact / dqt / torso jitter.
Usage: e10_table.py tag ... (logs/diag_bat_contact_<tag>.json)"""
import json
import sys

import numpy as np

for t in sys.argv[1:]:
    try:
        P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{t}.json"))["per_pitch"].items()}
    except OSError:
        print(f"{t}: missing"); continue
    z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
    h = z & (P["hit"] > 0)
    print(f"{t}: hit {h.sum() / max(z.sum(), 1):.3f} exit {np.nanmedian(P['ev_contact'][h]):.1f} launch {np.nanmedian(P['launch_contact'][h]):.1f} "
          f"ss {np.nanmedian(P['ss_contact'][h]):.1f} | dqt {np.nanmedian(P['jit_dqt'][z]):.4f} (q90 {np.nanpercentile(P['jit_dqt'][z], 90):.4f}) "
          f"torso {np.nanmedian(P['jit_torso_hp'][z]):.3f} (q90 {np.nanpercentile(P['jit_torso_hp'][z], 90):.3f}) "
          f"da {np.nanmedian(P['jit_da'][z]):.4f}")
