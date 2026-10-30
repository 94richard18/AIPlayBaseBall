"""Carry columns (planning 14, 10-31) on a diag_bat_contact --contact_log per-pitch JSON, same in-zone hits as
armaim_gate.py: launch median, exit q25 / median / q75, D share in [0, 2.5 cm], fair share of contacts (|spray| <= 45 deg,
the launch direction only: no flight, so a fair-launched ball that lands foul still counts fair).

    python scripts/carry_cols.py --diag logs/diag_bat_contact_<tag>.json [--label X]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--label", default="")
args = ap.parse_args()
P = {k: np.array(v, dtype=float) for k, v in json.load(open(args.diag))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
hz = P["hit"].astype(bool) & z
ev = P["ev_any"][hz]
ev = ev[np.isfinite(ev)]
q = np.percentile(ev, [25, 50, 75]) if len(ev) else [float("nan")] * 3
D = P["D_any"][hz]
D = D[np.isfinite(D)]
sp = P["spray_contact"][hz]
sp = sp[np.isfinite(sp)]
print(f"carry {args.label}: launch {np.nanmedian(P['launch_any'][hz]):.1f}; exit q25 / med / q75 {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}; "
      f"D share {((D >= 0) & (D <= 0.025)).mean():.3f}; fair share {(np.abs(sp) <= 45).mean():.3f} (n {len(sp)} of {hz.sum()} in-zone hits)")
