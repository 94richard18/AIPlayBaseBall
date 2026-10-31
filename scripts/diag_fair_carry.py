"""Training-comparable stats from a diag_bat_contact --contact_log JSON (planning 14, 10-31 (c) falsifier): in-zone hit rate,
fair-ball carry mean (each in-zone hit re-flown from its contact exit / launch / spray / backspin with the env's drag +
Magnus flight, launched at the nominal contact height) and launch mean - the env logs bat/hit_rate, bat/fair_carry and
bat/launch_angle as means, so means here too.

    python scripts/diag_fair_carry.py --diag logs/diag_bat_contact_<tag>.json [--label X] [--z0 0.9]
"""
import argparse
import json

import numpy as np
import torch

from aibaseball.physics import PhysicsSpec
from aibaseball.physics.ball_flight import launch_state, simulate_flight

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--z0", type=float, default=0.9)
args = ap.parse_args()
P = {k: np.array(v, dtype=float) for k, v in json.load(open(args.diag))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
hz = P["hit"].astype(bool) & z
ev, la, sp, bs = (P[k][hz] for k in ("ev_contact", "launch_contact", "spray_contact", "backspin_contact"))
ok = np.isfinite(ev) & np.isfinite(la) & np.isfinite(sp) & np.isfinite(bs)
ev, la, sp, bs = ev[ok], la[ok], sp[ok], bs[ok]
t = lambda x: torch.tensor(x, dtype=torch.float32)
v, w = launch_state(t(ev), t(la), t(bs), t(sp))
pos = torch.zeros(len(ev), 3); pos[:, 2] = args.z0
carry = simulate_flight(pos, v, w, PhysicsSpec()).carry.numpy()
fair = np.abs(sp) <= 45
print(f"train-comparable {args.label}: in-zone hit {hz.sum() / max(z.sum(), 1):.3f}; fair_carry mean {carry[fair].mean():.1f} m "
      f"(fair n {fair.sum()}, share < 10 m {(carry[fair] < 10).mean():.2f}); fair_rate {fair.mean():.3f}; launch mean {la.mean():.1f}")
