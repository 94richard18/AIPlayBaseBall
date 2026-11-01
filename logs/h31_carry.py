"""Diag json -> fair carry median on in-zone fair hits (env drag + Magnus flight from exit / launch / backspin at contact,
release height 0.9 m, as the training carry reward) and the fell-by-episode-end rate on in-zone pitches. Usage: h31_carry.py diag.json"""
import json
import sys

import numpy as np
import torch

from aibaseball.physics import PhysicsSpec
from aibaseball.physics.ball_flight import launch_state, simulate_flight

P = {k: np.array(v, float) for k, v in json.load(open(sys.argv[1]))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
f = z & (P["hit"] > 0) & (np.abs(P["spray_contact"]) <= 45) & np.isfinite(P["ev_contact"]) & np.isfinite(P["launch_contact"])
cm = float("nan")
if f.any():
    bs = np.nan_to_num(P["backspin_contact"][f])
    v, w = launch_state(torch.tensor(P["ev_contact"][f], dtype=torch.float32), torch.tensor(P["launch_contact"][f], dtype=torch.float32),
                        torch.tensor(bs, dtype=torch.float32))
    pos = torch.zeros(int(f.sum()), 3); pos[:, 2] = 0.9
    cm = float(np.median(simulate_flight(pos, v, w, PhysicsSpec()).carry.numpy()))
print(f"carry fair med {cm:.1f} m (n {int(f.sum())}), fell by episode end {np.nanmean(P['fell'][z] > 0):.3f}")
