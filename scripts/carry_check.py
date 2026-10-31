"""(b) of 14's 15:41 order: env training carry (ball_flight.simulate_flight, the env's model) vs the ONE OUTS first bounce
(game.one_outs.first_bounce: ball_path with grass) on the same exit / launch / backspin grid. CPU only."""
import json
import math
import os
import sys

os.chdir(r"D:/MyProject/AIPlayBaseBall"); sys.path.insert(0, r"D:/MyProject/AIPlayBaseBall")
import numpy as np
import torch

from aibaseball.game.agents import BattedBall
from aibaseball.game.field import MOUND_DIST, FieldSpec
from aibaseball.game.fielding import FieldingSpec
from aibaseball.game.one_outs import INFIELD_RADIUS_M, first_bounce
from aibaseball.physics import PhysicsSpec
from aibaseball.physics.ball_flight import launch_state, simulate_flight
from aibaseball.physics.grass import GrassSpec

sw = json.load(open("logs/h30c_long/table_13298_1.json"))["swings"]
spin = np.array([s["backspin_rpm"] for s in sw]); la_t = np.array([s["launch_deg"] for s in sw])
print(f"table 13298_1: n {len(sw)}, backspin q25 / med / q75 {np.percentile(spin, [25, 50, 75]).round(0)}, "
      f"launch q25 / med / q75 {np.percentile(la_t, [25, 50, 75]).round(1)}")
phys, fld, fs, grass = PhysicsSpec(), FieldSpec(), FieldingSpec(), GrassSpec()
Z0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
LA = np.arange(0, 71, 5.0)
for ev in (18.0, 21.6, 25.0):
    for bs in (float(np.median(spin)), 2500.0):
        n = len(LA)
        v, w = launch_state(torch.full((n,), ev), torch.tensor(LA, dtype=torch.float32), torch.full((n,), bs))
        pos = torch.zeros(n, 3); pos[:, 2] = Z0
        res = simulate_flight(pos, v, w, phys)
        env = res.carry.numpy()
        game, of = [], []
        for la in LA:
            end, land = first_bounce(BattedBall(ev, float(la), 0.0, bs), fld, fs, grass, phys)
            if land is None:
                game.append(float("inf")); of.append(end)
            else:
                x, y = land; game.append(math.hypot(x, y))
                of.append("OF" if math.hypot(x - MOUND_DIST, y) > INFIELD_RADIUS_M and math.hypot(x, y) > MOUND_DIST else "in")
        game = np.array(game)
        print(f"exit {ev:4.1f} spin {bs:5.0f}: env carry best LA {LA[env.argmax()]:.0f} ({env.max():.1f} m), game bounce best LA "
              f"{LA[np.where(np.isfinite(game), game, -1).argmax()]:.0f} ({np.nanmax(np.where(np.isfinite(game), game, np.nan)):.1f} m)")
        print("   LA   " + " ".join(f"{x:5.0f}" for x in LA))
        print("   env  " + " ".join(f"{x:5.1f}" for x in env))
        print("   game " + " ".join(f"{x:5.1f}" for x in game))
        print("   res  " + " ".join(f"{x:>5}" for x in of))
