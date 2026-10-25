"""W1 / ONE OUTS pitch tiers at any plate_yaw_deg, offline, from eval_stage --release_csv tables (CPU, no Isaac).

The pitcher turned toward first base on the rubber by theta = the release state turned +theta about the rubber
(stage plate_yaw_deg), so one eval at plate_yaw_deg 0 gives the tiers at every theta. Drag + Magnus flight to the plate
(t_max 3 s, as eval_stage), tiers as eval_stage: thrown and |heading| <= 10 deg (T1), reached the plate in the air and
|y| <= 1.0 m (T2), |y| <= 0.5 m and z 0.2-1.5 m (T3), strike (T4).

    python scripts/release_csv_tiers.py <table.csv> [...] [--yaw 0 13.5 16 17 19] [--yaw_auto]
--yaw_auto adds theta = -(median release heading of the at-rest throws) for each table.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aibaseball.physics.ball_flight import simulate_to_plane  # noqa: E402
from aibaseball.physics.constants import PhysicsSpec  # noqa: E402

PLATE_X, ZONE_Y = 18.44, 0.0

ap = argparse.ArgumentParser()
ap.add_argument("tables", nargs="+")
ap.add_argument("--yaw", type=float, nargs="*", default=[0.0])
ap.add_argument("--yaw_auto", action="store_true")
args = ap.parse_args()
spec = PhysicsSpec()


def turn(a, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1], a[:, 2]], 1)


for path in args.tables:
    d = np.genfromtxt(path, delimiter=",", names=True)
    y0 = float(d["plate_yaw_deg"][0])  # the table is already in the plate frame of this yaw
    rel, rest, thrown = d["released"] == 1, d["at_rest"] == 1, d["thrown"] == 1
    pos0 = np.stack([d["x"], d["y"], d["z_sim"]], 1)
    vel0 = np.stack([d["vx"], d["vy"], d["vz"]], 1)
    om0 = np.stack([d["wx"], d["wy"], d["wz"]], 1)
    fz = float(d["field_z"][0])
    yaws = list(args.yaw)
    if args.yaw_auto and (rest & thrown).any():
        yaws.append(round(-float(np.median(d["heading"][rest & thrown])) + y0, 1))
    n = len(d)
    print(f"== {os.path.basename(path)}: {n} episodes, at rest + thrown {int((rest & thrown).sum())}, table yaw {y0:g}")
    for yw in yaws:
        th = np.radians(yw - y0)
        p, v, w = (torch.tensor(turn(a, th), dtype=torch.float64) for a in (pos0, vel0, om0))
        cr, vp, _, re = simulate_to_plane(p, v, w, spec, PLATE_X, dt=0.005, t_max=3.0, ground_z=fz)
        cr, re = cr.numpy(), re.numpy() & rel
        py, pz = cr[:, 1] - ZONE_Y, cr[:, 2] - fz
        head = np.degrees(np.arctan2(v[:, 1].numpy(), v[:, 0].numpy()))
        t1 = thrown & (np.abs(head) <= 10.0)
        t2 = t1 & re & (np.abs(py) <= 1.0)
        t3 = t2 & (np.abs(py) <= 0.5) & (pz >= 0.2) & (pz <= 1.5)
        t4 = t3 & (np.abs(py) <= 0.2159 + 0.0366) & (pz >= 0.50) & (pz <= 1.10)
        m = rest & thrown & re
        print(f"  yaw {yw:5.1f}: heading median {np.median(head[rest & thrown]) if (rest & thrown).any() else 0:+.1f} | "
              f"at rest: T2 {int((rest & t2).sum())}, T3+ {int((rest & t3).sum())}, T4 {int((rest & t4).sum())} /{n} | "
              f"reached in the air {int((rest & thrown & re).sum())}" + (f", plate y p10/50/90 {np.percentile(py[m], 10):+.2f}/"
              f"{np.median(py[m]):+.2f}/{np.percentile(py[m], 90):+.2f} m, z {np.percentile(pz[m], 10):.2f}/"
              f"{np.median(pz[m]):.2f}/{np.percentile(pz[m], 90):.2f} m, speed {np.median(vp.numpy()[m]):.1f} m/s" if m.any() else ""))
