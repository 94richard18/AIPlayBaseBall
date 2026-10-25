"""W1 / ONE OUTS pitch tiers at any plate_yaw_deg, offline, from eval_stage --release_csv tables (CPU, no Isaac).

The pitcher turned toward first base on the rubber by theta = the release state turned +theta about the rubber
(stage plate_yaw_deg), so one eval at plate_yaw_deg 0 gives the tiers at every theta. Drag + Magnus flight to the plate
(t_max 3 s, as eval_stage), tiers as eval_stage: thrown and |heading| <= 10 deg (T1), reached the plate in the air and
|y| <= 1.0 m (T2), |y| <= 0.5 m and z 0.2-1.5 m (T3), strike (T4).

    python scripts/release_csv_tiers.py <table.csv> [...] [--yaw 0 13.5 16 17 19] [--yaw_auto]
        [--yaw_range 16 20 0.25] [--game] [--write <stem>]
--yaw_auto adds theta = -(median release heading of the at-rest throws) for each table.
--game also counts the ONE OUTS outcomes of every released pitch (as the game's pitch table holds them, at rest or not):
  strikes (agents.StrikeZone) and hit-by-pitch (one_outs BODY_Y_M / BODY_Z_M, a right-handed batter on the 3B side).
--write <stem> (one table, one yaw): writes <stem>_pitches.csv (eval_stage --pitch_csv format) and <stem>_release.csv
  (--release_csv format) at that yaw - the same as an eval at that plate_yaw_deg. Never overwrites.
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
STRIKE_HALF_Y, STRIKE_Z = 0.2159 + 0.0366, (0.50, 1.10)  # aibaseball/game/agents.py StrikeZone
BODY_Y, BODY_Z = (0.50, 0.90), (0.20, 1.70)  # aibaseball/game/one_outs.py, + toward third base = - in this frame

ap = argparse.ArgumentParser()
ap.add_argument("tables", nargs="+")
ap.add_argument("--yaw", type=float, nargs="*", default=[0.0])
ap.add_argument("--yaw_auto", action="store_true")
ap.add_argument("--yaw_range", type=float, nargs=3, default=None, metavar=("LO", "HI", "STEP"))
ap.add_argument("--game", action="store_true")
ap.add_argument("--write", type=str, default=None)
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
    if args.yaw_range:
        lo, hi, st = args.yaw_range
        yaws = [round(x, 3) for x in np.arange(lo, hi + st / 2, st)]
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
        if args.game:
            strike = re & (np.abs(py) <= STRIKE_HALF_Y) & (pz >= STRIKE_Z[0]) & (pz <= STRIKE_Z[1])
            hbp = re & (-py >= BODY_Y[0]) & (-py <= BODY_Y[1]) & (pz >= BODY_Z[0]) & (pz <= BODY_Z[1])
            print(f"           game, all {int(rel.sum())} released: strikes {int(strike.sum())}, hit by pitch {int(hbp.sum())}, "
                  f"not reaching {int((rel & ~re).sum())} | at rest {int((rel & rest).sum())}: strikes {int((strike & rest).sum())}, "
                  f"hit by pitch {int((hbp & rest).sum())}")
        if args.write:
            assert len(args.tables) == 1 and len(yaws) == 1, "--write: one table, one yaw"
            for suf in ("_pitches.csv", "_release.csv"):
                assert not os.path.exists(args.write + suf), f"{args.write + suf} exists - not overwriting"
            with open(args.write + "_pitches.csv", "w", encoding="utf-8") as f:
                f.write("speed_mps,plate_y,plate_z,crossed,flight" + chr(10))
                for i in np.where(rel)[0]:
                    f.write(f"{d['speed'][i]:.3f},{py[i]:.4f},{pz[i]:.4f},{int(re[i])},drag" + chr(10))
            pp, vv, ww = p.numpy(), v.numpy(), w.numpy()
            el = np.degrees(np.arctan2(vv[:, 2], np.linalg.norm(vv[:, :2], axis=1)))
            tier = t1.astype(int) + t2 + t3 + t4
            with open(args.write + "_release.csv", "w", encoding="utf-8") as f:
                f.write("released,at_rest,thrown,release_t,speed,heading,elevation,x,y,z,aero_reached,aero_y,aero_z,tier,"
                        "vx,vy,vz,wx,wy,wz,z_sim,field_z,plate_yaw_deg" + chr(10))
                for i in range(n):
                    f.write(f"{int(rel[i])},{int(rest[i])},{int(thrown[i])},{d['release_t'][i]:.4f},{d['speed'][i]:.3f},"
                            f"{head[i]:.2f},{el[i]:.2f},{pp[i, 0]:.3f},{pp[i, 1]:.3f},{pp[i, 2] - fz:.3f},{int(re[i])},"
                            f"{py[i]:.3f},{pz[i]:.3f},{int(tier[i])},{vv[i, 0]:.4f},{vv[i, 1]:.4f},{vv[i, 2]:.4f},"
                            f"{ww[i, 0]:.3f},{ww[i, 1]:.3f},{ww[i, 2]:.3f},{pp[i, 2]:.4f},{fz:.4f},{yw}" + chr(10))
            print(f"           wrote {args.write}_pitches.csv ({int(rel.sum())} rows) and {args.write}_release.csv ({n} rows)")
