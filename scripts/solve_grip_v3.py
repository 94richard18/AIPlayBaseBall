"""grip_v3 = grip_v2 + the ring finger curled onto the ball (fourth support), and a support check for old / v2 / v3.

Windup slips are slow roll-outs at ~7 N with the palm sideways (diag_grip, gz 7300 at start noise 0.03): with the
palm sideways gravity points out of the thumb / index / middle triangle. Support check: for every gravity direction in
the palm plane (palm normal horizontal), is the ball centre's projection along gravity inside the convex hull of the
contact points (shapes within 1 mm of the ball surface)?

    python scripts/solve_grip_v3.py [--out assets/aib1_pitcher/grip_v3.json]
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.physics.constants import BallSpec  # noqa: E402
from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip  # noqa: E402
from aibaseball.robot.humanoid import R_PALM_RIGHT, build_humanoid  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(ROOT, "assets", "aib1_pitcher", "grip_v3.json"))
args = parser.parse_args()

robot = build_humanoid(with_bat=False, name="aib1_pitcher")
G = {k: BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", f)) for k, f in
     (("old", "grip.json"), ("v2", "grip_v2.json"))}
R_BALL = BallSpec().radius
names = [f"r_{n}" for n in FINGER_JOINTS]
lo = np.array([robot.joint(n).lower for n in names])
hi = np.array([robot.joint(n).upper for n in names])
HAND = [l for l in robot.links if l == "r_hand" or l.startswith(("r_index", "r_middle", "r_ring", "r_pinky", "r_thumb"))]
c_hand = np.asarray(G["v2"].ball_center_hand)


def shapes(q):
    """(link, i) -> (surface gap m, closest surface point in the hand frame)."""
    P = robot.fk(q)
    Rh, ph = P["r_hand"]
    out = {}
    for ln in HAND:
        Rl, pl = P[ln]
        Rl, pl = Rh.T @ Rl, Rh.T @ (pl - ph)  # hand frame
        for i, s in enumerate(robot.links[ln].shapes):
            if not s.collision:
                continue
            if s.kind == "cylinder":
                r, length = s.size
                ax = np.asarray(s.rot)[:, 2] * length / 2
                a, b = pl + Rl @ (np.asarray(s.pos) - ax), pl + Rl @ (np.asarray(s.pos) + ax)
                t = np.clip(np.dot(c_hand - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
                q_ = a + t * (b - a)
                d = np.linalg.norm(c_hand - q_)
                out[(ln, i)] = (d - r - R_BALL, c_hand + (q_ - c_hand) / d * R_BALL)
            elif s.kind == "box":
                Rs = Rl @ np.asarray(s.rot)
                cs = pl + Rl @ np.asarray(s.pos)
                loc = Rs.T @ (c_hand - cs)
                half = np.asarray(s.size) / 2
                cl = np.clip(loc, -half, half)
                d = np.abs(loc) - half
                gap = np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0) - R_BALL
                pt = cs + Rs @ cl
                out[(ln, i)] = (gap, c_hand + (pt - c_hand) / max(np.linalg.norm(pt - c_hand), 1e-9) * R_BALL)
    return out


def support(q, tol=0.001):
    sh = shapes(q)
    pts = np.array([p for (g, p) in sh.values() if g < tol])
    keys = [k[0].replace("r_", "") for k, (g, p) in sh.items() if g < tol]
    palm_n = R_PALM_RIGHT @ np.array([0.0, 0.0, -1.0])  # volar normal (hand frame)
    e1 = np.cross(palm_n, [1.0, 0, 0]) if abs(palm_n[0]) < 0.9 else np.cross(palm_n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(palm_n, e1)
    ok = 0
    angs = np.radians(np.arange(0, 360, 10))
    for a in angs:
        gdir = np.cos(a) * e1 + np.sin(a) * e2  # gravity in the palm plane (palm sideways)
        u = np.cross(gdir, [0, 0, 1.0]) if abs(gdir[2]) < 0.9 else np.cross(gdir, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(gdir, u)
        below = pts[np.dot(pts - c_hand, gdir) > -0.002]  # contacts on the gravity side of the ball can carry it
        if len(below) < 3:
            continue
        pp = np.c_[np.dot(below - c_hand, u), np.dot(below - c_hand, v)]
        try:
            ConvexHull(pp)
            if Delaunay(pp).find_simplex(np.zeros((1, 2)))[0] >= 0:
                ok += 1
        except Exception:
            pass
    return keys, ok, len(angs)


x0 = np.array([G["v2"].joint_pos[n] for n in names])
RING = ("r_ring_middle", "r_ring_proximal")


def residual(x):
    sh = shapes(dict(zip(names, x)))
    res = []
    for (ln, i), (g, _) in sh.items():
        if ln in ("r_index_distal", "r_middle_distal", "r_thumb_distal") and i == 0:
            res.append(1000.0 * g)  # keep the three contacts
        else:
            res.append(1000.0 * min(0.0, g + 0.0005))  # no penetration
    ring = min(g for (ln, i), (g, _) in sh.items() if ln == "r_ring_middle")
    res.append(1000.0 * ring)  # the ring finger (middle segment + its distal tail) touches
    res.append(0.05 * (x - x0))
    return np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in res])


sol = least_squares(residual, np.clip(x0, lo + 1e-4, hi - 1e-4), bounds=(lo, hi), max_nfev=4000)
q3 = dict(zip(names, np.clip(sol.x, lo + 1e-3, hi - 1e-3).tolist()))
for lab, q in (("old", G["old"].joint_pos), ("v2", G["v2"].joint_pos), ("v3", q3)):
    sh = shapes(q)
    keys, ok, tot = support(q)
    ring = min(g for (ln, i), (g, _) in sh.items() if ln == "r_ring_middle") * 1000
    pen = min(g for g, _ in sh.values()) * 1000
    print(f"{lab}: contacts (< 1 mm) {sorted(set(keys))}; ring gap {ring:+.1f} mm; deepest penetration {pen:+.1f} mm; "
          f"palm-sideways gravity directions supported {ok}/{tot}")
print("v3 joint changes (rad): " + ", ".join(f"{n.replace('r_', '')} {G['v2'].joint_pos[n]:+.2f}->{q3[n]:+.2f}"
                                            for n in names if abs(q3[n] - G["v2"].joint_pos[n]) > 0.02))
lim = [n for n in names if q3[n] <= robot.joint(n).lower + 2e-3 or q3[n] >= robot.joint(n).upper - 2e-3]
print(f"v3 joints at a limit: {lim or 'none'}")
sh = shapes(q3)
BallGrip(joint_pos=q3, ball_center_hand=c_hand.tolist(),
         tip_gaps_mm={k: float(min(g for (ln, i), (g, _) in sh.items() if ln == f"r_{k}") * 1000)
                      for k in ("index_distal", "middle_distal", "thumb_distal", "ring_middle")},
         palm_gap_mm=float(sh[("r_hand", 0)][0] * 1000)).save(args.out)
print(f"wrote {args.out}")
