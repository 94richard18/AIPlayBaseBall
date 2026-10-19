"""Re-solve the four-seam grip against the hand's real collision shapes (CPU, offline).

ball_grip.solve_grip() measured "pad contact" at a pad point 6 mm volar of the distal segment axis plus an 8 mm pad
radius (14 mm off the axis); the collision cylinders are only 8.1 / 8.3 / 9.5 mm (index / middle / thumb distal), so the
scripted grip left the real finger surfaces ~3-4 mm short of the ball and the ball settled ~8 mm off its pose in sim
(diag_grip, gv 7200). This keeps the ball centre where grip.json has it (release geometry unchanged) and solves the
16 finger angles so that the index / middle / thumb distal cylinders touch the ball and nothing else penetrates it.

    python scripts/solve_grip_v2.py [--out assets/aib1_pitcher/grip_v2.json]
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.physics.constants import BallSpec  # noqa: E402
from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(ROOT, "assets", "aib1_pitcher", "grip_v2.json"))
args = parser.parse_args()

robot = build_humanoid(with_bat=False, name="aib1_pitcher")
old = BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json"))
R_BALL = BallSpec().radius
names = [f"r_{n}" for n in FINGER_JOINTS]
lo = np.array([robot.joint(n).lower for n in names])
hi = np.array([robot.joint(n).upper for n in names])
HAND_LINKS = [l for l in robot.links if l == "r_hand" or l.startswith(("r_index", "r_middle", "r_ring", "r_pinky", "r_thumb"))]
CONTACT = ("r_index_distal", "r_middle_distal", "r_thumb_distal")
c_hand = np.asarray(old.ball_center_hand)


def shape_gaps(q):
    """Surface gap (m, < 0 = penetration) from the ball to every collision shape of the hand, keyed (link, i)."""
    P = robot.fk(q)
    Rh, ph = P["r_hand"]
    c = ph + Rh @ c_hand
    out = {}
    for ln in HAND_LINKS:
        Rl, pl = P[ln]
        for i, s in enumerate(robot.links[ln].shapes):
            if not s.collision:
                continue
            if s.kind == "cylinder":
                r, length = s.size
                a = pl + Rl @ (np.asarray(s.pos) - np.asarray(s.rot)[:, 2] * length / 2)
                b = pl + Rl @ (np.asarray(s.pos) + np.asarray(s.rot)[:, 2] * length / 2)
                t = np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
                out[(ln, i)] = np.linalg.norm(c - (a + t * (b - a))) - r - R_BALL
            elif s.kind == "box":
                Rs = Rl @ np.asarray(s.rot)
                local = Rs.T @ (c - (pl + Rl @ np.asarray(s.pos)))
                half = np.asarray(s.size) / 2
                d = np.abs(local) - half
                out[(ln, i)] = np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0) - R_BALL
            elif s.kind == "sphere":
                out[(ln, i)] = np.linalg.norm(c - (pl + Rl @ np.asarray(s.pos))) - s.size[0] - R_BALL
    return out


def q_of(x):
    return dict(zip(names, x)) | {}


x0 = np.array([old.joint_pos[n] for n in names])


def residual(x):
    g = shape_gaps(q_of(x))
    res = []
    for (ln, i), v in g.items():
        if ln in CONTACT and i == 0:
            res.append(1000.0 * v)  # touch: surface gap 0
        else:
            res.append(1000.0 * min(0.0, v + 0.0005))  # elsewhere: no penetration (0.5 mm slack)
    res.append(0.05 * (x - x0))
    return np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in res])


sol = least_squares(residual, np.clip(x0, lo + 1e-4, hi - 1e-4), bounds=(lo, hi), max_nfev=4000)
q_new = dict(zip(names, np.clip(sol.x, lo + 1e-3, hi - 1e-3).tolist()))


def report(lab, q):
    g = shape_gaps(q)
    near = sorted(((v, ln, i) for (ln, i), v in g.items()), key=lambda t: t[0])[:8]
    print(f"{lab}: contact gaps (mm) " + ", ".join(f"{ln.replace('r_', '')} {g[(ln, 0)] * 1000:+.1f}" for ln in CONTACT)
          + "; closest shapes: " + ", ".join(f"{ln.replace('r_', '')}[{i}] {v * 1000:+.1f}" for v, ln, i in near))
    return g


report("old grip.json", old.joint_pos)
g = report("new grip", q_new)
print("joint changes (rad): " + ", ".join(f"{n.replace('r_', '')} {old.joint_pos[n]:+.2f}->{q_new[n]:+.2f}"
                                         for n in names if abs(q_new[n] - old.joint_pos[n]) > 0.02))
at_lim = [n for n in names if q_new[n] <= robot.joint(n).lower + 2e-3 or q_new[n] >= robot.joint(n).upper - 2e-3]
print(f"joints at a limit: {at_lim or 'none'}")
ok = all(abs(g[(ln, 0)]) < 0.001 for ln in CONTACT) and min(g.values()) > -0.001
new = BallGrip(joint_pos=q_new, ball_center_hand=c_hand.tolist(),
               tip_gaps_mm={ln.replace("r_", "").replace("_distal", ""): float(g[(ln, 0)] * 1000) for ln in CONTACT},
               palm_gap_mm=float(g[("r_hand", 0)] * 1000))
new.save(args.out)
print(f"{'PASS' if ok else 'CHECK'}: wrote {args.out}")
