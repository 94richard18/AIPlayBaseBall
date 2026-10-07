"""Four-seam fastball grip for the AIB-1 right hand (offline IK).

Index + middle finger pads on top of the ball (about 2.5 cm apart), thumb pad underneath (opposed),
ring finger supporting the side, pinky curled away; the ball is held by the finger pads, not pressed
into the palm (a small gap), as pitchers do for a fastball.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from ..physics.constants import BallSpec
from .humanoid import R_PALM_RIGHT, build_humanoid
from .kinematics import Robot

# fingertip pad points: (link, local point in the link frame, pad radius)
# finger links are authored along the palm +x axis; the pad is on the volar (-z palm) side.
_X = R_PALM_RIGHT @ np.array([1.0, 0, 0])
_VOLAR = R_PALM_RIGHT @ np.array([0, 0, -1.0])
TIPS = {
    "index": ("r_index_distal", 0.020 * _X + 0.006 * _VOLAR, 0.008),
    "middle": ("r_middle_distal", 0.022 * _X + 0.006 * _VOLAR, 0.008),
    "ring": ("r_ring_middle", 0.045 * _X + 0.006 * _VOLAR, 0.008),
}

FINGER_JOINTS = [
    "index_abd", "index_mcp", "index_pip", "index_dip",
    "middle_abd", "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip",
    "thumb_cmc_rot", "thumb_cmc_flex", "thumb_mcp", "thumb_ip",
]


@dataclass
class BallGrip:
    joint_pos: dict[str, float]  # right-hand finger joints
    ball_center_hand: list[float]  # ball centre in the r_hand link frame
    tip_gaps_mm: dict[str, float]
    palm_gap_mm: float

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path) -> "BallGrip":
        with open(path, encoding="utf-8") as f:
            return BallGrip(**json.load(f))


def _thumb_tip(robot: Robot):
    j = robot.joint("r_thumb_ip")
    t_x = None
    # thumb distal is authored along the thumb axis; recover it from the ip joint offset of the previous segment
    jm = robot.joint("r_thumb_mcp")
    t_x = jm.pos / np.linalg.norm(jm.pos)
    t_flex = j.axis  # flex axis; pad faces the palm: perpendicular to the thumb axis, toward the palm
    pad = np.cross(t_flex, t_x)
    return "r_thumb_distal", 0.026 * t_x - 0.006 * pad / np.linalg.norm(pad), 0.0085


def solve_grip(robot: Robot | None = None) -> BallGrip:
    robot = robot or build_humanoid(with_bat=False)
    ball = BallSpec()
    r = ball.radius
    names = [f"r_{n}" for n in FINGER_JOINTS]
    lo = np.array([robot.joint(n).lower for n in names])
    hi = np.array([robot.joint(n).upper for n in names])
    tips = dict(TIPS)
    tips["thumb"] = _thumb_tip(robot)
    palm_volar_z = -0.015  # palm box half thickness (palm frame)

    def hand_frame(q):
        P = robot.fk(q)
        Rh, ph = P["r_hand"]
        return P, Rh, ph

    def tip_pos(P, Rh, ph, key):
        link, off, _ = tips[key]
        Rl, pl = P[link]
        return Rh.T @ (pl + Rl @ off - ph)

    q_init = dict(zip(names, [0.0, 0.6, 0.7, 0.4, 0.0, 0.6, 0.7, 0.4, 0.9, 1.0, 1.4, 1.5, 1.2, 0.6, 0.3, 0.3]))
    c0 = R_PALM_RIGHT @ np.array([0.12, 0.0, -0.045])
    x0 = np.concatenate([[q_init[n] for n in names], c0])

    def residual(x):
        q = dict(zip(names, x[:16]))
        c = x[16:]
        P, Rh, ph = hand_frame(q)
        res = []
        tp = {}
        for key, (_, _, pr) in tips.items():
            tp[key] = tip_pos(P, Rh, ph, key)
            w = 1.0 if key != "ring" else 0.5
            res.append(w * 20 * (np.linalg.norm(tp[key] - c) - (r + pr)))
        # four-seam: index & middle ~2.5 cm apart on top, thumb opposite to their midpoint
        res.append(10 * (np.linalg.norm(tp["index"] - tp["middle"]) - 0.025))
        mid = 0.5 * (tp["index"] + tp["middle"]) - c
        th = tp["thumb"] - c
        res.append(5 * (np.dot(mid, th) / (np.linalg.norm(mid) * np.linalg.norm(th)) + 1.0))
        # ball off the palm: centre at least r + 1 cm from the volar palm plane (palm frame z)
        c_palm = R_PALM_RIGHT.T @ c
        res.append(10 * max(0.0, (palm_volar_z - c_palm[2]) * -1 + (r + 0.010)))
        # ball in front of the palm (distal), centred across it
        res.append(5 * c_palm[1])
        res.append(0.02 * (x[:16] - x0[:16]))
        return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in res])

    lb = np.concatenate([lo, -np.ones(3)])
    ub = np.concatenate([hi, np.ones(3)])
    sol = least_squares(residual, np.clip(x0, lb + 1e-4, ub - 1e-4), bounds=(lb, ub), max_nfev=3000)
    q = dict(zip(names, np.clip(sol.x[:16], lo + 1e-3, hi - 1e-3).tolist()))
    c = sol.x[16:]
    P, Rh, ph = hand_frame(q)
    gaps = {k: float((np.linalg.norm(tip_pos(P, Rh, ph, k) - c) - (r + tips[k][2])) * 1000) for k in tips}
    c_palm = R_PALM_RIGHT.T @ c
    return BallGrip(joint_pos=q, ball_center_hand=c.tolist(), tip_gaps_mm=gaps,
                    palm_gap_mm=float((-(c_palm[2]) + palm_volar_z - r) * 1000))
