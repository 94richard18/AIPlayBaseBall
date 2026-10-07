"""Batting-stance IK (offline, numpy/scipy) for a right-handed batter.

World layout used by the task (Z up):
  * tee / home plate at the origin, center field = +X, pitcher side = +X.
  * right-handed batter faces the plate (-Y world), so robot +y (left) points to +X.
Robot frame (pelvis): x forward (toward the plate), y left (toward the pitcher), z up.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..physics.bat_model import BatSpec
from .humanoid import build_humanoid, palm_grip_frame
from .kinematics import Robot

# Finger joint targets for a closed power grip around a ~2.6 cm handle.
GRIP_FINGERS = {
    "index_abd": 0.0, "index_mcp": 1.25, "index_pip": 1.35, "index_dip": 0.75,
    "middle_abd": 0.0, "middle_mcp": 1.25, "middle_pip": 1.40, "middle_dip": 0.75,
    "ring_mcp": 1.30, "ring_pip": 1.40,
    "pinky_mcp": 1.00, "pinky_pip": 1.30,
    "thumb_cmc_rot": 0.6, "thumb_cmc_flex": 0.0, "thumb_mcp": 0.0, "thumb_ip": 0.2,
}


@dataclass
class StanceTargets:
    pelvis_height: float = 0.94
    foot_y: float = 0.40  # half stance width (robot frame)
    foot_x: float = 0.0
    bottom_hand_pos: tuple = (0.20, -0.14, 0.36)  # relative to pelvis (robot frame)
    bat_dir: tuple = (-0.25, -0.35, 0.90)  # knob -> barrel (robot frame)
    torso_lean: float = 0.20  # waist pitch nominal (forward)


@dataclass
class Stance:
    root_height: float
    joint_pos: dict[str, float]
    # loop joint (top hand <-> bat) local positions
    top_hand_local: list[float]
    bat_top_local: list[float]
    residual: float
    info: dict = field(default_factory=dict)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "Stance":
        with open(path, encoding="utf-8") as f:
            return Stance(**json.load(f))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def solve_stance(robot: Robot | None = None, bat: BatSpec | None = None, tgt: StanceTargets | None = None) -> Stance:
    bat = bat or BatSpec()
    robot = robot or build_humanoid(with_bat=True, bat=bat)
    tgt = tgt or StanceTargets()

    names = [j.name for j in robot.actuated if not any(k in j.name for k in GRIP_FINGERS)]
    lo = np.array([robot.joint(n).lower for n in names])
    hi = np.array([robot.joint(n).upper for n in names])
    nominal = {
        "hip_pitch": -0.35, "knee": 0.70, "ankle_pitch": -0.35,
        "shoulder_pitch": -0.9, "elbow": 1.4, "waist_pitch": tgt.torso_lean,
    }
    q0 = np.array([next((v for k, v in nominal.items() if n.endswith(k)), 0.0) for n in names])
    q0 = np.clip(q0, lo + 1e-3, hi - 1e-3)
    fingers = {f"{s}_{k}": v for s in ("l", "r") for k, v in GRIP_FINGERS.items()}

    root_p = np.array([0.0, 0.0, tgt.pelvis_height])
    sole_off = np.array([0.04, 0.0, -0.08])  # foot link frame -> sole centre
    bat_dir = _unit(tgt.bat_dir)
    hand_l = root_p + np.array(tgt.bottom_hand_pos)
    gp_l, _ = palm_grip_frame("l")
    gp_r, gd_r = palm_grip_frame("r")

    def residual(x):
        q = dict(zip(names, x)) | fingers
        P = robot.fk(q, root_pos=root_p)
        res = []
        for s, sy in (("l", 1.0), ("r", -1.0)):
            R, p = P[f"{s}_foot"]
            sole = p + R @ sole_off
            res += list(10 * (sole - np.array([tgt.foot_x + 0.04, sy * tgt.foot_y, 0.0])))
            res += list(5 * (R - np.eye(3)).ravel()[[2, 5, 6, 7, 1]])  # flat & no yaw
        Rb, pb = P["bat"]
        Rl, pl = P["l_hand"]
        res += list(10 * (pl + Rl @ gp_l - hand_l))
        res += list(5 * (Rb[:, 0] - bat_dir))
        Rr, pr = P["r_hand"]
        top_on_bat = pb + Rb[:, 0] * bat.top_hand_x
        res += list(10 * (pr + Rr @ gp_r - top_on_bat))
        res += list(3 * (Rr @ gd_r - Rb[:, 0]))
        # torso upright-ish (z axis of torso close to vertical leaned forward by torso_lean)
        Rt, _ = P["torso"]
        res += list(2 * (Rt[:, 2] - np.array([math.sin(tgt.torso_lean), 0, math.cos(tgt.torso_lean)])))
        res += list(0.05 * (x - q0))
        return np.array(res)

    sol = least_squares(residual, q0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=4000)
    # keep strictly inside the limits (Isaac Lab rejects defaults that sit exactly on a limit)
    q = dict(zip(names, np.clip(sol.x, lo + 1e-3, hi - 1e-3).tolist())) | fingers
    P = robot.fk(q, root_pos=root_p)
    Rb, pb = P["bat"]
    Rr, pr = P["r_hand"]
    err_top = float(np.linalg.norm(pr + Rr @ gp_r - (pb + Rb[:, 0] * bat.top_hand_x)))
    feet_err = [float(np.linalg.norm(P[f"{s}_foot"][1] + P[f"{s}_foot"][0] @ sole_off
                                     - np.array([tgt.foot_x + 0.04, sy * tgt.foot_y, 0.0])))
                for s, sy in (("l", 1), ("r", -1))]
    sweet = pb + Rb[:, 0] * bat.sweet_spot_x
    return Stance(
        root_height=tgt.pelvis_height,
        joint_pos=q,
        top_hand_local=gp_r.tolist(),
        bat_top_local=[bat.top_hand_x, 0.0, 0.0],
        residual=float(sol.cost),
        info=dict(top_hand_gap_m=err_top, feet_err_m=feet_err, sweet_spot_robot=sweet.tolist(),
                  bat_dir=Rb[:, 0].tolist(), at_limit=[n for n, v in zip(names, sol.x)
                                                       if v <= robot.joint(n).lower + 1e-3 or v >= robot.joint(n).upper - 1e-3]),
    )


if __name__ == "__main__":
    st = solve_stance()
    print("cost", st.residual)
    for k, v in st.info.items():
        print(k, v)
    for k, v in st.joint_pos.items():
        if "_" in k and not any(f in k for f in GRIP_FINGERS):
            print(f"  {k:22s} {v:+.3f}")
