"""Synthesised pitching follow-through: the OBP captures stop ~0.14 s after release, mid-motion.

The appended segment (IK per frame with the robot's own kinematics):
  * lead (left) foot stays planted where the capture ends;
  * the trailing (right) foot swings over and lands beside the lead foot (quintic path + lift);
  * the pelvis decelerates from its captured velocity to a point between the feet, rises and squares up to the plate;
  * waist and arms decelerate from their captured velocity to the set position (first frame of the clip).
Without it the imitation target froze on the last captured frame (trailing foot in the air, body still moving
forward), and the pitcher fell ~0.2 s after every release.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from ..robot.kinematics import Robot
from .retarget import KEYPOINTS, RetargetResult, _quat_wxyz

FOOT_KEYS = ("ankle", "toe", "heel")
LEGS = ("hip", "knee", "ankle")


def _quintic(p0, v0, p1, T, t):
    """Position at t of the quintic from (p0, v0, a=0) to (p1, v=0, a=0) over T; held at p1 afterwards."""
    s = np.clip(t / T, 0.0, 1.0)
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    g = s - (6 * s**3 - 8 * s**4 + 3 * s**5)  # velocity-carrying term: g(0)=0, g'(0)=1, g(1)=g'(1)=g''(1)=0
    return p0 + v0 * T * g + (p1 - p0) * h


def append_follow_through(res: RetargetResult, robot: Robot, fingers: dict, duration: float = 0.8, rate: float = 120.0,
                          land_time: float = 0.40, settle_time: float = 0.55, stance_width: float = 0.42,
                          lift: float = 0.10, pelvis_rise: float = 0.08) -> RetargetResult:
    names = res.joint_names
    lo = np.array([robot.joint(n).lower for n in names])
    hi = np.array([robot.joint(n).upper for n in names])
    dt = 1.0 / rate
    rv = Rotation.from_quat(np.concatenate([res.root_quat[:, 1:], res.root_quat[:, :1]], -1)).as_rotvec()
    X = np.concatenate([res.root_pos, rv, res.joint_pos], 1)
    x_end = X[-1]
    v_end = (X[-1] - X[-3]) / (2 * dt)  # captured velocity at the end (joints, pelvis)

    def fk(x):
        return robot.fk(dict(zip(names, x[6:])) | fingers,
                        root_pos=x[:3], root_rot=Rotation.from_rotvec(x[3:6]).as_matrix())

    def keypts(P, side):
        out = {}
        for k in FOOT_KEYS:
            link, off, _ = KEYPOINTS[f"{side}_{k}"]
            Rl, pl = P[link]
            out[k] = pl + Rl @ np.asarray(off)
        return out

    P1, P0 = fk(X[-1]), fk(X[-3])
    lead = keypts(P1, "l")
    trail1, trail0 = keypts(P1, "r"), keypts(P0, "r")
    ground = min(lead["toe"][2], lead["heel"][2])
    # trailing foot lands beside the lead foot (to the throwing-arm side), toes toward the plate (+x)
    lead_c = (lead["toe"] + lead["heel"]) / 2
    land_c = lead_c + np.array([-0.05, -stance_width, 0.0])
    half = (KEYPOINTS["r_toe"][1][0] - KEYPOINTS["r_heel"][1][0]) / 2
    land_c[2] = ground
    land = {"toe": land_c + [half, 0, 0], "heel": land_c - [half, 0, 0]}
    land["ankle"] = land["heel"] + [-KEYPOINTS["r_heel"][1][0], 0, lead["ankle"][2] - ground]

    # pelvis: between the feet, slightly taller, square to the plate, upright
    feet_mid = (lead_c + land_c) / 2
    p_goal = np.array([feet_mid[0] - 0.05, feet_mid[1], x_end[2] + pelvis_rise])
    R_end = Rotation.from_rotvec(x_end[3:6])
    R_goal = Rotation.from_euler("z", 0.0)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_end, R_goal]))
    upper = np.array([not any(k in n for k in LEGS) for n in names])
    q_goal = np.where(upper, res.joint_pos[0], x_end[6:])

    T = int(round(duration * rate))
    frames, kp_err = [], []
    x = x_end.copy()
    x_prev = x_end.copy()
    big = np.full(6, np.inf)
    lb, ub = np.concatenate([-big, lo]), np.concatenate([big, hi])
    for f in range(1, T + 1):
        t = f * dt
        tgt = {"l": lead}
        sw = {}
        for k in FOOT_KEYS:
            v0 = (trail1[k] - trail0[k]) / (2 * dt)
            p = _quintic(trail1[k], v0, land[k], land_time, t)
            p[2] += lift * math.sin(math.pi * min(t / land_time, 1.0))
            sw[k] = p
        tgt["r"] = sw
        on_ground_r = t >= land_time
        p_ref = _quintic(x_end[:3], v_end[:3], p_goal, settle_time, t)
        R_ref = slerp(min(t / settle_time, 1.0) ** 2 * (3 - 2 * min(t / settle_time, 1.0)))
        q_ref = _quintic(x_end[6:], v_end[6:], q_goal, settle_time, t)

        def residual(xx):
            P = fk(xx)
            r = []
            for side in ("l", "r"):
                kp = keypts(P, side)
                for k in FOOT_KEYS:
                    r.append(3.0 * (kp[k] - tgt[side][k]))
                Rf, _ = P[f"{side}_foot"]
                if side == "l" or on_ground_r:
                    r.append(Rf[:, 2] - np.array([0, 0, 1.0]))
            r.append(1.0 * (xx[:3] - p_ref))
            Rx = Rotation.from_rotvec(xx[3:6])
            r.append(1.0 * (Rx * R_ref.inv()).as_rotvec())
            r.append(0.5 * upper * (xx[6:] - q_ref))
            r.append(0.1 * (xx - x_prev))
            return np.concatenate(r)

        sol = least_squares(residual, np.clip(x, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), max_nfev=80, xtol=1e-6, ftol=1e-6)
        x = sol.x
        x_prev = x.copy()
        frames.append(x)
        P = fk(x)
        e = [np.linalg.norm(keypts(P, s)[k] - tgt[s][k]) for s in ("l", "r") for k in FOOT_KEYS]
        kp_err.append(math.sqrt(np.mean(np.square(e))))

    F = np.stack(frames)
    quats = np.stack([_quat_wxyz(Rotation.from_rotvec(r).as_matrix()) for r in F[:, 3:6]])
    q_all = np.concatenate([res.root_quat, quats])
    for i in range(len(res.root_quat), len(q_all)):
        if np.dot(q_all[i], q_all[i - 1]) < 0:
            q_all[i] = -q_all[i]
    return RetargetResult(
        time=np.concatenate([res.time, res.time[-1] + dt * np.arange(1, T + 1)]),
        root_pos=np.concatenate([res.root_pos, F[:, :3]]), root_quat=q_all, joint_names=names,
        joint_pos=np.concatenate([res.joint_pos, F[:, 6:]]), contact_time=res.contact_time, tee_pos=res.tee_pos,
        keypoint_err=np.concatenate([res.keypoint_err, kp_err]), bat_err=np.concatenate([res.bat_err, np.zeros(T)]),
        grip_gap=np.concatenate([res.grip_gap, np.zeros(T)]), source=res.source + f"+follow_through{duration:.2f}s",
    )
