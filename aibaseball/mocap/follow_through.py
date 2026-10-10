"""Synthesised pitching follow-through: the OBP captures stop ~0.14 s after release, mid-motion.

The appended segment (IK per frame with the robot's own kinematics):
  * lead (left) foot stays planted where the capture ends and its knee straightens into a post;
  * the trunk folds over the lead leg while the trailing (right) leg swings up and back, then that foot comes down
    beside the lead foot (quintic paths);
  * the pelvis decelerates from its captured velocity over the lead foot, then settles between the feet;
  * arms decelerate from their captured velocity to the set position (first frame of the clip).
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


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def append_follow_through(res: RetargetResult, robot: Robot, fingers: dict, duration: float = 1.1, rate: float = 120.0,
                          balance_time: float = 0.25, kick_time: float = 0.25, land_time: float = 0.6,
                          settle_time: float = 0.85, stance_width: float = 0.40, back_foot_x: float = 0.05,
                          kick_back: float = 0.75, kick_height: float = 0.40, kick_width: float = 0.25, lift: float = 0.06,
                          lead_knee: float = 0.25, lead_knee_end: float = 0.35, balance_pitch_deg: float = 35.0,
                          balance_waist: float = 0.35, settle_pitch_deg: float = 20.0, settle_waist: float = 0.25
                          ) -> RetargetResult:
    """Pro finish over a straight lead leg (the user's reference videos: a pro pitcher's "lock that front leg out ... be
    over that front leg as much as possible", and a 95 mph skeleton): right after the release the lead knee straightens
    (lead_knee rad) into a post, the trunk folds over the lead leg (pelvis pitch + waist flexion) and the back leg swings
    up and back (kick_back behind the lead foot, kick_height up) as a counterweight; then (land_time) the back foot comes
    down beside the lead foot (back_foot_x ahead, stance_width to the throwing-arm side) and the body rises into a
    fielding stance. Times are from the end of the capture (~0.14 s after the release).
    (Earlier: capture step, the back foot straight through 0.30 m ahead with the lead knee bent ~65 deg: the robot sank on
    the bent knee and fell ~1.1 s after the release, the back foot left 0.7-0.9 m behind. Before that: back foot beside /
    behind the lead foot, upright -> fell ~1 s after the release; crouched -> sooner, trunk overshot to 110 deg.)
    """
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
    lead_c = (lead["toe"] + lead["heel"]) / 2
    half = (KEYPOINTS["r_toe"][1][0] - KEYPOINTS["r_heel"][1][0]) / 2

    def foot_at(c):
        f = {"toe": c + [half, 0, 0], "heel": c - [half, 0, 0]}
        f["ankle"] = f["heel"] + [-KEYPOINTS["r_heel"][1][0], 0, lead["ankle"][2] - ground]
        return f

    # back foot: up and back as a counterweight, then down beside the lead foot (throwing-arm side), toes to the plate
    kick = foot_at(lead_c + np.array([-kick_back, -kick_width, kick_height]))
    land_c = lead_c + np.array([back_foot_x, -stance_width, 0.0])
    land_c[2] = ground
    land = foot_at(land_c)

    # pelvis over the lead foot while balancing on it, then between the feet; pitched forward (trunk folds over)
    feet_mid = (lead_c + land_c) / 2
    p_bal = np.array([lead_c[0] - 0.15, lead_c[1] - 0.08, x_end[2] + 0.05])
    p_goal = np.array([feet_mid[0] - 0.05, feet_mid[1], x_end[2] + 0.05])
    R_end = Rotation.from_rotvec(x_end[3:6])
    R_bal = Rotation.from_euler("y", math.radians(balance_pitch_deg))  # facing the plate, pitched forward
    R_goal = Rotation.from_euler("y", math.radians(settle_pitch_deg))
    slerp1 = Slerp([0.0, 1.0], Rotation.concatenate([R_end, R_bal]))
    slerp2 = Slerp([0.0, 1.0], Rotation.concatenate([R_bal, R_goal]))
    upper = np.array([not any(k in n for k in LEGS) for n in names])
    q_goal = np.where(upper, res.joint_pos[0], x_end[6:])
    i_wp = names.index("waist_pitch")  # positive = forward flexion
    i_lk = names.index("l_knee")

    def phased(v0, vel0, v_bal, v_fin, t):
        """quintic to v_bal by balance_time, held, then to v_fin over land_time -> settle_time"""
        if t < land_time:
            return _quintic(v0, vel0, v_bal, balance_time, t)
        return _quintic(v_bal, 0.0 * v_bal, v_fin, settle_time - land_time, t - land_time)

    T = int(round(duration * rate))
    frames, kp_err = [], []
    x = x_end.copy()
    x_prev = x_end.copy()
    big = np.full(6, np.inf)
    lb, ub = np.concatenate([-big, lo]), np.concatenate([big, hi])
    w_root = np.array([3.0, 3.0, 0.2])  # pelvis height follows from the straight lead leg
    for f in range(1, T + 1):
        t = f * dt
        tgt = {"l": lead}
        sw = {}
        for k in FOOT_KEYS:
            if t <= kick_time:
                v0 = (trail1[k] - trail0[k]) / (2 * dt)
                p = _quintic(trail1[k], v0, kick[k], kick_time, t)
            else:
                s2 = (t - kick_time) / (land_time - kick_time)
                p = _quintic(kick[k], 0.0 * kick[k], land[k], land_time - kick_time, t - kick_time)
                p[2] += lift * math.sin(math.pi * min(s2, 1.0))
            sw[k] = p
        tgt["r"] = sw
        on_ground_r = t >= land_time
        w_foot = {"l": 3.0, "r": 3.0 if on_ground_r else 1.0}  # a swing foot out of reach must not drag the pelvis
        p_ref = phased(x_end[:3], v_end[:3], p_bal, p_goal, t)
        R_ref = (slerp1(_smooth(t / balance_time)) if t < land_time
                 else slerp2(_smooth((t - land_time) / (settle_time - land_time))))
        q_ref = _quintic(x_end[6:], v_end[6:], q_goal, settle_time, t)
        q_ref[i_wp] = phased(x_end[6 + i_wp], v_end[6 + i_wp], balance_waist, settle_waist, t)
        knee_ref = phased(x_end[6 + i_lk], v_end[6 + i_lk], lead_knee, lead_knee_end, t)

        def residual(xx):
            P = fk(xx)
            r = []
            for side in ("l", "r"):
                kp = keypts(P, side)
                for k in FOOT_KEYS:
                    r.append(w_foot[side] * (kp[k] - tgt[side][k]))
                Rf, _ = P[f"{side}_foot"]
                if side == "l" or on_ground_r:
                    r.append(Rf[:, 2] - np.array([0, 0, 1.0]))
            r.append(w_root * (xx[:3] - p_ref))
            Rx = Rotation.from_rotvec(xx[3:6])
            r.append(1.0 * (Rx * R_ref.inv()).as_rotvec())
            r.append(0.5 * upper * (xx[6:] - q_ref))
            r.append([2.0 * (xx[6 + i_lk] - knee_ref)])
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
