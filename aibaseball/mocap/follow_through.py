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
                          balance_time: float = 0.4, kick_time: float = 0.25, land_time: float = 0.6,
                          settle_time: float = 1.0, stance_width: float = 0.40, back_foot_x: float = 0.05,
                          finish_yaw_deg: float = 70.0, shift_time: float = 0.5, kick_back: float = 0.55,
                          kick_height: float = 0.30,
                          kick_width: float = 0.40, lift: float = 0.06, lead_knee: float = 0.35, lead_knee_end: float = 0.35,
                          pelvis_rise: float = 0.10, com_shift: float = 0.08, balance_pitch_deg: float = 50.0,
                          balance_waist: float = 0.35, settle_pitch_deg: float = 20.0, settle_waist: float = 0.25
                          ) -> RetargetResult:
    """Pro finish: pivot on a straight lead leg, the back foot comes around and lands to take the rest of the momentum.

    From the user's reference videos (a pro pitcher: "lock that front leg out ... be over that front leg"; 83 vs 95 mph
    skeletons) and the user's own view: after the release the leftover energy is dissipated by rotating about the lead
    leg (the pivot) and then planting the back foot. Times are from the end of the capture (~0.14 s after the release).
      * lead foot planted; its knee straightens slowly (balance_time) to lead_knee, the pelvis rising at most
        pelvis_rise: a fast straightening launched the robot (COM up at ~1 m/s, lead foot off the ground ~0.2 s);
      * the pelvis keeps turning in the throwing direction (heading carried from the capture, with its rate, to
        finish_yaw_deg) instead of being turned back to face the plate against the rotation;
      * the pelvis stays over the lead foot (COM over the foot, not behind the heel, which needed forward speed the robot
        does not carry); the trunk folds over the lead leg (pelvis pitch + waist flexion);
      * the back leg swings up and around the outside (kick_back behind, kick_width out, kick_height up) and lands at
        land_time beside / ahead of the lead foot, on the stance line turned by finish_yaw_deg, toes along the heading.
    (Earlier: v7 straight-leg finish: knee straightened in 0.2 s, pelvis +21 cm, COM behind the heel, pelvis turned back
    to face the plate -> lead foot lifted at +0.35 s, 1/32 still up. v6 capture step: lead knee bent ~65 deg, sank. Older:
    upright -> fell ~1 s after the release; crouched -> sooner.)
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

    def heading(rotvec):
        M = Rotation.from_rotvec(rotvec).as_matrix()
        return math.atan2(M[1, 0], M[0, 0])

    P1, P0 = fk(X[-1]), fk(X[-3])
    lead = keypts(P1, "l")
    trail1, trail0 = keypts(P1, "r"), keypts(P0, "r")
    ground = min(lead["toe"][2], lead["heel"][2])
    lead_c = (lead["toe"] + lead["heel"]) / 2
    half = (KEYPOINTS["r_toe"][1][0] - KEYPOINTS["r_heel"][1][0]) / 2
    psi_fin = math.radians(finish_yaw_deg)
    Rz_fin = Rotation.from_euler("z", psi_fin).as_matrix()

    def foot_at(c, yaw=0.0):
        u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        f = {"toe": c + half * u, "heel": c - half * u}
        f["ankle"] = f["heel"] - KEYPOINTS["r_heel"][1][0] * u + np.array([0.0, 0.0, lead["ankle"][2] - ground])
        return f

    # back foot: up, back and out (around the outside), then down on the turned stance line
    kick = foot_at(lead_c + np.array([-kick_back, -kick_width, kick_height]))
    land_c = lead_c + Rz_fin @ np.array([back_foot_x, -stance_width, 0.0])
    land_c[2] = ground
    land = foot_at(land_c, psi_fin)

    # pelvis over the lead foot while pivoting on it, then between the feet
    feet_mid = (lead_c + land_c) / 2
    p_bal = np.array([lead_c[0] + com_shift, lead_c[1] - 0.05, x_end[2] + pelvis_rise])
    p_goal = np.array([feet_mid[0], feet_mid[1], x_end[2] + pelvis_rise])
    v_pel = v_end[:3].copy()
    v_pel[2] = min(v_pel[2], 0.0)
    # orientation = heading (carried on, with its captured rate) x tilt (pitched forward over the lead leg)
    psi0 = heading(x_end[3:6])
    dpsi = (psi0 - heading(X[-3][3:6]) + math.pi) % (2 * math.pi) - math.pi
    psi_rate = dpsi / (2 * dt)
    tilt_end = Rotation.from_euler("z", -psi0) * Rotation.from_rotvec(x_end[3:6])
    R_bal = Rotation.from_euler("y", math.radians(balance_pitch_deg))
    R_goal = Rotation.from_euler("y", math.radians(settle_pitch_deg))
    slerp1 = Slerp([0.0, 1.0], Rotation.concatenate([tilt_end, R_bal]))
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
    w_root = np.array([3.0, 3.0, 1.0])  # pelvis height held (rise <= pelvis_rise): no launch off the lead leg
    for f in range(1, T + 1):
        t = f * dt
        tgt = {"l": lead}
        sw = {}
        for k in FOOT_KEYS:
            if t <= kick_time:
                v0 = (trail1[k] - trail0[k]) / (2 * dt)
                v0[2] = min(v0[2], 0.0)  # its captured upward speed overshot the kick (+0.27 m), lifting the COM
                p = _quintic(trail1[k], v0, kick[k], kick_time, t)
            else:
                s2 = (t - kick_time) / (land_time - kick_time)
                p = _quintic(kick[k], 0.0 * kick[k], land[k], land_time - kick_time, t - kick_time)
                p[2] += lift * math.sin(math.pi * min(s2, 1.0))
            sw[k] = p
        tgt["r"] = sw
        on_ground_r = t >= land_time
        w_foot = {"l": 3.0, "r": 3.0 if on_ground_r else 0.3}  # a swing foot out of reach must not drag the pelvis (up)
        # over the lead foot quickly (shift_time); height without the captured upward speed (it carried the pelvis up
        # 15 cm past the target)
        if t < land_time:
            p_ref = _quintic(x_end[:3], v_pel, p_bal, np.array([shift_time, shift_time, balance_time]), t)
        else:
            p_ref = _quintic(p_bal, 0.0 * p_bal, p_goal, settle_time - land_time, t - land_time)
        psi = float(_quintic(psi0, psi_rate, psi_fin, land_time, t))
        tilt = (slerp1(_smooth(t / balance_time)) if t < land_time
                else slerp2(_smooth((t - land_time) / (settle_time - land_time))))
        R_ref = Rotation.from_euler("z", psi) * tilt
        q_ref = _quintic(x_end[6:], v_end[6:], q_goal, settle_time, t)
        q_ref[i_wp] = phased(x_end[6 + i_wp], v_end[6 + i_wp], balance_waist, settle_waist, t)
        knee_ref = phased(x_end[6 + i_lk], 0.0, lead_knee, lead_knee_end, t)  # (its captured speed overshot to 0 deg)

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
            r.append([0.5 * (xx[6 + i_lk] - knee_ref)])
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
