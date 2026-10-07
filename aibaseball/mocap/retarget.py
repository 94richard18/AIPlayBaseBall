"""Retarget an OBP swing onto AIB-1 (per-frame IK with the robot's own kinematics).

Free variables per frame: pelvis position, pelvis rotation (rotation vector), 29 body joints.
Fingers stay in the grip pose. Targets (scaled by robot/athlete height):
  joint centres (hips, knees, ankles, shoulders, elbows, wrists), toes/heels, head/neck,
  bat knob + barrel end (high weight), top-hand loop closure on the handle (high weight),
  feet flat while they are on the ground, temporal smoothness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..physics.bat_model import BatSpec
from ..robot.humanoid import R_PALM_RIGHT, build_humanoid, palm_grip_frame
from ..robot.kinematics import Robot
from ..robot.stance import GRIP_FINGERS
from .obp import SwingTrial

ROBOT_HEIGHT = 1.85

# robot keypoint: (link, local offset) -> human landmark, weight
KEYPOINTS = {
    "l_hip": ("l_hip_roll_link", (0, 0, 0), 1.0), "r_hip": ("r_hip_roll_link", (0, 0, 0), 1.0),
    "l_knee": ("l_shank", (0, 0, 0), 1.0), "r_knee": ("r_shank", (0, 0, 0), 1.0),
    "l_ankle": ("l_ankle_pitch_link", (0, 0, 0), 1.5), "r_ankle": ("r_ankle_pitch_link", (0, 0, 0), 1.5),
    "l_toe": ("l_foot", (0.16, 0, -0.06), 1.0), "r_toe": ("r_foot", (0.16, 0, -0.06), 1.0),
    "l_heel": ("l_foot", (-0.08, 0, -0.06), 1.0), "r_heel": ("r_foot", (-0.08, 0, -0.06), 1.0),
    "l_shoulder": ("l_shoulder_pitch_link", (0, 0, 0), 1.0), "r_shoulder": ("r_shoulder_pitch_link", (0, 0, 0), 1.0),
    "l_elbow": ("l_forearm", (0, 0, 0), 0.7), "r_elbow": ("r_forearm", (0, 0, 0), 0.7),
    "l_wrist": ("l_wrist_pitch_link", (0, 0, 0), 0.5), "r_wrist": ("r_wrist_pitch_link", (0, 0, 0), 0.5),
    "head": ("torso", (0, 0, 0.62), 0.7), "neck": ("torso", (0, 0, 0.47), 0.7),
}
# extra hand-orientation keypoints when there is no bat (pitching): wrist markers on the forearm, FIN on the hand dorsum
_DORSUM = (R_PALM_RIGHT @ np.array([0.08, 0.0, 0.02])).tolist()
HAND_KEYPOINTS = {
    "r_wrist_radial": ("r_forearm_distal", (0.028, 0, -0.14), 1.0), "r_wrist_ulnar": ("r_forearm_distal", (-0.028, 0, -0.14), 1.0),
    "l_wrist_radial": ("l_forearm_distal", (0.028, 0, -0.14), 0.7), "l_wrist_ulnar": ("l_forearm_distal", (-0.028, 0, -0.14), 0.7),
    "r_hand": ("r_hand", tuple(_DORSUM), 1.5), "l_hand": ("l_hand", (_DORSUM[0], -_DORSUM[1], _DORSUM[2]), 0.7),
}
BAT_WEIGHT = 3.0
GRIP_WEIGHT = 5.0


@dataclass
class RetargetResult:
    time: np.ndarray  # (T,)
    root_pos: np.ndarray  # (T,3) pelvis
    root_quat: np.ndarray  # (T,4) wxyz
    joint_names: list[str]
    joint_pos: np.ndarray  # (T,J) body joints
    contact_time: float
    tee_pos: np.ndarray  # (3,) ball position = reference sweet spot at contact
    keypoint_err: np.ndarray  # (T,) RMS keypoint error [m]
    bat_err: np.ndarray  # (T,) bat endpoint error [m]
    grip_gap: np.ndarray  # (T,) top-hand loop-closure gap [m]
    source: str


def _quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def retarget(trial: SwingTrial, t_before: float = 1.2, t_after: float = 0.45, rate: float = 120.0,
             robot: Robot | None = None, bat: BatSpec | None = None, verbose: bool = True,
             finger_pose: dict | None = None, ball_local: np.ndarray | None = None) -> RetargetResult:
    """Batting (robot with bat) or pitching (no bat: hand keypoints; `ball_local` = ball centre in r_hand)."""
    bat = bat or BatSpec()
    robot = robot or build_humanoid(with_bat=True, bat=bat)
    has_bat = "bat" in robot.links
    keypoints = dict(KEYPOINTS) if has_bat else dict(KEYPOINTS) | HAND_KEYPOINTS
    scale = ROBOT_HEIGHT / trial.height
    body = [j for j in robot.actuated if not any(k in j.name for k in GRIP_FINGERS)]
    names = [j.name for j in body]
    lo = np.array([j.lower for j in body])
    hi = np.array([j.upper for j in body])
    fingers = finger_pose if finger_pose is not None else {f"{s}_{k}": v for s in ("l", "r") for k, v in GRIP_FINGERS.items()}
    gp_r, _ = palm_grip_frame("r")

    # time window and resampling
    t_src = np.arange(trial.T) / trial.rate
    t_c = trial.contact / trial.rate
    t0, t1 = max(0.0, t_c - t_before), min(t_src[-1], t_c + t_after)
    times = np.arange(t0, t1, 1.0 / rate)
    lm = {k: np.stack([np.interp(times, t_src, v[:, i]) for i in range(3)], -1) for k, v in trial.lm.items()}
    for k in lm:  # scale positions about the lab origin (home plate); directions unchanged
        if k not in ("pelvis_fwd", "pelvis_left"):
            lm[k] = lm[k] * scale
    # ground: put the lowest heel/toe height at the robot's sole height
    foot_min = min(lm[k][:, 2].min() for k in ("l_toe", "r_toe", "l_heel", "r_heel"))
    for k in lm:
        if k not in ("pelvis_fwd", "pelvis_left"):
            lm[k] = lm[k] - np.array([0, 0, foot_min - 0.02])

    def unpack(x):
        p = x[:3]
        R = Rotation.from_rotvec(x[3:6]).as_matrix()
        q = dict(zip(names, x[6:])) | fingers
        return p, R, q

    def residual(x, f, x_prev):
        p, R, q = unpack(x)
        P = robot.fk(q, root_pos=p, root_rot=R)
        res = []
        for key, (link, off, w) in keypoints.items():
            Rl, pl = P[link]
            res.append(w * (pl + Rl @ np.asarray(off) - lm[key][f]))
        if has_bat:
            Rb, pb = P["bat"]
            res.append(BAT_WEIGHT * (pb - lm["bat_knob"][f]))
            res.append(BAT_WEIGHT * (pb + Rb[:, 0] * bat.length - lm["bat_tip"][f]))
            Rr, pr = P["r_hand"]
            res.append(GRIP_WEIGHT * (pr + Rr @ gp_r - (pb + Rb[:, 0] * bat.top_hand_x)))
        # pelvis heading consistent with the athlete's pelvis
        res.append(0.5 * (R[:, 0] - lm["pelvis_fwd"][f]))
        for s in ("l", "r"):  # feet flat when near the ground
            Rf, _ = P[f"{s}_foot"]
            on_ground = float(min(lm[f"{s}_toe"][f, 2], lm[f"{s}_heel"][f, 2]) < 0.10)
            res.append(1.0 * on_ground * Rf[:, 2] - on_ground * np.array([0, 0, 1.0]))
        if x_prev is not None:
            res.append(0.1 * (x - x_prev))
        return np.concatenate(res)

    # initial guess: batting stance solution, heading from the athlete's pelvis
    from ..robot.stance import solve_stance

    init_q = solve_stance(robot, bat).joint_pos if has_bat else {n: 0.0 for n in names}
    fwd = lm["pelvis_fwd"][0]
    yaw = math.atan2(fwd[1], fwd[0])
    x = np.concatenate([lm["pelvis"][0], [0, 0, yaw], [init_q[n] for n in names]])
    x[6:] = np.clip(x[6:], lo + 1e-4, hi - 1e-4)
    big = np.full(6, np.inf)
    lb, ub = np.concatenate([-big, lo]), np.concatenate([big, hi])

    T = len(times)
    out = np.zeros((T, len(x)))
    kp_err, bat_err, grip = np.zeros(T), np.zeros(T), np.zeros(T)
    x_prev = None
    for f in range(T):
        sol = least_squares(residual, x, args=(f, x_prev), bounds=(lb, ub), max_nfev=200 if f == 0 else 60,
                            xtol=1e-6, ftol=1e-6)
        x = sol.x
        out[f] = x
        x_prev = x.copy()
        p, R, q = unpack(x)
        P = robot.fk(q, root_pos=p, root_rot=R)
        errs = [np.linalg.norm(P[link][1] + P[link][0] @ np.asarray(off) - lm[key][f]) for key, (link, off, _) in keypoints.items()]
        kp_err[f] = math.sqrt(np.mean(np.square(errs)))
        if has_bat:
            Rb, pb = P["bat"]
            bat_err[f] = max(np.linalg.norm(pb - lm["bat_knob"][f]), np.linalg.norm(pb + Rb[:, 0] * bat.length - lm["bat_tip"][f]))
            Rr, pr = P["r_hand"]
            grip[f] = np.linalg.norm(pr + Rr @ gp_r - (pb + Rb[:, 0] * bat.top_hand_x))
        if verbose and f % 20 == 0:
            print(f"  frame {f:4d}/{T}: keypoint RMS {kp_err[f]*100:5.1f} cm, bat {bat_err[f]*100:5.1f} cm, "
                  f"grip gap {grip[f]*1000:5.1f} mm", flush=True)

    rotvec = out[:, 3:6]
    quats = np.stack([_quat_wxyz(Rotation.from_rotvec(r).as_matrix()) for r in rotvec])
    for i in range(1, T):  # keep quaternion sign continuous
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    c_idx = int(round((t_c - t0) * rate))
    # tee = robot sweet spot at contact (consistent with the robot's own kinematics)
    p, R, q = unpack(out[c_idx])
    if has_bat:
        Rb, pb = robot.fk(q, root_pos=p, root_rot=R)["bat"]
        tee = pb + Rb[:, 0] * bat.sweet_spot_x
    else:  # pitching: ball position at release
        Rh, ph = robot.fk(q, root_pos=p, root_rot=R)["r_hand"]
        tee = ph + Rh @ np.asarray(ball_local if ball_local is not None else np.zeros(3))
    return RetargetResult(
        time=times - t0, root_pos=out[:, :3], root_quat=quats, joint_names=names, joint_pos=out[:, 6:],
        contact_time=t_c - t0, tee_pos=tee, keypoint_err=kp_err, bat_err=bat_err, grip_gap=grip, source=trial.name,
    )


def save(res: RetargetResult, path: str):
    np.savez(path, time=res.time, root_pos=res.root_pos, root_quat=res.root_quat,
             joint_names=np.array(res.joint_names), joint_pos=res.joint_pos, contact_time=res.contact_time,
             tee_pos=res.tee_pos, keypoint_err=res.keypoint_err, bat_err=res.bat_err, grip_gap=res.grip_gap,
             source=res.source)
