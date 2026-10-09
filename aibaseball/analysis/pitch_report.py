"""Pitching analysis shared by the athlete reference and the simulated robot (pure numpy, no simulator).

Every function takes per-frame arrays of one pitch so the same metric is computed the same way for both:
  * mechanics at foot strike / release (hip-shoulder separation, arm launch position, pelvis opening, pivot knee,
    trunk tilt),
  * kinetic chain: peak-speed timing of pelvis, trunk, shoulder internal rotation, elbow extension, hand,
  * energy flow: mechanical energy of segment groups (legs + pelvis, trunk, throwing arm, ball), the rate of energy
    flowing into the throwing arm (+ ball), ground reaction force from the COM (braking, vertical, in body weights).
"""

from __future__ import annotations

import math
import os

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

G = 9.81
BALL_MASS = 0.145

SEGMENTS = ["pelvis rotation", "trunk rotation", "shoulder int. rotation", "elbow extension", "hand speed"]
SEG_UNITS = ["deg/s", "deg/s", "deg/s", "deg/s", "m/s"]
GROUPS = ["legs+pelvis", "trunk", "throwing arm", "glove arm", "ball"]


def group_of(name: str) -> str:
    if name == "pelvis" or any(k in name for k in ("hip", "thigh", "shank", "ankle", "foot", "knee")):
        return "legs+pelvis"
    if "waist" in name or name == "torso":
        return "trunk"
    return "throwing arm" if name.startswith("r_") else "glove arm"


# ----------------------------------------------------------------------------- kinematic helpers
def yaw_deg(R):
    f = R[..., :, 0]
    return np.degrees(np.arctan2(f[..., 1], f[..., 0]))


def ang_vel_from_rot(R, dt):
    """World angular velocity (T,3) from rotation matrices (T,3,3) by central differences."""
    r = Rotation.from_matrix(R)
    w = np.zeros((len(R), 3))
    w[1:-1] = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * dt)
    w[0], w[-1] = w[1], w[-2]
    return w


def smooth(x, dt, window_s=0.04, deriv=0):
    win = max(5, int(round(window_s / dt)) | 1)  # odd
    n = len(x) if len(x) % 2 else len(x) - 1
    win = min(win, n)
    return savgol_filter(x, win, 3, deriv=deriv, delta=dt, axis=0)


# ----------------------------------------------------------------------------- mechanics
def mechanics_series(R_pel, R_tor, p_sh, p_hand, knee_r_rad):
    sep = (yaw_deg(R_pel) - yaw_deg(R_tor) + 180) % 360 - 180
    up = R_tor[:, :, 2]
    return {
        "pelvis heading vs target (deg, 0 = facing the plate)": (yaw_deg(R_pel) + 180) % 360 - 180,
        "hip-shoulder separation (deg)": sep,
        "throwing hand above shoulder (m)": p_hand[:, 2] - p_sh[:, 2],
        "throwing hand behind shoulder (m)": -(p_hand[:, 0] - p_sh[:, 0]),
        "trunk tilt from vertical (deg)": np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))),
        "pivot (back) knee flexion (deg)": np.degrees(knee_r_rad),
    }


def chain_series(R_pel, R_tor, qd_sh_yaw, qd_elbow, hand_v, dt):
    s = np.stack([np.abs(ang_vel_from_rot(R_pel, dt)[:, 2]) * 57.3, np.abs(ang_vel_from_rot(R_tor, dt)[:, 2]) * 57.3,
                  np.abs(qd_sh_yaw) * 57.3, np.abs(qd_elbow) * 57.3, hand_v], 1)
    return smooth(s, dt, 0.02)


def chain_peaks(t, s, t_rel, lo=-0.45, hi=0.15):
    m = (t >= t_rel + lo) & (t <= t_rel + hi)
    out = []
    for j in range(s.shape[1]):
        i = int(np.argmax(np.where(m, s[:, j], -1)))
        out.append((t[i] - t_rel, s[i, j]))
    return out


# ----------------------------------------------------------------------------- energy
def segment_energy(m, I_body, R_body, p_com, v_com, w_world):
    """Kinetic + potential energy per body (T,B). I_body (B,3,3) about the COM in the body frame, R_body (T,B,3,3)."""
    ke_t = 0.5 * m[None, :] * (v_com ** 2).sum(-1)
    w_b = np.einsum("tbji,tbj->tbi", R_body, w_world)  # R^T w
    ke_r = 0.5 * np.einsum("tbi,bij,tbj->tb", w_b, I_body, w_b)
    pe = m[None, :] * G * p_com[..., 2]
    return ke_t + ke_r, pe


def energy_flow(t, names, E_kin, E_pot, ball_e, released, dt, internal_power=None):
    """Group energies (T, groups) and the energy rate of the throwing arm + held ball (W).

    `internal_power` (T,): muscle power of joints inside the arm (elbow, wrist, fingers); subtracting it leaves the
    energy that flows in through the shoulder (from the trunk).
    """
    groups = np.array([group_of(n) for n in names])
    E = np.zeros((len(t), len(GROUPS)))
    for j, g in enumerate(GROUPS[:-1]):
        sel = groups == g
        E[:, j] = E_kin[:, sel].sum(1) + E_pot[:, sel].sum(1)
    E[:, -1] = ball_e
    arm_ball = E[:, 2] + np.where(released, 0.0, ball_e)
    rate = smooth(arm_ball, dt, 0.03, deriv=1)
    flow = rate - (internal_power if internal_power is not None else 0.0)
    return E, rate, flow


def angular_momentum(m, I_body, R_body, p_com, v_com, w_world):
    """Whole-body angular momentum about the COM (T,3)."""
    M = m.sum()
    C = (p_com * m[None, :, None]).sum(1) / M
    Cv = (v_com * m[None, :, None]).sum(1) / M
    lin = np.cross(p_com - C[:, None], v_com - Cv[:, None]) * m[None, :, None]
    Iw = np.einsum("tbij,bjk,tblk,tbl->tbi", R_body, I_body, R_body, w_world)
    return (lin + Iw).sum(1)


def after_release_series(t, com, com_v, feet_link, feet_on, ground_fn, pelvis_z, head_z, trunk_tilt, L, lead_knee,
                         lead_hip):
    """Per-frame quantities for the after-release timeline (feet: 0 = lead / left, 1 = back / right)."""
    sole = feet_link[..., 2] - 0.08 - ground_fn(feet_link[..., 0])
    toe_x = feet_link[:, 0, 0] + 0.13
    h = np.clip(com[:, 2] - ground_fn(com[:, 0]), 0.3, None)
    cp_x = com[:, 0] + com_v[:, 0] * np.sqrt(h / G)
    return {
        "lead planted": feet_on[:, 0].astype(float), "lead sole (cm)": sole[:, 0] * 100,
        "back planted": feet_on[:, 1].astype(float), "back sole (cm)": sole[:, 1] * 100,
        "COM ahead of lead foot (m)": com[:, 0] - feet_link[:, 0, 0], "COM vx (m/s)": com_v[:, 0],
        "COM vz (m/s)": com_v[:, 2], "capture pt past lead toe (m)": cp_x - toe_x,
        "pelvis z (m)": pelvis_z, "head z (m)": head_z, "trunk tilt (deg)": trunk_tilt,
        "fwd ang. momentum (kg m2/s)": L[:, 1], "lead knee (deg)": np.degrees(lead_knee),
        "lead hip pitch (deg)": np.degrees(lead_hip),
    }


def grf_from_com(com_acc, mass):
    """Total ground reaction force (T,3) in body weights from the COM acceleration."""
    return (com_acc + np.array([0.0, 0.0, G])) / G


# ----------------------------------------------------------------------------- athlete reference
def reference_profile(cfg) -> dict:
    """Everything above for the reference motion (robot masses / inertias, the env's ref_yaw rotation)."""
    from ..mocap.centroidal import mound_z, reference_centroidal
    from ..robot.ball_grip import FINGER_JOINTS, BallGrip
    from ..robot.humanoid import build_humanoid

    d = np.load(cfg.motion_file, allow_pickle=True)
    t = d["time"]
    dt = float(t[1] - t[0])
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    grip = BallGrip.load(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "aib1_pitcher", "grip.json"))
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos
    Rz = Rotation.from_euler("z", math.radians(cfg.ref_yaw_deg))
    Rr = Rz * Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1))
    root = Rz.apply(d["root_pos"])
    links = list(robot.links.values())
    lnames = [lk.name for lk in links]
    T, B = len(t), len(links)
    Rb = np.zeros((T, B, 3, 3))
    pc = np.zeros((T, B, 3))
    pl = {}
    for f in range(T):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])) | fingers, root_pos=root[f], root_rot=Rr[f].as_matrix())
        for b, lk in enumerate(links):
            R, p = P[lk.name]
            Rb[f, b] = R
            pc[f, b] = p + R @ lk.com
        for key in ("pelvis", "torso", "r_shoulder_pitch_link", "r_hand"):
            pl.setdefault(key, np.zeros((T, 3)))[f] = P[key][1]
    m = np.array([lk.mass for lk in links])
    I = np.array([lk.inertia for lk in links])
    v = smooth(pc, dt, 0.04, deriv=1)
    w = np.stack([ang_vel_from_rot(Rb[:, b], dt) for b in range(B)], 1)
    Ek, Ep = segment_energy(m, I, Rb, pc, v, w)
    ball = Rz.apply(d["key_pos"][:, -1])
    ball_v = smooth(ball, dt, 0.03, deriv=1)
    t_rel = float(d["contact_time"])
    released = t >= t_rel
    ball_e = 0.5 * BALL_MASS * (ball_v ** 2).sum(-1) + BALL_MASS * G * ball[:, 2]
    E, rate, flow = energy_flow(t, lnames, Ek, Ep, ball_e, released, dt)
    com = (pc * m[None, :, None]).sum(1) / m.sum()
    com_acc = smooth(com, dt, 0.08, deriv=2)
    cen = reference_centroidal(cfg.motion_file, lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height, cfg.mound_slope,
                                                                   cfg.mound_slope_start_x))
    strike = float(t[np.argmax(cen["contact"][:, 0] & (t > 0.3))])
    i_pel, i_tor = lnames.index("pelvis"), lnames.index("torso")
    hand_v = np.linalg.norm(smooth(pl["r_hand"], dt, 0.03, deriv=1), axis=-1)
    qd = d["joint_vel"]
    return dict(
        t=t, dt=dt, release=t_rel, strike=strike, names=lnames, mass=m.sum(),
        mech=mechanics_series(Rb[:, i_pel], Rb[:, i_tor], pl["r_shoulder_pitch_link"], pl["r_hand"],
                              d["joint_pos"][:, names.index("r_knee")]),
        chain=chain_series(Rb[:, i_pel], Rb[:, i_tor], qd[:, names.index("r_shoulder_yaw")], qd[:, names.index("r_elbow")],
                           hand_v, dt),
        E=E, arm_rate=rate, arm_flow=flow, grf=grf_from_com(com_acc, m.sum()), ball_speed=np.linalg.norm(ball_v, axis=-1),
        after=after_release_series(
            t, com, smooth(com, dt, 0.04, deriv=1),
            np.stack([Rz.apply(d["key_pos"][:, 2]), Rz.apply(d["key_pos"][:, 3])], 1), cen["contact"],
            lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height, cfg.mound_slope, cfg.mound_slope_start_x),
            pl["pelvis"][:, 2], pl["torso"][:, 2] + Rb[:, i_tor, 2, 2] * 0.62,
            np.degrees(np.arccos(np.clip(Rb[:, i_tor, 2, 2], -1, 1))),
            angular_momentum(m, I, Rb, pc, v, w), d["joint_pos"][:, names.index("l_knee")],
            d["joint_pos"][:, names.index("l_hip_pitch")]),
    )


def at(t, x, when):
    return float(np.interp(when, t, x))
