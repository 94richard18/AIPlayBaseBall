"""Reference centre of mass and foot-contact schedule of a pitching motion, computed with AIB-1's own masses."""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

SOLE = [np.array([x, y, -0.08]) for x in (-0.09, 0.17) for y in (-0.055, 0.055)]  # foot box corners (ankle frame)


def mound_z(x, top: float, height: float, slope: float, x0: float):
    """Mound surface height at x (capture / env frame, +x toward the plate)."""
    if height <= 0:
        return np.zeros_like(x)
    return np.where(x < x0, top, np.maximum(top - (x - x0) * slope, top - height))


def reference_centroidal(motion_file: str, ground=lambda x: np.zeros_like(x), contact_gap: float = 0.025,
                         contact_speed: float = 1.0, touch_speed: float = 3.0, fill_s: float = 0.15) -> dict:
    """COM (T,3), COM velocity (T,3), feet on the ground (T,2: left, right) and planted-still feet (T,2).

    `contact`: the lowest sole corner is within `contact_gap` of the ground and slower than `touch_speed`; this
    includes the pivot foot dragging on its toe after the push-off (~1.5 m/s; it touches the mound until ~1.03 s,
    after the lead foot lands at 0.88 s) but not the lead foot skimming over the mound in its swing (4-6 m/s).
    `still`: in contact and the sole moves slower than `contact_speed` (no-slip applies only to these).
    Gaps shorter than `fill_s` (rolling heel-to-toe, IK jitter) are filled in both.
    """
    from ..robot.ball_grip import FINGER_JOINTS, BallGrip
    from ..robot.humanoid import build_humanoid

    d = np.load(motion_file, allow_pickle=True)
    t = d["time"]
    dt = float(t[1] - t[0])
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    grip = BallGrip.load(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "aib1_pitcher", "grip.json"))
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos
    links = list(robot.links.values())
    m = np.array([lk.mass for lk in links])
    R = Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1)).as_matrix()
    T = len(t)
    com = np.zeros((T, 3))
    sole = np.zeros((T, 2, 4, 3))
    for f in range(T):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])) | fingers, root_pos=d["root_pos"][f], root_rot=R[f])
        com[f] = sum(mi * (P[lk.name][1] + P[lk.name][0] @ lk.com) for mi, lk in zip(m, links)) / m.sum()
        for i, s in enumerate("lr"):
            Rf, pf = P[f"{s}_foot"]
            sole[f, i] = [pf + Rf @ c for c in SOLE]
    win = 9 if T > 20 else 5
    com_vel = savgol_filter(com, win, 3, deriv=1, delta=dt, axis=0)
    sole_vel = np.gradient(sole, dt, axis=0)
    gap = sole[..., 2].min(-1) - ground(sole[..., 0].mean(-1))
    speed = np.linalg.norm(sole_vel, axis=-1).max(-1)
    contact = (gap < contact_gap) & (speed < touch_speed)
    still = contact & (speed < contact_speed)
    k = int(round(fill_s / dt))
    for arr in (contact, still):
        for i in range(2):  # closing: fill short gaps between spans
            c = arr[:, i]
            on = np.nonzero(c)[0]
            for a, b in zip(on[:-1], on[1:]):
                if 1 < b - a <= k:
                    c[a:b] = True
    return dict(time=t, com=com, com_vel=com_vel, contact=contact, still=still)
