"""Load OBP hitting C3D trials and derive joint-centre landmarks + bat pose.

OBP lab frame: +x toward the mound, +y toward the right-handed batter's box, +z up (metres).
This matches the simulation world frame (center field = +X, right-handed batter on +Y).
Marker set: Plug-in Gait style + medial knee/ankle/elbow markers; bat: Marker1 = knob end,
Marker2/Marker3 = barrel end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import c3d
import numpy as np
from scipy.signal import butter, filtfilt

INCH = 0.0254


@dataclass
class SwingTrial:
    name: str
    rate: float  # Hz
    lm: dict[str, np.ndarray]  # landmark -> (T, 3)
    height: float  # athlete height [m]
    side: str
    exit_velo_mph: float
    contact: int  # frame index of (estimated) ball contact
    bat_speed_mph: np.ndarray  # (T,) sweet-spot speed

    @property
    def T(self) -> int:
        return next(iter(self.lm.values())).shape[0]


def _read_c3d(path):
    with open(path, "rb") as f:
        r = c3d.Reader(f)
        labels = [lbl.strip() for lbl in r.point_labels]
        rate = float(r.point_rate)
        frames = [pts[:, :3] for _, pts, _ in r.read_frames()]
    P = np.stack(frames)  # (T, N, 3)
    scale = 1.0 if np.nanmedian(np.abs(P)) < 10 else 0.001  # m vs mm
    return {lbl: P[:, i] * scale for i, lbl in enumerate(labels)}, rate


def _lowpass(x, rate, cutoff):
    b, a = butter(4, cutoff / (0.5 * rate))
    return filtfilt(b, a, x, axis=0)


def _frame(origin, x_hint, z_hint):
    """Orthonormal frame (T,3,3) with columns x,y,z; x from x_hint, z ~ z_hint."""
    x = x_hint / np.linalg.norm(x_hint, axis=-1, keepdims=True)
    y = np.cross(z_hint, x)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def load_trial(path: str, body_cutoff: float = 12.0, bat_cutoff: float = 20.0, sweet_from_tip: float = 6 * INCH) -> SwingTrial:
    m, rate = _read_c3d(path)
    base = os.path.basename(path)[:-4]
    parts = base.split("_")
    height = float(parts[2]) * INCH
    pitching = "Marker1" not in m  # pitching trials: USER_SESSION_HEIGHT_WEIGHT_PITCH_TYPE_SPEED
    side = "R" if pitching else parts[4]
    ev = float(parts[-1]) / 10.0

    def f(name, cutoff=body_cutoff):
        return _lowpass(m[name], rate, cutoff)

    lasi, rasi, lpsi, rpsi = f("LASI"), f("RASI"), f("LPSI"), f("RPSI")
    asis_mid = 0.5 * (lasi + rasi)
    psis_mid = 0.5 * (lpsi + rpsi)
    width = np.linalg.norm(lasi - rasi, axis=-1, keepdims=True)
    # pelvis frame: x forward (PSIS -> ASIS), y left (RASI -> LASI)
    R = _frame(asis_mid, asis_mid - psis_mid, np.cross(asis_mid - psis_mid, lasi - rasi))
    lm: dict[str, np.ndarray] = {}
    # Bell et al. (1990) hip joint centre: (-0.19 W, +-0.36 W, -0.30 W) in the pelvis frame
    for s, sy in (("l", 1.0), ("r", -1.0)):
        off = np.concatenate([-0.19 * width, sy * 0.36 * width, -0.30 * width], -1)
        lm[f"{s}_hip"] = asis_mid + np.einsum("tij,tj->ti", R, off)
    lm["pelvis"] = 0.5 * (asis_mid + psis_mid)
    lm["pelvis_fwd"] = R[..., 0]
    lm["pelvis_left"] = R[..., 1]
    for s, S in (("l", "L"), ("r", "R")):
        lm[f"{s}_knee"] = 0.5 * (f(f"{S}KNE") + f(f"{S}MKNE"))
        lm[f"{s}_ankle"] = 0.5 * (f(f"{S}ANK") + f(f"{S}MANK"))
        lm[f"{s}_toe"] = f(f"{S}TOE")
        lm[f"{s}_heel"] = f(f"{S}HEE")
        lm[f"{s}_elbow"] = 0.5 * (f(f"{S}ELB") + f(f"{S}MELB"))
        lm[f"{s}_wrist"] = 0.5 * (f(f"{S}WRA") + f(f"{S}WRB"))
        lm[f"{s}_hand"] = f(f"{S}FIN")
        lm[f"{s}_wrist_radial"] = f(f"{S}WRA")
        lm[f"{s}_wrist_ulnar"] = f(f"{S}WRB")
    # thorax / shoulders: joint centre ~ 5 cm below the acromion marker along the trunk axis
    c7, clav, strn, t10 = f("C7"), f("CLAV"), f("STRN"), f("T10")
    trunk_up = 0.5 * (c7 + clav) - 0.5 * (t10 + strn)
    trunk_up /= np.linalg.norm(trunk_up, axis=-1, keepdims=True)
    for s, S in (("l", "L"), ("r", "R")):
        lm[f"{s}_shoulder"] = f(f"{S}SHO") - 0.05 * trunk_up
    lm["neck"] = 0.5 * (c7 + clav)
    lm["chest"] = 0.5 * (clav + strn)
    lm["back"] = 0.5 * (c7 + t10)
    lm["head"] = 0.25 * (f("LFHD") + f("RFHD") + f("LBHD") + f("RBHD"))
    if pitching:
        # release ~ peak speed of the throwing-hand marker (exact OBP events live in the full-signal tables)
        hand = _lowpass(m["RFIN"], rate, bat_cutoff)
        v = np.gradient(hand, 1.0 / rate, axis=0)
        speed = np.linalg.norm(v, axis=-1) / 0.44704
        lm["r_hand"] = hand
        return SwingTrial(name=base, rate=rate, lm=lm, height=height, side=side, exit_velo_mph=ev,
                          contact=int(np.argmax(speed)), bat_speed_mph=speed)
    knob = _lowpass(m["Marker1"], rate, bat_cutoff)
    tip = _lowpass(0.5 * (m["Marker2"] + m["Marker3"]), rate, bat_cutoff)
    lm["bat_knob"] = knob
    lm["bat_tip"] = tip
    bat_dir = (tip - knob) / np.linalg.norm(tip - knob, axis=-1, keepdims=True)
    length = np.median(np.linalg.norm(tip - knob, axis=-1))
    lm["bat_sweet"] = knob + bat_dir * (length - sweet_from_tip)

    v = np.gradient(lm["bat_sweet"], 1.0 / rate, axis=0)
    speed = np.linalg.norm(v, axis=-1) / 0.44704
    # contact ~ the frame of peak sweet-spot speed (OBP releases the exact event only in full-signal tables)
    contact = int(np.argmax(speed))
    return SwingTrial(name=base, rate=rate, lm=lm, height=height, side=side, exit_velo_mph=ev,
                      contact=contact, bat_speed_mph=speed)
