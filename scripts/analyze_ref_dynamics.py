"""Is a reference motion dynamically possible for AIB-1? Centroidal analysis of the reference (no simulator).

For every frame: whole-body COM (robot masses), centroidal angular momentum, the ground reaction it needs
(vertical force, friction ratio) and its zero-moment point (ZMP / centre of pressure) against the feet that touch
the ground. A ZMP outside the support polygon, a vertical force <= 0 or a friction ratio above ~1 means no ground
contact can produce the motion: a robot that tracks it must fall (or slip).

    python scripts/analyze_ref_dynamics.py [assets/motions/pitch_2916-4.npz] [--t0 0.6 --t1 1.9]
"""

import argparse
import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402

G = 9.81
SOLE = dict(x=(-0.09, 0.17), y=(-0.055, 0.055), z=-0.08)  # foot box (0.26 x 0.11), bottom 8 cm under the ankle frame


def mound_z(x, top=0.058, h=0.254, slope=1 / 12, x0=0.54):
    """Mound surface fitted to the capture: pivot sole at +0.058 m, the lead foot lands at -0.03 m (x = 1.6 m)."""
    return np.where(x < x0, top, np.maximum(top - (x - x0) * slope, top - h))


def cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross2(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross2(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def signed_dist(p, poly):
    """< 0 inside the convex polygon (distance to the edge), > 0 outside."""
    if len(poly) == 0:
        return np.inf
    if len(poly) < 3:
        a = np.array(poly[0])
        b = np.array(poly[-1])
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))
    inside = all(cross2(poly[i], poly[(i + 1) % len(poly)], p) >= 0 for i in range(len(poly)))
    d = []
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d.append(np.linalg.norm(p - (a + t * ab)))
    return -min(d) if inside else min(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", nargs="?", default=os.path.join(ROOT, "assets", "motions", "pitch_2916-4.npz"))
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=99.0)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--flat", action="store_true", help="flat floor instead of the mound")
    a = ap.parse_args()

    d = np.load(a.motion, allow_pickle=True)
    t, dt = d["time"], float(d["time"][1] - d["time"][0])
    names = [str(n) for n in d["joint_names"]]
    robot = build_humanoid(with_bat=False, name="aib1_pitcher", arm_masses="human")
    grip = BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json"))
    fingers = {f"l_{n}": 0.2 for n in FINGER_JOINTS} | grip.joint_pos
    links = list(robot.links.values())
    m = np.array([lk.mass for lk in links])
    M = m.sum()
    Rroot = Rotation.from_quat(np.concatenate([d["root_quat"][:, 1:], d["root_quat"][:, :1]], -1)).as_matrix()
    T = len(t)
    pc = np.zeros((T, len(links), 3))  # link COM positions
    Rl = np.zeros((T, len(links), 3, 3))
    soles = {s: np.zeros((T, 4, 3)) for s in ("l", "r")}
    for f in range(T):
        P = robot.fk(dict(zip(names, d["joint_pos"][f])) | fingers, root_pos=d["root_pos"][f], root_rot=Rroot[f])
        for i, lk in enumerate(links):
            R, p = P[lk.name]
            Rl[f, i] = R
            pc[f, i] = p + R @ lk.com
        for s in ("l", "r"):
            R, p = P[f"{s}_foot"]
            soles[s][f] = [p + R @ np.array([x, y, SOLE["z"]]) for x in SOLE["x"] for y in SOLE["y"]]
    C = (pc * m[None, :, None]).sum(1) / M
    win = 9
    Cd = savgol_filter(C, win, 3, deriv=1, delta=dt, axis=0)
    Cdd = savgol_filter(C, win, 3, deriv=2, delta=dt, axis=0)
    v = savgol_filter(pc, win, 3, deriv=1, delta=dt, axis=0)
    # link angular velocity from rotation finite differences
    w = np.zeros((T, len(links), 3))
    for i in range(len(links)):
        r = Rotation.from_matrix(Rl[:, i])
        w[1:-1, i] = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * dt)
    w[0], w[-1] = w[1], w[-2]
    L = np.zeros((T, 3))
    for i, lk in enumerate(links):
        Iw = np.einsum("tij,jk,tlk,tl->ti", Rl[:, i], lk.inertia, Rl[:, i], w[:, i])
        L += m[i] * np.cross(pc[:, i] - C, v[:, i] - Cd) + Iw
    Ld = savgol_filter(L, win, 3, deriv=1, delta=dt, axis=0)

    fz = M * (G + Cdd[:, 2])
    print(f"motion {os.path.basename(a.motion)}  mass {M:.1f} kg  release (ref) {float(d['contact_time']):.3f} s  "
          f"length {t[-1]:.3f} s  ground: {'flat' if a.flat else 'mound'}")
    print("   t    | COM x     y     z  | COM vel x   y    z  | Fz/Mg | mu_req | contact | ZMP x     y   | ZMP to support (cm, <0 inside)")
    bad = []
    for f in range(0, T, 1):
        if not (a.t0 <= t[f] <= a.t1):
            continue
        h_g = []
        corners = []
        on = ""
        for s in ("l", "r"):
            S = soles[s][f]
            gz = np.zeros(4) if a.flat else mound_z(S[:, 0])
            touching = (S[:, 2] - gz) < 0.025
            if touching.any():
                on += s
                corners += [c[:2] for c in S[touching]]
                h_g += list(gz[touching])
        hg = float(np.mean(h_g)) if h_g else 0.0
        den = M * (G + Cdd[f, 2])
        zx = C[f, 0] - (C[f, 2] - hg) * Cdd[f, 0] / (G + Cdd[f, 2]) - Ld[f, 1] / den
        zy = C[f, 1] - (C[f, 2] - hg) * Cdd[f, 1] / (G + Cdd[f, 2]) + Ld[f, 0] / den
        dist = signed_dist(np.array([zx, zy]), hull(corners)) if corners else np.inf
        mu = np.linalg.norm(M * Cdd[f, :2]) / max(fz[f], 1e-6)
        if dist > 0.02 or fz[f] <= 0 or mu > 1.0:
            bad.append((t[f], dist, fz[f] / (M * G), mu, on))
        if (f % a.every) == 0:
            flag = " <-- outside" if dist > 0.02 else ""
            print(f" {t[f]:.3f} | {C[f,0]:5.2f} {C[f,1]:5.2f} {C[f,2]:5.2f} | {Cd[f,0]:+5.2f} {Cd[f,1]:+5.2f} {Cd[f,2]:+5.2f} | "
                  f"{fz[f]/(M*G):5.2f} | {mu:5.2f}  | {on or '-':7s} | {zx:5.2f} {zy:5.2f} | {dist*100:+7.1f}{flag}")
    print(f"\nframes needing an impossible ground force (ZMP > 2 cm outside the feet, Fz <= 0 or mu > 1): {len(bad)}")
    if bad:
        spans, s0, prev = [], bad[0][0], bad[0][0]
        for tb, *_ in bad[1:]:
            if tb - prev > 1.5 * dt:
                spans.append((s0, prev))
                s0 = tb
            prev = tb
        spans.append((s0, prev))
        print("  time spans:", ", ".join(f"{x:.3f}-{y:.3f}" for x, y in spans))
        worst = max(bad, key=lambda b: b[1] if np.isfinite(b[1]) else 0)
        print(f"  worst ZMP excursion {worst[1]*100:.1f} cm at t = {worst[0]:.3f} s (contact '{worst[4]}')")


if __name__ == "__main__":
    main()
