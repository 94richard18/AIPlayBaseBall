"""ArmAim step 0 (planning 14, user 10-30 "動打者架構"): how far a bounded residual on the arm joints can move the sweet
spot sideways at contact. CPU only, robot kinematics (build_humanoid, the URDF's tree), no physics.

For every two-hand reference clip (the swing policy's clips) at its contact frame, and at contact - 40 / - 80 ms:
  joint sets  A = shoulders + elbows (both arms), B = A + wrists, C = B + waist_yaw
  bounds      |dq_i| <= 0.25 / 0.40 rad (0.60 for set C only) (and the joint limits)
  constraint  the top (right) hand stays on the handle (position loop closure as in the retarget, gp_r on the bat at
              top_hand_x; the bat is fixed to l_hand); variant "z held": the sweet-spot height is also held
  objective   max / min of the sweet spot's world y (plate y; + = third-base side) -> reach + / -
Linear (Jacobian LP) and nonlinear (SLSQP on the full FK) both reported; the nonlinear one is the number.
Also the sweet spot's shift along the bat axis (station change of a fixed ball) for the same dq.
P3c spread: plate_y of the pitch table (all / in-zone) and in-zone |plate_y - y0| with y0 = -0.055 (band_select).

    python scripts/armaim_authority.py --out logs/armaim_auth
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aibaseball.game.agents import StrikeZone  # noqa: E402
from aibaseball.physics.bat_model import BatSpec  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid, palm_grip_frame  # noqa: E402
from aibaseball.robot.stance import GRIP_FINGERS  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pitches", default="assets/game/p3c_identical_t18_pitches.csv")
ap.add_argument("--y0", type=float, default=-0.055)
ap.add_argument("--offsets_ms", default="0,-40,-80")
ap.add_argument("--out", default="logs/armaim_auth")
args = ap.parse_args()
SS = 0.7112
bat = BatSpec()
robot = build_humanoid(with_bat=True, bat=bat)
gp_r, _ = palm_grip_frame("r")
fingers = {f"{s}_{k}": v for s in ("l", "r") for k, v in GRIP_FINGERS.items()}
lim = {j.name: (j.lower, j.upper) for j in robot.actuated}
ARM = [f"{s}_{j}" for s in ("l", "r") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
WR = [f"{s}_wrist_{j}" for s in ("l", "r") for j in ("roll", "pitch", "yaw")]
SETS = {"A shoulders+elbows": ARM, "B +wrists": ARM + WR, "C +wrists+waist_yaw": ARM + WR + ["waist_yaw"]}


def frame(clip, t):
    z = np.load(clip)
    names = [str(n) for n in z["joint_names"]]
    i = int(np.argmin(np.abs(z["time"] - t)))
    q = dict(zip(names, z["joint_pos"][i])) | fingers
    qw = z["root_quat"][i]
    R = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
    return q, z["root_pos"][i], R


def feats(q, p, R):
    P = robot.fk(q, root_pos=p, root_rot=R)
    Rb, pb = P["bat"]
    Rr, pr = P["r_hand"]
    ss = pb + Rb[:, 0] * SS
    grip = pr + Rr @ gp_r - (pb + Rb[:, 0] * bat.top_hand_x)
    return ss, Rb[:, 0], grip


def with_dq(q, js, dq):
    q2 = dict(q)
    for j, d in zip(js, dq):
        q2[j] = q[j] + d
    return q2


def solve(q, p, R, js, b, sign, hold_z):
    ss0, ax0, g0 = feats(q, p, R)
    lo = np.array([max(-b, lim[j][0] - q[j]) for j in js])
    hi = np.array([min(b, lim[j][1] - q[j]) for j in js])
    lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)  # a reference pose past a limit: no motion further out
    # Jacobians (central differences)
    eps = 1e-5
    Js, Jg = np.zeros((3, len(js))), np.zeros((3, len(js)))
    for k in range(len(js)):
        e = np.zeros(len(js))
        e[k] = eps
        a, _, ga = feats(with_dq(q, js, e), p, R)
        c, _, gc = feats(with_dq(q, js, -e), p, R)
        Js[:, k], Jg[:, k] = (a - c) / (2 * eps), (ga - gc) / (2 * eps)
    Aeq = np.vstack([Jg, Js[2:3]]) if hold_z else Jg
    lp = linprog(-sign * Js[1], A_eq=Aeq, b_eq=np.zeros(len(Aeq)), bounds=list(zip(lo, hi)), method="highs")
    lin = sign * -lp.fun if lp.success else float("nan")
    x0 = lp.x if lp.success else np.zeros(len(js))

    def cons(dq):
        ss, _, g = feats(with_dq(q, js, dq), p, R)
        c = list(g - g0)
        if hold_z:
            c.append(ss[2] - ss0[2])
        return np.array(c)

    res = minimize(lambda dq: -sign * (feats(with_dq(q, js, dq), p, R)[0][1] - ss0[1]), x0, method="SLSQP",
                   bounds=list(zip(lo, hi)), constraints=[dict(type="eq", fun=cons)], options=dict(maxiter=200, ftol=1e-9))
    ss, ax, g = feats(with_dq(q, js, res.x), p, R)
    d = ss - ss0
    return dict(lin=lin, nl=float(d[1]), dx=float(d[0]), dz=float(d[2]), along_axis=float(d @ ax0),
                axis_rot_deg=float(np.degrees(np.arccos(np.clip(ax @ ax0, -1, 1)))),
                grip_err=float(np.linalg.norm(g - g0)), ok=bool(res.success), sat=float(np.mean(np.isclose(np.abs(res.x), b, atol=1e-3))))


clips = []
for f in sorted(glob.glob("assets/motions/obp_*.npz")):
    z = np.load(f)
    if float(z["grip_gap"].max()) <= 0.010:
        clips.append((f, float(z["contact_time"])))
L = [f"ArmAim authority (kinematic, {len(clips)} two-hand clips: " + " ".join(os.path.basename(c)[4:-4] for c, _ in clips) + ")",
     "reach = sweet-spot world-y shift (m) with the top hand held on the handle; + / - = third / first-base side; "
     "lin = Jacobian LP, nl = nonlinear FK (the number); per cell: median over clips [min]"]
R_ = {}
for off in [float(x) / 1000 for x in args.offsets_ms.split(",")]:
    for hold_z in (False, True):
        for sn, js in SETS.items():
            for b in (0.25, 0.40, 0.60):
                rows = []
                if b > 0.5 and not sn.startswith("C"):
                    continue
                for c, tc in clips:
                    q, p, R = frame(c, tc + off)
                    rows.append((solve(q, p, R, js, b, +1, hold_z), solve(q, p, R, js, b, -1, hold_z)))
                plus = np.array([r[0]["nl"] for r in rows])
                minus = np.array([r[1]["nl"] for r in rows])
                lin_p = np.array([r[0]["lin"] for r in rows])
                lin_m = np.array([r[1]["lin"] for r in rows])
                along = np.array([[r[0]["along_axis"], r[1]["along_axis"]] for r in rows])
                rot = np.array([[r[0]["axis_rot_deg"], r[1]["axis_rot_deg"]] for r in rows])
                dzs = np.array([[r[0]["dz"], r[1]["dz"]] for r in rows])
                ge = max(max(r[0]["grip_err"], r[1]["grip_err"]) for r in rows)
                key = f"t{off * 1000:+.0f}ms {'z held' if hold_z else 'z free'} {sn} +-{b:.2f}"
                R_[key] = dict(plus=plus.tolist(), minus=minus.tolist(), lin_plus=lin_p.tolist(), lin_minus=lin_m.tolist(),
                               along=along.tolist(), rot=rot.tolist(), dz=dzs.tolist())
                L.append(f"{key:52s} reach + {np.median(plus) * 100:5.1f} [{plus.min() * 100:5.1f}] / - {np.median(-minus) * 100:5.1f} "
                         f"[{(-minus).min() * 100:5.1f}] cm (lin + {np.median(lin_p) * 100:5.1f} / - "
                         f"{np.median(-lin_m) * 100:5.1f}); along bat axis {np.median(along[:, 0]) * 100:+5.1f} / {np.median(along[:, 1]) * 100:+5.1f} cm"
                         f"; bat axis turn {np.median(rot):4.1f} deg; dz {np.median(np.abs(dzs)) * 100:4.1f} cm; grip err max {ge * 1000:.1f} mm")
# P3c spread
zone = StrikeZone()
pt = np.genfromtxt(args.pitches, delimiter=",", names=True)
y = -pt["plate_y"]  # TablePitcher flips the CSV's sign (engine plate_y)
iz = np.array([zone.contains(a, b) for a, b in zip(y, pt["plate_z"])]) & (pt["crossed"] > 0)
pc = lambda v: " ".join(f"p{k} {np.percentile(v, k) * 100:+.1f}" for k in (5, 10, 25, 50, 75, 90, 95))  # noqa: E731
L.append(f"P3c pitch table ({len(y)} rows, in-zone {iz.mean():.3f}) engine plate_y cm: all {pc(y)}; sd {y.std() * 100:.1f}")
L.append(f"  in-zone {pc(y[iz])}; sd {y[iz].std() * 100:.1f}")
d = np.abs(y[iz] - args.y0)
L.append(f"  in-zone |plate_y - y0 ({args.y0 * 100:+.1f} cm)| cm: p50 {np.percentile(d, 50) * 100:.1f} p75 {np.percentile(d, 75) * 100:.1f} "
         f"p90 {np.percentile(d, 90) * 100:.1f} p95 {np.percentile(d, 95) * 100:.1f} max {d.max() * 100:.1f}; needed side: + {(y[iz] - args.y0).max() * 100:.1f} / - "
         f"{(args.y0 - y[iz]).max() * 100:.1f} (max), p95 + {np.percentile(y[iz] - args.y0, 95) * 100:.1f} / - {np.percentile(args.y0 - y[iz], 95) * 100:.1f}")
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(L) + "\n")
json.dump(R_, open(os.path.join(args.out, "armaim_auth.json"), "w"), indent=1)
print("\n".join(L))
print("ARMAIM_AUTH_END")
