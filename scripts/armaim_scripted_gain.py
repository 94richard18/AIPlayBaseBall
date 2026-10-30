"""ArmAim scripted-aim falsifier (after the +300 FAIL, 10-30): per-metre joint offsets that move the sweet spot sideways
with the top hand held on the handle and the sweet-spot height held (step 0's "z held", joint set C); and the same for a
height shift with y held (zplus / zminus, 4 cm, for D). CPU only.

For each two-hand clip at its contact frame: min ||dq||^2 s.t. sweet-spot world y shift = +-shift, grip + height held,
|dq| <= 0.4 and the joint limits (SLSQP on the full FK). gain = dq / shift, median over clips per joint, separately for the
+ (third-base) and - (first-base / outside) side. Written as JSON for ArmAimEnvCfg.aim_scripted_gain_{plus,minus}.

    python scripts/armaim_scripted_gain.py --out logs/armaim_auth/scripted_gain.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aibaseball.physics.bat_model import BatSpec  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid, palm_grip_frame  # noqa: E402
from aibaseball.robot.stance import GRIP_FINGERS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shift", type=float, default=0.08)
ap.add_argument("--bound", type=float, default=0.4)
ap.add_argument("--out", default="logs/armaim_auth/scripted_gain.json")
args = ap.parse_args()
SS = 0.7112
bat = BatSpec()
robot = build_humanoid(with_bat=True, bat=bat)
gp_r, _ = palm_grip_frame("r")
fingers = {f"{s}_{k}": v for s in ("l", "r") for k, v in GRIP_FINGERS.items()}
lim = {j.name: (j.lower, j.upper) for j in robot.actuated}
# the order of ArmAimEnvCfg.aim_joints (pitched_env_cfg._ARMAIM_JOINTS)
JS = [f"{s}_{j}" for s in ("l", "r") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")] + \
     [f"{s}_wrist_{j}" for s in ("l", "r") for j in ("roll", "pitch", "yaw")] + ["waist_yaw"]


def frame(clip, t):
    z = np.load(clip)
    names = [str(n) for n in z["joint_names"]]
    i = int(np.argmin(np.abs(z["time"] - t)))
    q = dict(zip(names, z["joint_pos"][i])) | fingers
    qw = z["root_quat"][i]
    return q, z["root_pos"][i], Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()


def feats(q, p, R):
    P = robot.fk(q, root_pos=p, root_rot=R)
    Rb, pb = P["bat"]
    Rr, pr = P["r_hand"]
    return pb + Rb[:, 0] * SS, pr + Rr @ gp_r - (pb + Rb[:, 0] * bat.top_hand_x)


def with_dq(q, dq):
    q2 = dict(q)
    for j, d in zip(JS, dq):
        q2[j] = q[j] + d
    return q2


def solve(q, p, R, target, axis=1):
    ss0, g0 = feats(q, p, R)
    lo = np.minimum([max(-args.bound, lim[j][0] - q[j]) for j in JS], 0)
    hi = np.maximum([min(args.bound, lim[j][1] - q[j]) for j in JS], 0)

    def cons(dq):
        ss, g = feats(with_dq(q, dq), p, R)
        other = 3 - axis  # y (1) moves -> z (2) held, and the other way round
        return np.concatenate([g - g0, [ss[other] - ss0[other], ss[axis] - ss0[axis] - target]])

    r = minimize(lambda dq: dq @ dq, np.zeros(len(JS)), method="SLSQP", bounds=list(zip(lo, hi)),
                 constraints=[dict(type="eq", fun=cons)], options=dict(maxiter=300, ftol=1e-10))
    err = float(np.abs(cons(r.x)).max())
    return r.x, err


clips = [(f, float(np.load(f)["contact_time"])) for f in sorted(glob.glob("assets/motions/obp_*.npz"))
         if float(np.load(f)["grip_gap"].max()) <= 0.010]
out = {"joints": JS, "shift": args.shift, "bound": args.bound, "clips": [os.path.basename(c) for c, _ in clips]}
for side, sgn, axis in (("plus", 1.0, 1), ("minus", -1.0, 1), ("zplus", 1.0, 2), ("zminus", -1.0, 2)):
    G, E = [], []
    for c, tc in clips:
        dq, err = solve(*frame(c, tc), sgn * args.shift * (1 if axis == 1 else 0.5), axis)
        G.append(dq / (args.shift * (1 if axis == 1 else 0.5)))
        E.append(err)
    G = np.array(G)
    out[f"gain_{side}"] = np.median(G, 0).tolist()  # rad per metre of sweet-spot y (z: zplus / zminus) shift, that direction
    out[f"err_{side}"] = E
    print(f"{side}: constraint err per clip (m) " + " ".join(f"{e:.3f}" for e in E))
    print("  median gain rad/m: " + " ".join(f"{j}={g:+.2f}" for j, g in zip(JS, out[f"gain_{side}"])))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print("SCRIPTED_GAIN_END")
