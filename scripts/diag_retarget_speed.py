"""Where does the pitch retarget lose hand speed? CPU only: OBP markers -> scaled landmarks -> 120 Hz resample ->
IK joint angles (assets/motions/pitch_<tag>.npz) -> robot FK, with the hand / ball speed at each stage.

    python scripts/diag_retarget_speed.py [data/c3d_pitching/<trial>.c3d] [assets/motions/pitch_<tag>.npz]
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.mocap.obp import _read_c3d, load_trial  # noqa: E402
from aibaseball.mocap.retarget import ROBOT_HEIGHT  # noqa: E402
from aibaseball.robot.ball_grip import BallGrip  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402

c3d_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "c3d_pitching", "001596_002916_73_191_012_FF_944.c3d")
npz_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "motions", "pitch_2916-4.npz")


def speed(p, dt):
    return np.linalg.norm(np.gradient(p, dt, axis=0), axis=-1)


raw, rate = _read_c3d(c3d_path)
tr = load_trial(c3d_path, body_cutoff=20.0, bat_cutoff=30.0)
scale = ROBOT_HEIGHT / tr.height
t_rel = tr.contact / rate
print(f"trial {tr.name}: {tr.exit_velo_mph:.1f} mph = {tr.exit_velo_mph * 0.44704:.1f} m/s ball, athlete {tr.height:.3f} m, "
      f"robot {ROBOT_HEIGHT} m (scale {scale:.3f}), capture {rate:.0f} Hz, release frame (peak RFIN speed) {t_rel:.3f} s")
k = tr.contact
w = slice(k - int(0.1 * rate), k + int(0.03 * rate))
for lab, p in (("RFIN raw (unfiltered)", raw["RFIN"]), ("RFIN 30 Hz low-pass", tr.lm["r_hand"]),
               ("wrist centre 20 Hz", tr.lm["r_wrist"]), ("elbow 20 Hz", tr.lm["r_elbow"]), ("shoulder 20 Hz", tr.lm["r_shoulder"])):
    s = speed(p, 1 / rate)
    print(f"  athlete {lab:24s}: at release {s[k]:5.1f} m/s, peak in window {np.nanmax(s[w]):5.1f} m/s "
          f"(x scale {scale:.3f} -> {s[k] * scale:5.1f} / {np.nanmax(s[w]) * scale:5.1f})")
# 120 Hz resample of the (scaled) hand landmark, as retarget() does
t_src = np.arange(tr.T) / rate
times = np.arange(t_rel - 1.2, t_rel + 0.14, 1 / 120.0)
for lab, key in (("RFIN 30 Hz", "r_hand"), ("wrist 20 Hz", "r_wrist")):
    p120 = np.stack([np.interp(times, t_src, tr.lm[key][:, i]) for i in range(3)], -1) * scale
    s120 = speed(p120, 1 / 120)
    kk = int(round((t_rel - times[0]) * 120))
    print(f"  scaled + 120 Hz {lab:12s}: at release {s120[kk]:5.1f} m/s, peak {s120[kk - 12:kk + 4].max():5.1f} m/s")
# segment lengths
L = lambda a, b: np.median(np.linalg.norm(tr.lm[a] - tr.lm[b], axis=-1))  # noqa: E731
print(f"  athlete segments (scaled to robot height): upper arm {L('r_shoulder', 'r_elbow') * scale:.3f} m, forearm "
      f"{L('r_elbow', 'r_wrist') * scale:.3f} m, wrist->RFIN {L('r_wrist', 'r_hand') * scale:.3f} m, shoulder width "
      f"{L('r_shoulder', 'l_shoulder') * scale:.3f} m")

# robot FK of the retargeted angles
robot = build_humanoid(with_bat=False, name="aib1_pitcher")
grip = BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json"))
d = np.load(npz_path)
names = [str(n) for n in d["joint_names"]]
T = len(d["time"])
q0 = {j.name: 0.0 for j in robot.actuated}
rq = d["root_quat"]
Rm = Rotation.from_quat(np.concatenate([rq[:, 1:], rq[:, :1]], -1)).as_matrix()
P = {b: np.zeros((T, 3)) for b in ("r_hand", "r_forearm", "r_upper_arm", "torso", "ball")}
links = set(robot.links)
for f in range(T):
    fk = robot.fk(q0 | dict(zip(names, d["joint_pos"][f])) | grip.joint_pos, root_pos=d["root_pos"][f], root_rot=Rm[f])
    for b in P:
        if b == "ball":
            Rh, ph = fk["r_hand"]
            P[b][f] = ph + Rh @ np.asarray(grip.ball_center_hand)
        elif b in fk:
            P[b][f] = fk[b][1]
kr = int(round(float(d["contact_time"]) * 120))
print(f"  robot FK (npz {os.path.basename(npz_path)}, 120 Hz, release frame {kr}):")
for b in P:
    if np.abs(P[b]).sum() == 0:
        continue
    s = speed(P[b], 1 / 120)
    print(f"    {b:12s}: at release {s[kr]:5.1f} m/s, peak {s[kr - 12:kr + 4].max():5.1f} m/s at frame {kr - 12 + int(s[kr - 12:kr + 4].argmax())}")
print(f"  npz key_pos ball (prepare's FK): at release {speed(d['key_pos'][:, -1], 1 / 120)[kr]:.1f} m/s")
print(f"  robot links available: {sorted(l for l in links if l.startswith('r_'))}")
# robot segment lengths
fk0 = robot.fk(q0 | grip.joint_pos)
for a, b in (("r_upper_arm", "r_forearm"), ("r_forearm", "r_hand")):
    if a in fk0 and b in fk0:
        print(f"  robot {a} -> {b} origin distance {np.linalg.norm(fk0[a][1] - fk0[b][1]):.3f} m")
Rh, ph = fk0["r_hand"]
print(f"  robot r_hand origin -> ball centre {np.linalg.norm(Rh @ np.asarray(grip.ball_center_hand)):.3f} m")
# joint angular velocities around the release (the arm)
qd = np.gradient(d["joint_pos"], 1 / 120, axis=0)
print("  retargeted joint |velocity| peak over the last 0.1 s before release (deg/s) / joint limit (rad/s):")
for n in ("waist_yaw", "waist_pitch", "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_pitch"):
    i = names.index(n)
    print(f"    {n:18s} {np.degrees(np.abs(qd[kr - 12:kr + 1, i]).max()):7.0f}   limit {robot.joint(n).velocity:.0f} rad/s; "
          f"range [{np.degrees(robot.joint(n).lower):.0f}, {np.degrees(robot.joint(n).upper):.0f}] deg, at limit in window: "
          f"{int(((d['joint_pos'][kr - 12:kr + 1, i] <= robot.joint(n).lower + 1e-3) | (d['joint_pos'][kr - 12:kr + 1, i] >= robot.joint(n).upper - 1e-3)).sum())} frames")
print(f"  keypoint RMS error at release {float(d['keypoint_err'][kr]) * 100:.1f} cm, over the last 0.1 s max "
      f"{float(d['keypoint_err'][kr - 12:kr + 1].max()) * 100:.1f} cm")
