"""Side-by-side preview of an OBP swing (scaled human landmarks) and the retargeted AIB-1 motion.

    python scripts/preview_retarget.py assets/motions/obp_swing_42_5.npz data/c3d/<trial>.c3d
Writes <npz>.mp4 and <npz>_grid.png
"""

import os
import sys

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aibaseball.mocap.obp import load_trial  # noqa: E402
from aibaseball.mocap.retarget import ROBOT_HEIGHT  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402
from aibaseball.robot.stance import GRIP_FINGERS  # noqa: E402
from scripts.build_robot import _shape_lines  # noqa: E402

BONES = [("l_hip", "r_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"), ("l_ankle", "l_heel"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"), ("r_ankle", "r_heel"),
         ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("neck", "head"), ("bat_knob", "bat_tip")]


def main(npz_path, c3d_path):
    d = np.load(npz_path)
    trial = load_trial(c3d_path)
    robot = build_humanoid()
    names = list(d["joint_names"])
    fingers = {f"{s}_{k}": v for s in ("l", "r") for k, v in GRIP_FINGERS.items()}
    scale = ROBOT_HEIGHT / trial.height
    t_src = np.arange(trial.T) / trial.rate
    t0 = trial.contact / trial.rate - float(d["contact_time"])
    lm = {k: v for k, v in trial.lm.items() if k not in ("pelvis_fwd", "pelvis_left")}
    foot_min = min(lm[k][:, 2].min() * scale for k in ("l_toe", "r_toe", "l_heel", "r_heel"))
    frames = []
    T = len(d["time"])
    step = 3
    for f in range(0, T, step):
        t = d["time"][f] + t0
        H = {k: np.array([np.interp(t, t_src, v[:, i]) for i in range(3)]) * scale - np.array([0, 0, foot_min - 0.02])
             for k, v in lm.items()}
        q = dict(zip(names, d["joint_pos"][f])) | fingers
        w, x, y, z = d["root_quat"][f]
        P = robot.fk(q, root_pos=d["root_pos"][f], root_rot=Rotation.from_quat([x, y, z, w]).as_matrix())
        fig = plt.figure(figsize=(12, 6))
        for k, (title, azim) in enumerate((("view from the pitcher (+X)", 180), ("view from 3B side", 90 + 180))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            for a, b in BONES:
                ax.plot(*zip(H[a], H[b]), c="k", lw=2, alpha=0.6)
            for name, link in robot.links.items():
                R, p = P[name]
                for s in link.shapes:
                    for a, b, r in _shape_lines(s, R, p):
                        col = "saddlebrown" if name == "bat" else ("tab:blue" if name.startswith("l_") else
                                                                   "tab:red" if name.startswith("r_") else "gray")
                        if not np.allclose(a, b):
                            ax.plot(*zip(a, b), c=col, lw=max(1.0, r * 80), alpha=0.7)
            tee = d["tee_pos"]
            ax.scatter(*tee, c="gold", s=40)
            ax.set_xlim(-1.5, 1.0), ax.set_ylim(-1.0, 1.5), ax.set_zlim(0, 2.5)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=10, azim=azim)
            ax.set_title(f"{title}  t={d['time'][f] - float(d['contact_time']):+.3f}s (0 = contact)")
        fig.text(0.02, 0.02, "black: OBP athlete (scaled)   colour: AIB-1 retarget   gold: ball/tee", fontsize=10)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(img)
    out = npz_path[:-4] + ".mp4"
    imageio.mimwrite(out, frames, fps=120 / step / 4, quality=8, macro_block_size=1)  # 4x slow motion
    c = int(float(d["contact_time"]) * 120 / step)
    pick = [0, len(frames) // 3, c - 6, c - 2, c, min(c + 6, len(frames) - 1)]
    grid = np.vstack([np.hstack([frames[i][::2, ::2] for i in pick[:3]]), np.hstack([frames[i][::2, ::2] for i in pick[3:]])])
    imageio.imwrite(npz_path[:-4] + "_grid.png", grid)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
