"""Retarget OBP swings onto AIB-1 and prepare imitation references.

    python scripts/build_motion.py                       # default: fastest right-handed swing
    python scripts/build_motion.py data/c3d/<trial>.c3d  # specific trial(s)
Outputs assets/motions/obp_<session_swing>.npz (+ .mp4 / _grid.png previews)
"""

import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aibaseball.mocap.motion import prepare  # noqa: E402
from aibaseball.mocap.obp import load_trial  # noqa: E402
from aibaseball.mocap.retarget import retarget, save  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402


def main(paths):
    sel = {r["file"]: r["session_swing"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "c3d", "selected.csv")))}
    robot = build_humanoid()
    os.makedirs(os.path.join(ROOT, "assets", "motions"), exist_ok=True)
    for path in paths:
        trial = load_trial(path)
        tag = sel.get(os.path.basename(path), trial.name).replace("_", "-")
        out = os.path.join(ROOT, "assets", "motions", f"obp_{tag}.npz")
        print(f"[motion] {trial.name}: athlete {trial.height:.2f} m, peak bat speed {trial.bat_speed_mph.max():.1f} mph")
        res = retarget(trial, t_before=1.2, t_after=0.45, rate=120.0, robot=robot, verbose=False)
        save(res, out)
        d = prepare(out, out)
        names = list(d["joint_names"])
        over = []
        for i, n in enumerate(names):
            j = robot.joint(n)
            m = np.abs(d["joint_vel"][:, i]).max()
            if m > j.velocity:
                over.append(f"{n} {m:.0f}>{j.velocity:.0f}")
        tip_v = np.linalg.norm(np.gradient(d["key_pos"][:, -1], 1 / 120, axis=0), axis=1).max()
        print(f"[motion]  -> {out}\n          keypoint RMS {res.keypoint_err.mean()*100:.1f} cm, bat err max "
              f"{res.bat_err.max()*100:.1f} cm, grip gap max {res.grip_gap.max()*1000:.1f} mm, bat tip {tip_v:.1f} m/s\n"
              f"          joints over velocity limit: {over or 'none'}")
        if not NO_PREVIEW:
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "preview_retarget.py"), out, path], check=False)


NO_PREVIEW = "--no-preview" in sys.argv

if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--no-preview"]
    if argv == ["--all"]:
        import glob

        argv = sorted(glob.glob(os.path.join(ROOT, "data", "c3d", "*.c3d")))
    args = argv or [os.path.join(ROOT, "data", "c3d", "000145_000042_76_213_R_016_1048.c3d")]
    main(args)
