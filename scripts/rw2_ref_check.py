"""RW-2 step 0a (plan docs/plans/fastball_restart.md 5-23): wiring + zero-residual PD target kinematics per arm scale s.

    python scripts/rw2_ref_check.py --headless

* cfg diff: each RW2S<s> task vs P3CRW100 (only throw_upper_scale / throw_upper_scale_joints may differ; S80 none);
* builds RW2S25 once, sets throw_upper_scale / the wrist override per s at run time and samples StageEnv._ref() on a
  120 Hz grid; S80 vs the P3CRW100 path is the same code (cfg identical), so _ref equality follows from the cfg diff;
* per s: waist_yaw target rel. set at the strike (release - 0.147 s) / release, target separation at the strike,
  r_hand / ball centre speed at the release (CPU FK, rw_ref_check's method). Kinematic references, not ceilings.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.analysis import pitch_report as PR  # noqa: E402
from aibaseball.robot.ball_grip import BallGrip  # noqa: E402
from aibaseball.robot.humanoid import build_humanoid  # noqa: E402

PFX = "AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3C"
STRIKE_REL = -0.147


def say(*a):
    print(*a, flush=True)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("throw_upper_scale_joints",):
            out |= flat(v, p + k + ".")
        else:
            out[p + k] = v
    return out


def main():
    ref = flat(load_cfg_from_registry(PFX + "RW100-v0", "env_cfg_entry_point").to_dict())
    for s in (25, 50, 75, 80):
        c = flat(load_cfg_from_registry(f"{PFX}RW2S{s}-v0", "env_cfg_entry_point").to_dict())
        diff = {k: (ref.get(k), c.get(k)) for k in set(ref) | set(c) if str(ref.get(k)) != str(c.get(k))}
        bad = [k for k in diff if k not in ("throw_upper_scale", "throw_upper_scale_joints")]
        say(f"[rw2cfg] S{s} vs RW100: differs in {sorted(diff)} -> {'OK' if not bad else 'FAIL ' + str(bad)} | "
            + "; ".join(f"{k} {diff[k][0]} -> {diff[k][1]}" for k in sorted(diff)))
    N = 160
    cfg = load_cfg_from_registry(PFX + "RW2S25-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = N
    base = gym.make(PFX + "RW2S25-v0", cfg=cfg).unwrapped
    dt = 1.0 / 120.0
    t = torch.arange(N, device=base.device, dtype=torch.float32) * dt
    jy = base._jcol["waist_yaw"]
    rel = float(base.release_ref)
    ks_, ki_ = int(round((rel + STRIKE_REL) / dt)), int(round(rel / dt))
    robot = build_humanoid(with_bat=False)
    grip = BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json"))
    q0 = {j.name: 0.0 for j in robot.actuated}
    names = list(base.body_names)
    o = base.scene.env_origins
    for s in (0.25, 0.5, 0.75, 0.8):
        base.cfg.throw_upper_scale = s
        base.cfg.throw_upper_scale_joints = {"r_wrist_.*": min(1.0, 1.25 * s)}
        with torch.inference_mode():
            r = base._ref(t)
        yd = np.degrees((r["joint_pos"][:, jy] - r["joint_pos"][0, jy]).cpu().numpy())
        q = r["joint_pos"].cpu().numpy()
        p = (r["root_pos"] - o).cpu().numpy()
        rq = r["root_quat"].cpu().numpy()
        Rm = Rotation.from_quat(np.concatenate([rq[:, 1:], rq[:, :1]], -1)).as_matrix()
        Rp, Rt, hand, ball = [], [], [], []
        for f in range(N):
            fk = robot.fk(q0 | dict(zip(names, q[f])) | grip.joint_pos, root_pos=p[f], root_rot=Rm[f])
            Rp.append(fk["pelvis"][0])
            Rt.append(fk["torso"][0])
            Rh, ph = fk["r_hand"]
            hand.append(ph)
            ball.append(ph + Rh @ np.asarray(grip.ball_center_hand))
        sep = (PR.yaw_deg(np.stack(Rp)) - PR.yaw_deg(np.stack(Rt)) + 180) % 360 - 180
        sh = np.linalg.norm(np.gradient(np.stack(hand), dt, axis=0), axis=-1)
        sb = np.linalg.norm(np.gradient(np.stack(ball), dt, axis=0), axis=-1)
        w = slice(ki_ - 12, ki_ + 1)
        say(f"[rw2a] s {s:.2f}: waist_yaw target rel. set strike {yd[ks_]:+.1f}, release {yd[ki_]:+.1f} deg | target "
            f"separation at strike {sep[ks_]:+.1f} deg | hand release {sh[ki_]:.1f} (peak {sh[w].max():.1f}) | ball centre "
            f"release {sb[ki_]:.1f} (peak {sb[w].max():.1f}) m/s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
