"""RW step 0a (plan docs/plans/fastball_restart.md 5-19): wiring check + zero-residual PD target kinematics.

    python scripts/rw_ref_check.py --headless [--ks 0 0.25 0.5 0.75 1.0]

Builds the RW k = 0.25 task once (no physics stepping), samples StageEnv._ref() on a 120 Hz time grid for each k
(cfg.waist_yaw_clock_scale set at run time), and reports:
  * wiring: k = 0 vs the task with the field at 0 (the P3C path) is the same code path; for k > 0 the max |diff| vs k = 0
    on every joint column other than waist_yaw (must be < 1e-6) and on waist_yaw;
  * the waist_yaw target rel. the set pose at the athlete's lead-foot strike (release - 0.147 s) and at the release;
  * CPU FK of the target pose (fingers at the grip, root on the task's root target): hip-shoulder separation
    wrap(pelvis yaw - torso yaw) at the strike, r_hand / ball centre speed at the release and its peak in the last 0.1 s.
These are kinematic references of the zero-residual target, not ceilings.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CRW25-v0")
parser.add_argument("--ks", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
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

STRIKE_REL = -0.147  # the retargeted reference's lead-foot strike rel. the release (logs/dchain_ref.txt)


def say(*a):
    print(*a, flush=True)


def main():
    N = 160
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = N
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    dev = base.device
    dt = 1.0 / 120.0
    t = torch.arange(N, device=dev, dtype=torch.float32) * dt
    jy = base._jcol["waist_yaw"]
    rel = float(base.release_ref)
    ks_, ki_ = int(round((rel + STRIKE_REL) / dt)), int(round(rel / dt))
    robot = build_humanoid(with_bat=False)
    grip = BallGrip.load(os.path.join(ROOT, "assets", "aib1_pitcher", "grip.json"))
    q0 = {j.name: 0.0 for j in robot.actuated}
    names = list(base.body_names)
    o = base.scene.env_origins
    say(f"== rw_ref_check: {args.task}, throw_upper_scale {cfg.throw_upper_scale}, release_ref {rel:.3f} s, strike frame "
        f"{ks_} ({ks_ * dt:.3f} s), release frame {ki_}")
    out = {}
    with torch.inference_mode():
        for k in args.ks:
            base.cfg.waist_yaw_clock_scale = float(k)
            r = base._ref(t)
            out[k] = {x: r[x].clone() for x in ("joint_pos", "joint_vel", "root_pos", "root_quat")}
    b0 = out[args.ks[0]]
    for k in args.ks:
        r = out[k]
        others = [c for c in range(r["joint_pos"].shape[1]) if c != jy]
        dq = (r["joint_pos"][:, others] - b0["joint_pos"][:, others]).abs().max().item()
        dv = (r["joint_vel"][:, others] - b0["joint_vel"][:, others]).abs().max().item()
        dr = max((r["root_pos"] - b0["root_pos"]).abs().max().item(), (r["root_quat"] - b0["root_quat"]).abs().max().item())
        dy = (r["joint_pos"][:, jy] - b0["joint_pos"][:, jy]).abs().max().item()
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
        ok = dq < 1e-6 and dv < 1e-6 and dr < 1e-6
        say(f"[rw0a] k {k:.2f}: wiring other joints max|dq| {dq:.2e} |dqd| {dv:.2e} root {dr:.2e} -> "
            f"{'OK' if ok else 'FAIL'}; waist_yaw max|d| vs k {args.ks[0]} {np.degrees(dy):.1f} deg | waist_yaw target rel. set: "
            f"strike {yd[ks_]:+.1f}, release {yd[ki_]:+.1f} deg | target separation at strike {sep[ks_]:+.1f} deg | "
            f"hand speed release {sh[ki_]:.1f} (peak {sh[w].max():.1f}) | ball centre release {sb[ki_]:.1f} (peak "
            f"{sb[w].max():.1f}) m/s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
