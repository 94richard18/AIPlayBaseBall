"""Per-step balance trace for a fall diagnosis (plan 5-25, Codex R36 / R37): when do the balance signs appear,
relative to the lead-foot touchdown and the release.

    python scripts/diag_fall_timeline.py --headless --task <task> --checkpoint <model.pt> --out <trace.npz> [--envs 128]
        [--set start_joint_noise=0.02 start_noise_ramp_steps=0]

Episodes keep running after a fall (dones disabled, as in diag_fall_release). Saved per env and per step (t = (k+1) dt):
COM position / velocity (world, mass-weighted from the body links), lead / back foot link position (xyz), foot planted
flags (contact force > contact_force_n), torso up-vector xy (lean: x toward the plate, y toward first base) and trunk
tilt (deg), lead knee angle (deg), released / fallen flags. Plus the end-of-episode at-rest flag. No summary here; the
event alignment is done offline by scripts/fall_timeline_report.py, so the definitions live in one place.
"""

import argparse
import ast
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--set", nargs="*", default=["start_joint_noise=0.02", "start_noise_ramp_steps=0"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        if isinstance(old, bool):
            v = v.lower() in ("1", "true")
        elif old is None:
            try:
                v = None if v == "None" else ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        else:
            v = type(old)(v)
        setattr(cfg, k, v)
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():
        orig()
        z = torch.zeros_like(base.fallen)
        return z, z
    base._get_dones = keep_running
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    n, dt, dev, rob = base.num_envs, base.step_dt, base.device, base.robot
    knee = rob.find_joints("l_knee")[0][0]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    keys = {"com": 3, "comv": 3, "lead": 3, "back": 3, "planted": 2, "lean": 2, "tilt": 1, "knee": 1, "rel": 1, "fallen": 1}
    tr = {k: np.zeros((n, T, d), dtype=np.float32) for k, d in keys.items()}
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            d = rob.data
            com, comv = base._com()
            org = base.scene.env_origins
            tr["com"][:, k] = (com - org).cpu().numpy()
            tr["comv"][:, k] = comv.cpu().numpy()
            tr["lead"][:, k] = (d.body_link_pos_w[:, base.foot_ids[0]] - org).cpu().numpy()
            tr["back"][:, k] = (d.body_link_pos_w[:, base.foot_ids[1]] - org).cpu().numpy()
            forces = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values
            tr["planted"][:, k] = (forces > cfg.contact_force_n).cpu().numpy()
            upv = quat_apply(d.body_link_quat_w[:, base.torso_id], up_local)
            tr["lean"][:, k] = upv[:, :2].cpu().numpy()
            tr["tilt"][:, k, 0] = torch.rad2deg(torch.acos(upv[:, 2].clamp(-1, 1))).cpu().numpy()
            tr["knee"][:, k, 0] = torch.rad2deg(d.joint_pos[:, knee]).cpu().numpy()
            tr["rel"][:, k, 0] = base.released.cpu().numpy()
            tr["fallen"][:, k, 0] = base.fallen.cpu().numpy()
        both, speed, tilt, head_z, ok = (x.cpu().numpy() for x in base._stand_state())
    fell = tr["fallen"][:, -1, 0] > 0
    np.savez_compressed(args.out, dt=dt, at_rest=ok & ~fell, task=args.task, checkpoint=args.checkpoint,
                        contact_force_n=cfg.contact_force_n, **tr)
    say(f"== diag_fall_timeline: {args.checkpoint} | {args.task}, {n} envs x {T} steps (dt {dt:.4f}), set {args.set}; "
        f"fell {int(fell.sum())}, at rest {int((ok & ~fell).sum())} -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
