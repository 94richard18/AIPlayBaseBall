"""Why do noisy-start pitches fall at ~1.8 s? Falls vs successes over the post-release window, one simulator launch.

    python scripts/diag_fall.py --headless --task AIB-Stage1dN70-v0 --checkpoint <model.pt> [--envs 64]
        [--set start_joint_noise=0.02 start_noise_ramp_steps=0]

Per window (1.0-1.4 / 1.4-1.8 / 1.8-2.2 s), for the episodes that fell in 1.5-2.2 s and the ones standing at the end:
foot contact share, foot slip (horizontal foot speed while in contact), COM distance to the support (segment between
the planted feet / the one planted foot), COM speed, and the joints most often at their torque limit (|tau| >= 95%).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dN70-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--set", nargs="*", default=[], help="env cfg overrides, key=value")
parser.add_argument("--sat", type=float, default=0.95, help="torque saturation threshold (fraction of the limit)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402

WINDOWS = ((1.0, 1.4), (1.4, 1.8), (1.8, 2.2))


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, v.lower() in ("1", "true") if isinstance(old, bool) else type(old)(ast.literal_eval(v)))
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    orig = base._get_dones

    def keep_running():  # record falls, never end an episode early (as eval_stage)
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
    n, dt = base.num_envs, base.step_dt
    names = list(base.body_names)  # aligned with base.body_ids
    lim = base.robot.root_physx_view.get_dof_max_forces().to(base.device)[0][base.body_ids].clamp_min(1.0)
    nw = len(WINDOWS)
    # per env, per window sums: steps, lead/back contact, lead/back slip (sum, count), COM-support distance, COM speed
    acc = {k: np.zeros((n, nw)) for k in ("steps", "c0", "c1", "s0", "s1", "sd", "sv")}
    sat = np.zeros((n, nw, len(names)))
    fell_at = np.full(n, np.nan)
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            t = (k + 1) * dt
            f = base.fallen.cpu().numpy()
            fell_at[np.isnan(fell_at) & f] = t
            wi = [i for i, (a, b) in enumerate(WINDOWS) if a <= t < b]
            if not wi:
                continue
            i = wi[0]
            d = base.robot.data
            forces = base.feet.data.net_forces_w_history[:, :, base.foot_sensor_ids].norm(dim=-1).max(1).values
            planted = forces > cfg.contact_force_n
            feet = d.body_link_pos_w[:, base.foot_ids]
            fv = d.body_link_lin_vel_w[:, base.foot_ids, :2].norm(dim=-1)
            com, com_v = base._com()
            p, a, b = com[:, :2], feet[:, 0, :2], feet[:, 1, :2]
            ab = b - a
            s = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
            d_seg = (p - (a + s.unsqueeze(-1) * ab)).norm(dim=-1)
            dist = torch.where(planted[:, 0] & planted[:, 1], d_seg,
                               torch.where(planted[:, 0], (p - a).norm(dim=-1),
                                           torch.where(planted[:, 1], (p - b).norm(dim=-1),
                                                       torch.full_like(d_seg, float("nan")))))
            pl, fvn = planted.cpu().numpy(), fv.cpu().numpy()
            acc["steps"][:, i] += 1
            acc["c0"][:, i] += pl[:, 0]
            acc["c1"][:, i] += pl[:, 1]
            acc["s0"][:, i] += np.where(pl[:, 0], fvn[:, 0], 0.0)
            acc["s1"][:, i] += np.where(pl[:, 1], fvn[:, 1], 0.0)
            acc["sd"][:, i] += np.nan_to_num(dist.cpu().numpy(), nan=0.0)
            acc["sv"][:, i] += com_v.norm(dim=-1).cpu().numpy()
            tq = (d.applied_torque[:, base.body_ids].abs() / lim).cpu().numpy()
            sat[:, i] += tq >= args.sat
        ok = base._stand_state()[4].cpu().numpy() & np.isnan(fell_at)
    late = (fell_at >= 1.5) & (fell_at < 2.2)
    say(f"== diag_fall: {args.checkpoint} | {n} episodes, {args.task}, set {' '.join(args.set) or '-'}")
    say(f"  fell {int((~np.isnan(fell_at)).sum())} (in 1.5-2.2 s: {int(late.sum())}), standing at the end {int(ok.sum())}")
    say(f"  fall times: {np.round(np.sort(fell_at[~np.isnan(fell_at)]), 2).tolist()}")
    for gname, g in (("fell 1.5-2.2 s", late), ("standing", ok)):
        if not g.any():
            continue
        say(f"  -- {gname} ({int(g.sum())} episodes)")
        for i, (a, b) in enumerate(WINDOWS):
            st = acc["steps"][g, i].clip(min=1)
            c0, c1 = acc["c0"][g, i], acc["c1"][g, i]
            say(f"    {a:.1f}-{b:.1f} s: contact lead {np.mean(c0 / st):.2f} back {np.mean(c1 / st):.2f} | slip lead "
                f"{np.sum(acc['s0'][g, i]) / max(c0.sum(), 1):.2f} back {np.sum(acc['s1'][g, i]) / max(c1.sum(), 1):.2f} m/s"
                f" | COM to support {np.mean(acc['sd'][g, i] / st):.3f} m, COM speed {np.mean(acc['sv'][g, i] / st):.2f} m/s")
            share = (sat[g, i] / st[:, None]).mean(0)
            top = np.argsort(-share)[:6]
            say("      at torque limit: " + ", ".join(f"{names[j]} {share[j]:.2f}" for j in top if share[j] > 0.005))


if __name__ == "__main__":
    main()
    app.close()
