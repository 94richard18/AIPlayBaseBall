"""Release-window sweep (eval only): hold the ball through the arm swing and, at every control step where the throwing
hand's horizontal heading is within +-HEAD deg of the plate line, ask where an ideal release there (the ball's own
state that step) would cross the plate - gravity only and with drag + Magnus (simulate_to_plane). Shows whether any
on-line moment of the swing gives an air crossing (W1 T2+) and at what hand / ball elevation.

    python scripts/diag_release_window.py --headless --task AIB-Stage1dK70GW2M-v0 --checkpoint <model.pt> [--zero]
        [--envs 64] [--noise 0.02]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2M-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--zero", action="store_true", help="zero actions (PD on the reference only)")
parser.add_argument("--noise", type=float, default=0.0, help="start joint noise (rad)")
parser.add_argument("--head", type=float, default=10.0, help="on-line: hand heading within +-this (deg)")
parser.add_argument("--hold_s", type=float, default=0.35, help="scripted release this long after the athlete's (s)")
parser.add_argument("--t0", type=float, default=0.88)
parser.add_argument("--t1", type=float, default=1.30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.physics.ball_flight import simulate_to_plane  # noqa: E402


def say(*a):
    print(*a, flush=True)


def q(x, ps=(10, 50, 90)):
    return "/".join(f"{np.percentile(x, p):+.1f}" for p in ps) if len(x) else "-"


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.start_joint_noise, cfg.start_noise_ramp_steps = args.noise, 0
    cfg.release_ahead_m, cfg.release_heading_deg = None, None
    cfg.scripted_release_s = args.hold_s  # keep the ball in the hand through the whole window
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
    n = base.num_envs
    rows = []  # env, t, hand heading, hand elev, hand speed, ball pos (3), ball vel (3), ball omega (3)
    T = int(round(args.t1 / base.step_dt)) + 4
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(T):
            act = pol(obs)
            if args.zero:
                act = torch.zeros_like(act)
            obs, _, _, _ = w.step(act)
            tr = base._ref_time()
            d = base.robot.data
            hv = d.body_link_lin_vel_w[:, base.hand_id]
            hd = torch.rad2deg(torch.atan2(hv[:, 1], hv[:, 0]))
            he = torch.rad2deg(torch.atan2(hv[:, 2], hv[:, :2].norm(dim=-1)))
            on = (tr >= args.t0) & (tr <= args.t1) & (hv[:, 0] > 5.0) & (hd.abs() <= args.head) & ~base.released
            if on.any():
                ids = on.nonzero().squeeze(-1)
                bp = base.ball.data.root_pos_w[ids] - base.scene.env_origins[ids]
                rows.append(torch.cat([ids.unsqueeze(-1).float(), tr[ids, None], hd[ids, None], he[ids, None],
                                       hv[ids].norm(dim=-1, keepdim=True), bp, base.ball.data.root_lin_vel_w[ids],
                                       base.ball.data.root_ang_vel_w[ids]], -1).cpu())
    c = cfg
    say(f"== diag_release_window: {args.checkpoint}{' (zero actions)' if args.zero else ''} | {args.task}, {n} envs, "
        f"start noise {args.noise}, on-line = hand heading within +-{args.head:.0f} deg over {args.t0}-{args.t1} s")
    if not rows:
        say("  no on-line step in any env")
        return
    a = torch.cat(rows).numpy()
    env_i, t, hd, he, hs = a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    p, v, om = a[:, 5:8], a[:, 8:11], a[:, 11:14]
    fz = base.field_z
    be = np.degrees(np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1])))
    bs = np.linalg.norm(v, axis=1)
    # gravity only
    tt = (c.plate_distance - p[:, 0]) / np.maximum(v[:, 0], 1e-6)
    zg = p[:, 2] + v[:, 2] * tt - 0.5 * 9.81 * tt**2 - fz
    yg = p[:, 1] + v[:, 1] * tt - c.zone_center_y
    # drag + Magnus
    cross, _, _, reached = simulate_to_plane(torch.tensor(p), torch.tensor(v), torch.tensor(om), base.phys,
                                             c.plate_distance, t_max=2.5, ground_z=fz)
    za = cross[:, 2].numpy() - fz
    ya = cross[:, 1].numpy() - c.zone_center_y
    ra = reached.numpy()
    rg = zg > 0
    envs = np.unique(env_i)
    say(f"  envs with an on-line step: {len(envs)}/{n}; on-line steps per env median "
        f"{np.median(np.bincount(env_i)[envs]):.0f}, at {q(t, (10, 50, 90))} s (p10/50/90)")
    say(f"  on-line steps (p10/50/90): hand elev {q(he)} deg, hand speed {q(hs)} m/s; ball elev {q(be)} deg, "
        f"ball speed {q(bs)} m/s, ball z {q(p[:, 2] - fz)} m")
    say(f"  ideal release at an on-line step reaches the plate in the air: gravity {rg.mean():.0%}, aero {ra.mean():.0%}; "
        f"T3 box (|y|<=0.5, z 0.2-1.5): gravity {(rg & (abs(yg) <= 0.5) & (zg >= 0.2) & (zg <= 1.5)).mean():.0%}, "
        f"aero {(ra & (abs(ya) <= 0.5) & (za >= 0.2) & (za <= 1.5)).mean():.0%}")
    say("  by hand elevation (deg): steps | ball speed med | ball z med | plate z med grav / aero | air grav / aero "
        "| T3 aero | envs")
    for lo in range(-60, 40, 10):
        m = (he >= lo) & (he < lo + 10)
        if not m.any():
            continue
        t3 = ra & (abs(ya) <= 0.5) & (za >= 0.2) & (za <= 1.5)
        say(f"    [{lo:+3d},{lo + 10:+3d}) {m.sum():5d} | {np.median(bs[m]):5.1f} | {np.median(p[m, 2] - fz):4.2f} | "
            f"{np.median(zg[m]):+6.2f} / {np.median(za[m]):+6.2f} | {rg[m].mean():4.0%} / {ra[m].mean():4.0%} | "
            f"{t3[m].mean():4.0%} | {len(np.unique(env_i[m]))}")
    best = np.full(n, -99.0)
    for i in range(len(a)):
        if ra[i]:
            best[env_i[i]] = max(best[env_i[i]], za[i])
    say(f"  envs with some on-line step reaching the plate in the air (aero): {(best > -99).sum()}/{n}"
        + (f", best plate z median {np.median(best[best > -99]):.2f} m" if (best > -99).any() else ""))


if __name__ == "__main__":
    main()
    app.close()
