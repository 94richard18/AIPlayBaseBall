"""Is the finish-COM term (w_finish_com) live, and with the right sign? Measures it inside the env's own reward.

    python scripts/check_finish_term.py --headless --task AIB-Stage1dK70GW2MBASVRT1D3F-v0 --checkpoint <model.pt> [--envs 32]

After every step the reward is recomputed with w_finish_com = 0 and with its configured value (same state); the difference
is the term actually paid. Per episode: the term's sum, whether the episode entered the window (released, not fallen at
release + t0), and for a few episodes f_x / f_y at release + 0.3 s next to COM - lead foot. (The double reward call
also moves the stage statistics EMA - this is a diagnostic run only.)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=32)
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


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0), ("start_joint_noise", 0.02),
                 ("start_noise_ramp_steps", 0)):
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
    n, c = base.num_envs, base.cfg
    wfin = c.w_finish_com
    T = int(round(c.episode_length_s / base.step_dt)) - 2
    term = np.zeros(n)
    entered = np.zeros(n, dtype=bool)
    probe = np.full((n, 4), np.nan)  # at release + 0.3 s: f_x, f_y, dx, dy
    with torch.inference_mode():
        obs, _ = w.reset()
        for k in range(T):
            obs, _, _, _ = w.step(pol(obs))
            c.w_finish_com = 0.0
            r0 = base._get_rewards().clone()
            c.w_finish_com = wfin
            r1 = base._get_rewards().clone()
            term += (r1 - r0).cpu().numpy()
            since = ((base.episode_length_buf - base.release_step).float() * base.step_dt).cpu().numpy()
            rel = base.released.cpu().numpy()
            fell = base.fallen.cpu().numpy()
            entered |= rel & ~fell & (since >= c.finish_com_t0) & (since < c.finish_com_t0 + base.step_dt * 1.5)
            at = rel & (np.abs(since - 0.3) < base.step_dt / 2)
            if at.any():
                com, _ = base._com()
                lead = base.robot.data.body_link_pos_w[:, base.foot_ids[0]]
                dx, dy = (com[:, 0] - lead[:, 0]).cpu().numpy(), (com[:, 1] - lead[:, 1]).cpu().numpy()
                fx = np.exp(-((np.maximum(dx + c.finish_com_margin, 0) / c.finish_com_sigma) ** 2))
                fy = np.exp(-((np.maximum(dy + c.finish_com_margin, 0) / c.finish_com_sigma) ** 2))
                probe[at] = np.stack([fx, fy, dx, dy], -1)[at]
        ok = base._stand_state()[-1].cpu().numpy()
    print(f"== check_finish_term: {args.checkpoint} | {args.task}, {n} envs, start noise 0.02, w_finish_com {wfin}", flush=True)
    print(f"  term paid per episode: mean {term.mean():.2f}, median {np.median(term):.2f}, max {term.max():.2f}, min "
          f"{term.min():.3f}; entered the window {entered.sum()}/{n}; at rest at the end {ok.sum()}/{n} "
          f"(term mean at rest {term[ok].mean() if ok.any() else float('nan'):.2f}, not at rest "
          f"{term[~ok].mean() if (~ok).any() else float('nan'):.2f})", flush=True)
    for i in np.nonzero(~np.isnan(probe[:, 0]))[0][:6]:
        print(f"  env {i}: at release + 0.3 s dx {probe[i, 2] * 100:+.1f} cm, dy {probe[i, 3] * 100:+.1f} cm -> f_x "
              f"{probe[i, 0]:.2f}, f_y {probe[i, 1]:.2f}; episode term {term[i]:.2f}, at rest {bool(ok[i])}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
