"""Consecutive episodes with real resets (dones on, as in training): is the eval's first reset (fresh scene, all envs at
once) the same start as a training reset (in place, after the previous episode ended - often mid-fall)? Scores every
ended episode by its index per env (1 = the eval's start, 2+ = training-like starts), and compares the state at the
start: the first observation per dimension, SEA motor vs joint, previous actions, ball vs hand on the first step.

    python scripts/diag_reset_episodes.py --headless --task AIB-Stage1dK70GW2M-v0 --checkpoint <model.pt>
        [--envs 128] [--episodes 3] [--noise 0.02] [--stochastic]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2M-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=3, help="run this many full-length episodes' worth of steps")
parser.add_argument("--noise", type=float, default=0.02)
parser.add_argument("--stochastic", action="store_true")
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


def say(*a):
    print(*a, flush=True)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    for k, v in (("landed_start_prob", 0.0), ("rsi_prob", 0.0), ("assist_force_n", 0.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.start_joint_noise, cfg.start_noise_ramp_steps = args.noise, 0
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    base.s_desync = False  # StageEnv cuts every env's first episode at a random length (desync for training): keep it full
    n = base.num_envs
    ep_idx = np.ones(n, dtype=int)  # index of the episode each env is in (1 = the first, from the fresh scene)
    rows = []  # ended episodes: (index, at rest & thrown, fell, end time s, dropped, timed out)
    drops = []  # dropped balls: release speed, release vx, release reference time, episode time at the drop
    first = []  # first step of each episode: (index, ball-hand dist, max |SEA motor - joint|, |prev_actions|)
    orig_rew = base._get_rewards

    def rew_hook():
        r = orig_rew()
        d = base.robot.data
        _, _, _, _, ok = base._stand_state()
        if base.cfg.stage == "1b":
            ok = ok & base.s_landed
        v = base.rel_vel.norm(dim=-1)
        thrown = base.released & (v >= 10.0) & (base.rel_ref_t >= 0.9)
        to, term = base.reset_time_outs, base.reset_terminated
        done = (to | term).cpu().numpy()
        el = base.episode_length_buf.cpu().numpy()
        if done.any():
            win = (to & ok & ~base.fallen & thrown).cpu().numpy()
            fell = base.fallen.cpu().numpy()
            drop = base.s_drop.cpu().numpy() if hasattr(base, "s_drop") else np.zeros(n, dtype=bool)
            vn, vx, rt = v.cpu().numpy(), base.rel_vel[:, 0].cpu().numpy(), base.rel_ref_t.cpu().numpy()
            for i in np.nonzero(done)[0]:
                rows.append((ep_idx[i], win[i], fell[i], el[i] * base.step_dt, drop[i], bool(to[i])))
                if drop[i]:
                    drops.append((vn[i], vx[i], rt[i], el[i] * base.step_dt))
        st = el == 1  # first step of an episode: the reset's writes are visible now
        if st.any():
            hand = d.body_link_pos_w[:, base.hand_id]
            bh = (base.ball.data.root_pos_w - hand).norm(dim=-1).cpu().numpy()
            sea = torch.zeros(n, device=base.device)
            for a in base._seas():
                jp = d.joint_pos[:, a.joint_indices] if not isinstance(a.joint_indices, slice) else d.joint_pos
                sea = torch.maximum(sea, (a.motor_pos - jp).abs().max(-1).values)
            sea = sea.cpu().numpy()
            pa = base.prev_actions.norm(dim=-1).cpu().numpy()
            for i in np.nonzero(st)[0]:
                first.append((ep_idx[i], bh[i], sea[i], pa[i]))
        ep_idx[done] += 1
        return r
    base._get_rewards = rew_hook
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    pol = r.get_inference_policy(device=base.device)
    if args.stochastic:
        pol = lambda x: r.alg.actor_critic.act(r.obs_normalizer(x))  # noqa: E731
    T = args.episodes * int(round(cfg.episode_length_s / base.step_dt))
    obs0 = {}  # first observation of each episode, by index (1, 2, 3+)
    with torch.inference_mode():
        obs, _ = w.reset()
        obs0.setdefault(1, []).append(obs.cpu().numpy())
        for _ in range(T):
            obs, _, dones, _ = w.step(pol(obs))
            dn = dones.bool().cpu().numpy()
            if dn.any():  # the returned obs of a done env is its next episode's first observation
                for i in np.nonzero(dn)[0]:
                    obs0.setdefault(min(ep_idx[i], 3), []).append(obs[i:i + 1].cpu().numpy())
    sc = cfg.scene
    say(f"  scene: env_spacing {sc.env_spacing} m, filter_collisions {getattr(sc, 'filter_collisions', '?')}, "
        f"replicate_physics {sc.replicate_physics}, num_envs {n}")
    say(f"== diag_reset_episodes: {args.checkpoint} | {args.task}, {n} envs, start noise {args.noise}, "
        f"{'stochastic' if args.stochastic else 'deterministic'}, {T} steps ({args.episodes} x {cfg.episode_length_s} s)")
    a = np.array(rows, dtype=float)
    for k in sorted(set(a[:, 0].astype(int))):
        m = a[:, 0] == k
        e = a[m]
        fell = e[:, 2] > 0
        say(f"  episode {k}: ended {m.sum()}, at rest & thrown {int(e[:, 1].sum())} ({e[:, 1].mean():.0%}), falls "
            f"{int(fell.sum())} ({fell.mean():.0%})" + (f", fall time median {np.median(e[fell, 3]):.2f} s" if fell.any() else "")
            + f", drops {int(e[:, 4].sum())}, timed out {int(e[:, 5].sum())}")
    if drops:
        dr = np.array(drops)
        say(f"  drops ({len(dr)}): release speed p10/50/90 {np.percentile(dr[:, 0], 10):.1f}/{np.median(dr[:, 0]):.1f}/"
            f"{np.percentile(dr[:, 0], 90):.1f} m/s, < {cfg.drop_min_speed} m/s {(dr[:, 0] < cfg.drop_min_speed).sum()}, "
            f"release before {cfg.drop_min_release_t} s {(dr[:, 2] < cfg.drop_min_release_t).sum()}, vx <= 0 "
            f"{(dr[:, 1] <= 0).sum()}; release time median {np.median(dr[:, 2]):.2f} s, ended at median {np.median(dr[:, 3]):.2f} s")
    f = np.array(first, dtype=float)
    for k in sorted(set(f[:, 0].astype(int))):
        m = f[:, 0] == k
        say(f"  first step of episode {k} ({m.sum()}): ball-hand {np.median(f[m, 1]):.3f} m (max {f[m, 1].max():.3f}), "
            f"max |SEA motor - joint| median {np.median(f[m, 2]):.4f} rad (max {f[m, 2].max():.4f}), "
            f"|prev_actions| median {np.median(f[m, 3]):.3f}")
    if 1 in obs0 and 2 in obs0:
        o1, o2 = np.concatenate(obs0[1]), np.concatenate(obs0[2] + obs0.get(3, []))
        sd = np.sqrt(0.5 * (o1.var(0) + o2.var(0))) + 1e-6
        z = np.abs(o1.mean(0) - o2.mean(0)) / sd
        raw = np.abs(o1.mean(0) - o2.mean(0))
        top = np.argsort(-raw)[:8]
        say(f"  first observation, episode 1 ({len(o1)}) vs 2+ ({len(o2)}): max |mean diff| {raw.max():.4f}; "
            "largest dims (dim: ep1 / ep2+ mean, z): " + ", ".join(f"{j}: {o1[:, j].mean():+.3f}/{o2[:, j].mean():+.3f} "
                                                              f"({z[j]:.1f})" for j in top))


if __name__ == "__main__":
    main()
    app.close()
