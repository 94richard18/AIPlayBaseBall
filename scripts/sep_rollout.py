"""SEP rollout check (plan docs/plans/fastball_restart.md 5-17): pay statistics of the separation term on real rollouts.

    python scripts/sep_rollout.py --headless --checkpoint <7600> --arm_teacher <7600> [--envs 128] [--episodes 2]

Training distribution of the SEP task (its own start noise / RSI share), deterministic policy mean. Per finished
episode (StageEnv.s_last): separation at the valid stride event, event flag, pay mask, SEP pay, speed-term pay.
Reports pay rate / paid mean / all-episode mean / p10-50-90; the only pre-registered stop is SEP pay rate < 5%.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100S1ESEP-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--arm_teacher", type=str, default=None)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--episodes", type=float, default=2.0, help="run length in episode lengths")
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


def stats(x):
    x = np.asarray(x)
    if not len(x):
        return "-"
    return f"mean {x.mean():.2f}, p10/50/90 {np.percentile(x, 10):.2f}/{np.percentile(x, 50):.2f}/{np.percentile(x, 90):.2f} (n {len(x)})"


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    w = RslRlVecEnvWrapper(env)
    ag = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    r = OnPolicyRunner(w, ag.to_dict(), log_dir=None, device=ag.device)
    r.load(args.checkpoint)
    r.eval_mode()
    if args.arm_teacher:
        from aibaseball.rl.arm_teacher import attach_arm_teacher

        attach_arm_teacher(r.alg, args.arm_teacher, list(base._arm_cols) + [29 + base.cfg.release_action_col], base.device)
    pol = r.get_inference_policy(device=base.device)
    T = int(round(args.episodes * cfg.episode_length_s / base.step_dt))
    E = {k: [] for k in base.s_last}
    prev = base.s_last_n.clone()
    with torch.inference_mode():
        obs, _ = w.reset()
        prev = base.s_last_n.clone()
        for _ in range(T):
            obs, _, _, _ = w.step(pol(obs))
            done = (base.s_last_n > prev).cpu().numpy()
            prev = base.s_last_n.clone()
            if done.any():
                for k, v in base.s_last.items():
                    E[k].extend(v.cpu().numpy()[done].tolist())
    E = {k: np.asarray(v) for k, v in E.items()}
    n = len(E["sep"])
    say(f"== sep_rollout: {args.checkpoint} | {args.task}, teacher {args.arm_teacher}, {args.envs} envs, {T} steps; "
        f"w_sep {cfg.w_stage_sep}, sep {cfg.sep_lo_deg} -> {cfg.sep_full_deg} deg; rsi_prob {cfg.rsi_prob}")
    if not n:
        say("[sep] no finished episodes")
        return
    ev = E["sep_set"] > 0.5
    pay = E["pay"] > 0.5
    sp = E["sep_pay"] > 0
    say(f"[sep] episodes {n}: set starts {int((E['set_start'] > 0.5).sum())}, valid stride event {int(ev.sum())}, "
        f"pay mask {int(pay.sum())} ({pay.mean():.1%})")
    say(f"[sep] separation at the event (deg): {stats(E['sep'][ev])}; event time - release_ref (s): "
        f"{stats(E['land_t'][ev] - float(base.release_ref))}")
    say(f"[sep] SEP pay: rate {sp.mean():.1%} ({int(sp.sum())}/{n}), paid mean {E['sep_pay'][sp].mean() if sp.any() else 0:.2f}, "
        f"all-episode mean {E['sep_pay'].mean():.2f}, all {stats(E['sep_pay'])}")
    vp = E["speed_pay"] > 0
    say(f"[sep] speed-term pay: rate {vp.mean():.1%}, paid mean {E['speed_pay'][vp].mean() if vp.any() else 0:.2f}, "
        f"all-episode mean {E['speed_pay'].mean():.2f}, all {stats(E['speed_pay'])}")
    say(f"[sep] verdict: " + ("STOP (SEP pay rate < 5%)" if sp.mean() < 0.05 else "go (SEP pay rate >= 5%)"))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        say(traceback.format_exc())
        raise
    finally:
        app.close()
