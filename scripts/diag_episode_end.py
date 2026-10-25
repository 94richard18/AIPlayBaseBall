"""How do batting episodes end in the TRAINING env (not play)? One episode per env with a deterministic policy; for the
episodes with contact, the share that end by a fall (pelvis / tilt), by losing the reference ('lost'), by the end of the
follow-through window, or by the time limit / reference end. Used to check that a training setup can see post-contact falls.

    python scripts/diag_episode_end.py --headless --checkpoint <model.pt> --task AIB-PitchedBatting-Human12-v0 --envs 1024
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="AIB-PitchedBatting-Human12-v0")
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--aim_offset_z", type=float, default=None)
parser.add_argument("--timing_noise", type=float, default=None)
parser.add_argument("--tag", type=str, default="h12")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402
from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingMimicPPORunnerCfg  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, num_envs=args.envs)
    cfg.seed = args.seed
    cfg.pitch_ramp_steps = 0  # final pitch range / timing noise from the start
    if args.aim_offset_z is not None:
        cfg.aim_offset_z = args.aim_offset_z
    if args.timing_noise is not None:
        cfg.timing_noise_s = args.timing_noise
    env = gym.make(args.task, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    agent = TeeBattingMimicPPORunnerCfg()
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    n, dev = base.num_envs, base.device
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    hit, fell, lost, follow, tout = (torch.zeros(n, dtype=torch.bool, device=dev) for _ in range(5))
    t_end = torch.zeros(n, device=dev)
    obs, _ = wrapped.reset()
    follow_steps = int(round(cfg.follow_through_s / base.step_dt))
    for _ in range(int(cfg.episode_length_s / base.step_dt) + 5):
        h = base.hit.clone()
        hs = base.hit_step.clone()
        el = base.episode_length_buf.clone()
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        d = dones.bool() & ~finished
        if d.any():
            lo = getattr(base, "lost", torch.zeros_like(d))
            real_fall = base.fallen & ~(lo & base.hit) if cfg.lost_after_hit_is_fall else base.fallen
            hit |= d & (h | base.hit)
            fell |= d & real_fall
            lost |= d & lo & ~real_fall
            follow |= d & h & (el + 1 - hs >= follow_steps) & ~real_fall & ~lo
            tout |= d & base.reset_time_outs & ~real_fall & ~lo
            t_end = torch.where(d, (el + 1).float() * base.step_dt, t_end)
            finished |= d
        if finished.all():
            break
    m = hit & finished
    k = int(m.sum())
    out = dict(checkpoint=args.checkpoint, task=args.task, episodes=int(finished.sum()), hits=k,
               fell=int((m & fell).sum()), lost=int((m & lost).sum()), follow_end=int((m & follow).sum()),
               time_out=int((m & tout).sum()), episode_s=float(cfg.episode_length_s), follow_through_s=float(cfg.follow_through_s))
    out["lost_plus_fell_share"] = (out["fell"] + out["lost"]) / max(k, 1)
    print("[end] " + json.dumps(out), flush=True)
    with open(os.path.join(ROOT, "logs", f"diag_episode_end_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
