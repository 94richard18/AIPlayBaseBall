"""Record landed stances from a stage-1b policy: for each env, the first step >= 0.1 s after its stride landed with
both feet down, COM < --max_v m/s and not fallen. Saved (root relative to the env origin, joint angles) for
StageEnv landed_start_prob (half the episodes start in a stance the robot really reaches).

    python scripts/capture_stage_states.py --headless --task AIB-Stage1b3-v0 --checkpoint <model.pt> \
        --out logs/stage_states/s1b3_landed.pt [--envs 512]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="AIB-Stage1b3-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--envs", type=int, default=512)
parser.add_argument("--max_v", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import aibaseball.tasks  # noqa: F401, E402


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    cfg.landed_start_prob = 0.0
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
    pol = lambda x: r.alg.actor_critic.act(r.obs_normalizer(x))  # noqa: E731  (noise: varied stances)
    n, dev = base.num_envs, base.device
    got = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
    root = torch.zeros(n, 13, device=dev)
    jp = torch.zeros(n, base.robot.num_joints, device=dev)
    T = int(round(cfg.episode_length_s / base.step_dt)) - 2
    with torch.inference_mode():
        obs, _ = w.reset()
        for _ in range(T):
            obs, _, _, _ = w.step(pol(obs))
            ever_fell |= base.fallen
            both, speed, tilt, head_z, ok = base._stand_state()
            since = (base.episode_length_buf - base.s_land_step).float() * base.step_dt
            take = (~got & ~ever_fell & base.s_landed & (since >= 0.1) & both & (speed < args.max_v)
                    & (head_z > cfg.stand_min_head_z))
            if take.any():
                d = base.robot.data
                st = d.root_state_w.clone()
                st[:, :3] -= base.scene.env_origins
                root = torch.where(take.unsqueeze(-1), st, root)
                jp = torch.where(take.unsqueeze(-1), d.joint_pos, jp)
                got |= take
    k = int(got.sum())
    print(f"[capture] {k}/{n} landed stances (COM < {args.max_v} m/s, both feet down, head up)", flush=True)
    if k:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"root": root[got].cpu(), "joint_pos": jp[got].cpu(), "checkpoint": args.checkpoint}, args.out)
        print(f"[capture] wrote {args.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
