# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--reset_normalizer", action="store_true", default=False,
                    help="refit the observation normalizers from the new task's data (starting a new stage)")
parser.add_argument("--max_noise_std", type=float, default=None,
                    help="cap the action noise std at this (overrides the agent cfg's max_noise_std)")
parser.add_argument("--critic_warmup", type=int, default=0,
                    help="first N PPO updates train the critic only (actor and noise frozen)")
parser.add_argument("--zero_actor_head", action="store_true", default=False,
                    help="zero the actor's output layer after loading: the policy starts as zero actions (the PD reference)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aibaseball.tasks  # noqa: F401, E402  (registers AIB-TeeBatting-v0)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.reset_normalizer and agent_cfg.empirical_normalization:
            # rsl_rl stops updating the normalizers after 1e8 samples: a checkpoint from an earlier stage keeps that
            # stage's statistics (1a: phase always 0, fingers still -> in 1b the phase came in at 88 and the finger
            # velocities at 30-70 normalized units). count 0: the next batch replaces them, then they keep learning
            for nm in (runner.obs_normalizer, runner.critic_obs_normalizer):
                nm.count.zero_()
            print("[INFO]: observation normalizers reset (refit on this task)")
        if args_cli.zero_actor_head:
            # the previous stage's policy fights a new PD reference (1a 700 pulled toward the set position against
            # the 1b1c weight shift: 64/64 falls vs 3/64 with zero actions); keep its features, start at zero actions
            head = [m for m in runner.alg.actor_critic.actor if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
            print("[INFO]: actor output layer zeroed (starts as zero actions)")
    if args_cli.critic_warmup > 0:
        # a new stage's critic (carried over from the last one) has no idea of the new returns: its first advantages
        # are noise, and the first PPO steps took the zero-action policy (1b1g: return 286 vs 37 after training) from
        # 0% to 85% falls in 11 iterations. Train the critic alone first.
        ac0 = runner.alg.actor_critic
        actor_params = list(ac0.actor.parameters()) + [ac0.std]
        update0 = runner.alg.update
        lr0 = runner.alg.learning_rate
        n_done = [0]

        def update_warm(*a, **k):
            frozen = n_done[0] < args_cli.critic_warmup
            for prm in actor_params:
                prm.requires_grad_(not frozen)
            out = update0(*a, **k)
            n_done[0] += 1
            if frozen:  # the adaptive schedule reads KL 0 while frozen and would raise the learning rate to 1e-2
                runner.alg.learning_rate = lr0
                for g in runner.alg.optimizer.param_groups:
                    g["lr"] = lr0
            if n_done[0] == args_cli.critic_warmup:
                print(f"[INFO]: critic warm-up done ({args_cli.critic_warmup} updates), actor training", flush=True)
            return out

        runner.alg.update = update_warm
        print(f"[INFO]: critic warm-up: actor frozen for {args_cli.critic_warmup} updates", flush=True)
    # cap the action noise (agent cfg max_noise_std): clamp the policy std at load and after every PPO update
    max_std = args_cli.max_noise_std or getattr(agent_cfg, "max_noise_std", None)
    if max_std:
        ac = runner.alg.actor_critic
        update = runner.alg.update

        def update_capped(*a, **k):
            out = update(*a, **k)
            with torch.no_grad():
                ac.std.clamp_(max=max_std)
            return out

        runner.alg.update = update_capped
        with torch.no_grad():
            ac.std.clamp_(max=max_std)
        print(f"[INFO] action noise std capped at {max_std}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
