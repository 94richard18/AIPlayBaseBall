"""ONE OUTS joint scene: pitcher and batter in one env (user 2026-10-28)."""

import gymnasium as gym

from aibaseball.tasks.pitching import agents

gym.register(
    id="AIB-OneOutsJoint-v0",
    entry_point=f"{__name__}.joint_env:JointPitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_env_cfg:JointPitchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)
