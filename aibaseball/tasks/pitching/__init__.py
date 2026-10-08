import gymnasium as gym

from . import agents

gym.register(
    id="AIB-Pitch-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Pitch-Play-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchElastic-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchElasticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchElastic-Play-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchElasticPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchBalance-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchBalancePPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchHandover-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchHandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStandFirst-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStandFirstEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchSpeed-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchSpeedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)
