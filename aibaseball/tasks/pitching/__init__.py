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

gym.register(
    id="AIB-PitchStand2-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand3-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand3PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand4-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand3PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand5-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand3PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand6-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand6PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand7-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand6PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand8-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand6PPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchStand9-v0",
    entry_point=f"{__name__}.pitch_env:PitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:PitchStand9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PitchStand6PPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1a-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)
