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

gym.register(
    id="AIB-Stage1b4-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b5-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd50-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd100-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd100EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1b-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1c-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1d-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1e-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1eEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1f-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1fEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1g-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1gEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StagePPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1b1h-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1b1gEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1c-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd50b-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd50bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd50a-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd50aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd100r-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd100rEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd100rf-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd100rfEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd100rf2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd100rf2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1bFwd100rf3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1bFwd100rf3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1d2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1d3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d4-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1d4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d5-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1d5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1d6-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1d6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF25-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF25EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF50-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF25r-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF25rEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageLongPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF25s-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF25rEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF50s-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dF75s-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dF75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dFs-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dA25-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dA25EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dA50-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dA50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dA75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dA75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dA-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dD75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dD75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dD-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dDEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dB75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dB75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dB-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dR75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dR75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dR-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)
