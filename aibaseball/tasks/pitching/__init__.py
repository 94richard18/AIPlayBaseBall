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

gym.register(
    id="AIB-Stage1dS75-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dS75EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dS-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dSEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dS60-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dS60EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dS70-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dS70EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dS80-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dS80EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dS90-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dS90EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dN70-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dN70EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dN80-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dN80EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dN90-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dN90EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dN-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dNEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dNR70-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dNR70EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dNR-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dNREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dP70-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dP70EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dP80-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dP80EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dP90-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dP90EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dP-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK80-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK80EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK90-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK90EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70G-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GI-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GIEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GR-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GB-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW15-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2M-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBAS-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASV-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVR-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVR9-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVR9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1WF-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1WFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SY110WF-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SY110WFEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SY110-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SY110EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SY110E40-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SY110E40EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYE10-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYE10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYE20-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYE20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYE40-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYE40EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYE50-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYE50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SY-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3SY-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3SYEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3N3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3N3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3C-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3CEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3B1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3B1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3B2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3B2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3F-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3FEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1D2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1D2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1C1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1C1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1C2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1C2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1W-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1WEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1S1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1S1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1S2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1S2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1S3-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1S3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRE15-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRE15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRE20-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRE20EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRA-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRB-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBAR-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBAREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBA-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MB-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MH-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GIW2M-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GIW2MEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dKEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW130-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW130EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110REnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW100R5-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW100R5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5N1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5N1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H65-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H65EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H8-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW1-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW1Y-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW1YEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5WT15-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5WT15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5WT12-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5WT12EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5S2F15-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5S2F15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5S2F15WT15-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5S2F15WT15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5HT-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5HTEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5HTE-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5HTEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PL-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3C-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CP-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CPC-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100P3CPCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100Y7-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLW100Y7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Stage1dK70GW2MBASVRT1SYW110R5H5HTEL-v0",
    entry_point=f"{__name__}.stage_env:StageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitch_env_cfg:Stage1dK70GW2MBASVRT1SYW110R5H5HTELEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StageSlowPPORunnerCfg",
    },
)
