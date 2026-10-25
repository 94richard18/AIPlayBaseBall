import gymnasium as gym

from . import agents

gym.register(
    id="AIB-TeeBatting-v0",
    entry_point=f"{__name__}.tee_batting_env:TeeBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tee_batting_env_cfg:TeeBattingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Play-v0",
    entry_point=f"{__name__}.tee_batting_env:TeeBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tee_batting_env_cfg:TeeBattingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Mimic-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-MimicPower-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicPowerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-MimicPower-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicPowerPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-MimicLaunch-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicLaunchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-MimicLaunch-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMimicLaunchPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Multi-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMultiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Multi-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingMultiPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Perception-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingPerceptionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Perception-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingPerceptionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Play-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Human-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingHumanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-TeeBatting-Human-Play-v0",
    entry_point=f"{__name__}.mimic_env:TeeBattingMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:TeeBattingHumanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHumanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human2-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human3-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human4-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human5-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human6-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human7-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human8-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human9-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHuman9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)

gym.register(
    id="AIB-PitchedBatting-Human-Play-v0",
    entry_point=f"{__name__}.pitched_env:PitchedBattingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pitched_env_cfg:PitchedBattingHumanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TeeBattingMimicPPORunnerCfg",
    },
)
