import gymnasium as gym

from . import agents

gym.register(
    id="AIB-Recover-v0",
    entry_point=f"{__name__}.recover_env:RecoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recover_env_cfg:RecoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RecoverPPORunnerCfg",
    },
)

gym.register(
    id="AIB-Recover-Eval-v0",
    entry_point=f"{__name__}.recover_env:RecoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recover_env_cfg:RecoverEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RecoverPPORunnerCfg",
    },
)
