from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from aibaseball.tasks.tee_batting.agents.rsl_rl_ppo_cfg import TeeBattingPPORunnerCfg


@configclass
class RecoverPPORunnerCfg(TeeBattingPPORunnerCfg):
    """Human25-R: a fresh actor (new inputs); residual PD actions, so start near a = 0 (the open-loop blend)."""

    experiment_name = "aib1_recover"
    max_iterations = 1000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
