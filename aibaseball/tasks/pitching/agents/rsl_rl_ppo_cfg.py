from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PitchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64  # 0.16 s at 400 Hz
    max_iterations = 3000
    save_interval = 50  # a checkpoint every 50 iterations (videos every 150)
    experiment_name = "aib1_pitch"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,  # 400 Hz control: same ~0.5 s effective horizon as 0.99 at 200 Hz
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PitchStand3PPORunnerCfg(PitchPPORunnerCfg):
    """No entropy bonus, action noise capped (2026-10-12): from scratch the noise std grew 0.35 -> 2.23 (45 actions at
    400 Hz: the PD and the inertia average the noise out, so the bonus cost nothing). Training still threw (1% drops),
    the noise-free policy dropped 60/64 - its mean action was never trained."""

    max_noise_std = 0.5  # scripts/train.py clamps the policy std after every update (and at load)

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.0


@configclass
class PitchStand6PPORunnerCfg(PitchStand3PPORunnerCfg):
    """Noise cap 0.5 -> 0.3 (2026-10-13). v9 1650: with the training noise 17/64 still up and the release at 0.454 s,
    the noise-free policy released at 0.253 s, before its stride (64/64 no pitch); the mean action again drifted away
    from what the noisy rollouts train."""

    max_noise_std = 0.3


@configclass
class PitchBalancePPORunnerCfg(PitchPPORunnerCfg):
    experiment_name = "aib1_pitch_balance"
