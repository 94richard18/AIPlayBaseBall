from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PitchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64  # 0.16 s at 400 Hz
    max_iterations = 3000
    save_interval = 100
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
