from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aibaseball.robot.aib1_cfg import make_aib1_cfg
from aibaseball.tasks.tee_batting.tee_batting_env_cfg import TeeBattingEnvCfg


@configclass
class RecoverEnvCfg(TeeBattingEnvCfg):
    """Human25-R (planning 14, 10-29, user "直接做新機制"): a separate post-contact recovery policy. Episodes start from
    the stage-0 bank (the frozen swing's state at BatterSide's hand-off trigger) and last 2 s; the only job is to stand.
    Action = the swing policy's PD parameterization (same 29 joints, action_scale, residual_scale 0.5, gains): targets =
    anchor + residual_scale * action_scale * a, where the anchor blends (smoothstep, anchor_blend_s) from the PD targets
    at the hand-off to the batting stance (default pose) - a = 0 is the open-loop after_done "recover" (100% on the
    ground in fall diag 3); the policy learns the feedback on top. No ball, no reference, no swing terms."""

    robot: ArticulationCfg = make_aib1_cfg(human_strength=True)  # effort via AIB_EFFORT_SCALE (1.5 for the delivery)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    episode_length_s = 2.0
    action_space = 29
    extra_actions: int = 0
    # joint pos 29 + joint vel 29 + gravity 3 + root lin / ang vel 6 + pelvis height 1 + anchor - q 29 + prev action 29 + t 1
    observation_space = 127
    residual_scale: float = 0.5
    anchor_blend_s: float = 0.8

    bank_path: str = "logs/recover_bank_s0.npz"
    bank_vel_noise: float = 0.05  # joint velocities x (1 +- this)
    bank_root_vel_noise: float = 0.1  # root linear velocity +- m/s per axis

    ground_pelvis_z: float = 0.35  # "on the ground" (14's film label): terminate, terminal penalty
    max_tilt_deg: float = 75.0
    terminate: bool = True  # eval: run the full 2 s and measure

    w_alive: float = 1.0
    w_height: float = 1.0
    sigma_height: float = 0.12
    w_upright: float = 1.0
    sigma_tilt: float = 0.35  # rad
    w_still: float = 1.0  # after still_after_s: root lin / ang velocity and joint velocity small
    still_after_s: float = 0.5
    sigma_lin: float = 0.4
    sigma_ang: float = 1.0
    sigma_jv: float = 3.0  # rad/s, rms over the body joints
    w_feet: float = 0.5  # both feet down (lift above the episode's lowest foot height)
    sigma_foot: float = 0.05
    w_com: float = 0.5  # whole-body COM over the feet segment
    sigma_com: float = 0.15
    w_action_rate: float = 0.01
    w_ground: float = 50.0  # terminal: on the ground / tilted over


@configclass
class RecoverEvalEnvCfg(RecoverEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)
    bank_path: str = "logs/recover_bank_s6.npz"
    bank_vel_noise: float = 0.0
    bank_root_vel_noise: float = 0.0
    terminate: bool = False
