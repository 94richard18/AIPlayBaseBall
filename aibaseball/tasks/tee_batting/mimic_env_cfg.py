from __future__ import annotations

import glob
import os

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .tee_batting_env_cfg import TeeBattingEnvCfg

def _two_hand_clips(max_grip_gap: float = 0.010) -> list:
    """Retargeted swings that keep both hands on the bat (AIB-1's grip is a closed loop; swings whose
    athlete lets go with the top hand in the follow-through are left out)."""
    import numpy as np

    out = []
    for f in sorted(glob.glob(os.path.join(MOTION_DIR, "obp_*.npz"))):
        if float(np.load(f)["grip_gap"].max()) <= max_grip_gap:
            out.append(f)
    return out


MOTION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "motions")


@configclass
class TeeBattingMimicEnvCfg(TeeBattingEnvCfg):
    motion_file: str = os.path.abspath(os.path.join(MOTION_DIR, "obp_42-5.npz"))
    motion_files: list = []  # several clips (MotionLib); empty -> [motion_file]
    tee_noise_ramp_steps: int = 0  # control steps to grow the tee randomisation to its full range
    speed: float = 1.0  # reference playback speed (>1: same swing, faster than the athlete)
    speed_end: float = 1.0  # curriculum target speed
    speed_ramp_steps: int = 0  # control steps (common counter) to ramp speed -> speed_end; 0 = fixed
    episode_length_s = 2.0  # capped by the end of the reference

    # base obs (110) + phase (1) + next-ref joint delta (29) + ref root offset (3) + ref bat-tip offset (3)
    observation_space = 146

    # residual actions on top of the reference: target = q_ref + residual_scale * action_scale * a
    residual_scale: float = 0.5
    vel_feedforward: float = 1.0  # PD velocity target = this x reference joint velocity

    # reference state initialisation
    rsi_prob: float = 0.8
    rsi_margin_before_contact: float = 0.10

    # the ball sits where the reference sweet spot meets it
    tee_offset_z: float = 0.0  # raise the ball above the reference sweet-spot path (stage 2: launch angle)
    tee_xy_noise: float = 0.0
    tee_height_noise: float = 0.0
    joint_pos_noise: float = 0.0

    # tracking reward (per control step, max = w_track)
    w_track: float = 1.0
    w_pose: float = 0.45
    w_vel: float = 0.10
    w_key: float = 0.30
    w_root: float = 0.15
    sigma_pose: float = 0.25  # rad RMS
    sigma_vel: float = 5.0  # rad/s RMS
    sigma_key: float = 0.10  # m RMS
    sigma_root: float = 0.10  # m
    sigma_rot: float = 0.30  # rad
    # contact guidance near the reference contact time
    w_contact: float = 2.0
    sigma_contact: float = 0.08  # m
    contact_window_s: float = 0.04
    max_key_err: float = 0.45  # terminate when key bodies drift this far (m RMS)

    # task shaping: tracking replaces the hand-made swing shaping
    w_bat_speed: float = 0.0
    w_approach: float = 0.0


@configclass
class TeeBattingMimicPlayEnvCfg(TeeBattingMimicEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class TeeBattingMimicPowerEnvCfg(TeeBattingMimicEnvCfg):
    """Stage 2: hitting is reliable -> reward power; play the swing faster; ball slightly above the bat path."""

    speed: float = 1.15  # resumed from the 1.0 -> 1.15 part of the curriculum
    speed_end: float = 1.45
    speed_ramp_steps: int = 32 * 1000  # ~1000 PPO iterations
    tee_offset_z: float = 0.025
    w_exit_velo: float = 20.0
    w_distance: float = 40.0
    w_success: float = 40.0
    w_launch: float = 8.0
    sigma_key: float = 0.15
    sigma_pose: float = 0.35


@configclass
class TeeBattingMimicPowerPlayEnvCfg(TeeBattingMimicPowerEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    speed: float = 1.45
    speed_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class TeeBattingMimicLaunchEnvCfg(TeeBattingMimicPowerEnvCfg):
    """Stage 3: exit velocity is there; raise the launch angle (ball higher above the bat path)."""

    speed_end: float = 1.45
    tee_offset_z: float = 0.04
    w_launch: float = 20.0
    launch_sigma_deg: float = 12.0  # sharp signal around the 28 deg optimum
    w_backspin: float = 10.0  # topspin kills carry (a 23 deg / 122 mph topspin drive only carried 108 m)
    rsi_prob: float = 0.3  # more full swings from the stance (what the evaluation/showcase runs)
    speed: float = 1.45
    speed_ramp_steps: int = 0


@configclass
class TeeBattingMimicLaunchPlayEnvCfg(TeeBattingMimicLaunchEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    speed: float = 1.45
    speed_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class TeeBattingMultiEnvCfg(TeeBattingMimicLaunchEnvCfg):
    """Step A: twelve OBP swings (different contact heights / locations) + randomised tee position."""

    motion_files: list = _two_hand_clips()
    tee_xy_noise: float = 0.10  # +-10 cm in the horizontal plane (around each clip's contact point)
    tee_height_noise: float = 0.08  # +-8 cm in height
    tee_noise_ramp_steps: int = 32 * 800  # ~800 PPO iterations
    rsi_prob: float = 0.3


@configclass
class TeeBattingMultiPlayEnvCfg(TeeBattingMultiEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    tee_noise_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class TeeBattingPerceptionEnvCfg(TeeBattingMultiEnvCfg):
    """Step B: the batter sees the ball with estimation error, noise and latency (rewards use the true ball)."""

    ball_obs_bias_std: float = 0.02  # 2 cm per-episode error
    ball_obs_noise_std: float = 0.01  # 1 cm per-step noise
    ball_obs_delay_s: float = 0.10  # 100 ms visuomotor delay
    tee_noise_ramp_steps: int = 0  # step A already reached the full tee randomisation


@configclass
class TeeBattingPerceptionPlayEnvCfg(TeeBattingPerceptionEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
