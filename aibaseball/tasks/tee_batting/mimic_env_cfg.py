from __future__ import annotations

import glob
import os

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aibaseball.robot.aib1_cfg import make_aib1_cfg

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
    # bat speed shaping: in the last ss_speed_window_s of reference time before contact (until the hit), per step
    # w_ss_speed * (sweet-spot speed along the reference sweet spot's velocity, capped at its speed) / the reference's
    # speed at contact (the reference ramps ~6 -> 31 m/s over the window, so the sum follows the speed profile)
    w_ss_speed: float = 0.0
    ss_speed_window_s: float = 0.1
    max_key_err: float = 0.45  # terminate when key bodies drift this far (m RMS)
    lost_after_hit_is_fall: bool = False  # True: losing the reference after contact counts as a fall (w_fall_after_hit)
    ref_hold_end: bool = False  # True: the episode does not end with the reference clip (it holds its last pose, as in play)
    clamp_ref_phase: bool = False  # True: the reference-time input t / duration is held at 1 past the clip end
    zero_ff_after_ref_end: bool = False  # True: no joint-velocity feed-forward once the reference holds its last pose
    # post-clip balance (Human19): past the clip end no tracking reward, no lost termination and no fall penalty; on hit
    # episodes only, per step bal_frac x (this episode's hit reward) / (bal_window_s / step_dt) x score, score = mean of
    # upright, whole-body COM over the feet line, low root speed, feet down (each 0..1): the sum over the window is at
    # most bal_frac x the hit reward, so falling can never cost more than hitting earns
    post_clip_balance: bool = False
    bal_frac: float = 0.5
    bal_window_s: float = 1.5
    bal_sigma_tilt: float = 0.3  # rad
    bal_sigma_com: float = 0.08  # m, horizontal COM distance from the segment between the feet
    bal_sigma_lin: float = 0.5  # m/s root speed
    bal_sigma_ang: float = 1.5  # rad/s root angular speed
    bal_sigma_foot: float = 0.03  # m, foot height above its lowest point this episode
    settle_tail_s: float = 0.0  # > 0: after the clip, the reference decelerates to a stop over this long (no frozen pose)
    settle_tau: float = 0.15  # root-velocity decay time constant of the settle tail (s)

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


@configclass
class TeeBattingHumanEnvCfg(TeeBattingPerceptionEnvCfg):
    """Step B with a human-strength batter (same limits as the pitcher): the swing is played at the athlete's own
    speed (the strong-motor batter played it 1.45x faster)."""

    robot: ArticulationCfg = make_aib1_cfg(human_strength=True)
    speed: float = 1.0
    speed_end: float = 1.0
    speed_ramp_steps: int = 0
    w_spray: float = 10.0  # the human batter pulled half its balls foul


@configclass
class TeeBattingHumanPlayEnvCfg(TeeBattingHumanEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
