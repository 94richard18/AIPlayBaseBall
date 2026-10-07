from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .mimic_env_cfg import TeeBattingPerceptionEnvCfg


@configclass
class PitchedBattingEnvCfg(TeeBattingPerceptionEnvCfg):
    """Step D: hit analytic pitches (drag + Magnus) with perception noise/latency and swing-tempo control."""

    action_space = 30  # 29 joints + tempo
    extra_actions: int = 1
    observation_space = 152  # mimic obs (146 + 1 extra prev action) + perceived ball velocity 3 + released 1 + rate 1
    rsi_prob: float = 0.0  # always start in the stance; the batter waits for the pitch
    episode_length_s = 2.6

    # pitches (library is built once for the final range; the curriculum samples within the current range)
    pitch_library_size: int = 8192
    pitch_speed_range_start: tuple = (75.0, 85.0)
    pitch_speed_range_final: tuple = (100.0, 150.0)
    timing_noise_s: float = 0.08  # +- arrival-time offset the batter must absorb (final)
    pitch_ramp_steps: int = 32 * 1500  # control steps to reach the final speed range / timing noise
    min_lead_s: float = 0.15  # the pitch is released at least this long after the episode start
    ball_obs_clip: float = 1.5  # keep the ball-relative inputs in the range the tee-trained policy knows
    tempo_scale: float = 0.3  # tempo action -> reference playback rate x (1 +- 0.3)
    w_tempo: float = 0.01
    w_miss: float = 2.0


@configclass
class PitchedBattingPlayEnvCfg(PitchedBattingEnvCfg):
    play_mode: bool = True
    pitch_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
