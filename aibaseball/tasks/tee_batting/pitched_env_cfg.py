from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aibaseball.robot.aib1_cfg import make_aib1_cfg

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
    timing_bias_s: float = 0.0  # + = the pitch arrives later than the reference contact (the batter is early)
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


@configclass
class PitchedBattingHumanEnvCfg(PitchedBattingEnvCfg):
    """One-month plan W2: the human-strength batter (TeeBattingHumanEnvCfg, same limits as the pitcher) against
    pitched fastballs at the game's speeds - the human-strength pitcher throws ~115 km/h, so the curriculum ends at
    100-125 km/h instead of 100-150. Start from the human tee batter grown to this task's inputs / tempo action:
    scripts/expand_checkpoint.py <human model_9396.pt> <out.pt> --obs 152 --act 30 --new_std 0.3 --obs_insert 110:1"""

    robot: ArticulationCfg = make_aib1_cfg(human_strength=True)
    speed: float = 1.0
    speed_end: float = 1.0
    speed_ramp_steps: int = 0
    w_spray: float = 10.0
    pitch_speed_range_start: tuple = (90.0, 100.0)
    pitch_speed_range_final: tuple = (100.0, 125.0)


@configclass
class PitchedBattingHuman2EnvCfg(PitchedBattingHumanEnvCfg):
    """Human v0 traded hitting for tracking: from ~iteration 85 the tempo action ran to its limit (|tempo| 0.24 -> 0.99),
    the swing lost the ball (hit rate 0.20 -> 0, closest bat-ball 0.19 -> 1.35 m) while the tracking reward rose
    0.72 -> 0.87 - a hit at ~21 m/s exit speed was worth less than the tracking it cost. Make a miss expensive and the
    tempo a small correction. Start from v0 model_9400 (before the drift)."""

    w_miss: float = 20.0
    w_tempo: float = 0.3


@configclass
class PitchedBattingHuman3EnvCfg(PitchedBattingHuman2EnvCfg):
    """One-month plan W2 (2026-10-20 schedule): the game's pitches cross the plate at >= 80 km/h (W1 gate), i.e.
    about 88-112 km/h at release with drag. Human2 (100-125 km/h) ended at hit rate ~0.14, exit velo ~21 m/s.
    Only change: the pitch speed range. Start from human_pitched2 model_9699."""

    pitch_speed_range_start: tuple = (88.0, 100.0)
    pitch_speed_range_final: tuple = (88.0, 112.0)


@configclass
class PitchedBattingHumanPlayEnvCfg(PitchedBattingHumanEnvCfg):
    play_mode: bool = True
    pitch_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
