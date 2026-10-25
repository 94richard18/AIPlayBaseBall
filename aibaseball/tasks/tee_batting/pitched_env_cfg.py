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
class PitchedBattingHuman4EnvCfg(PitchedBattingHuman3EnvCfg):
    """2026-10-20 diag (scripts/diag_bat_contact.py, Human3 model_10298, 256 pitches): hit rate 0.215, flat over the
    pitch speed; misses are late - 90% of them 28-131 ms behind the hits' timing, the ball passing nearest the knob.
    The policy can wait for a late pitch but cannot catch up with an early one. Zero-shot timing bias: +0.04 s ->
    hit rate 0.328, +0.08 s -> 0.277 (now early). Only change: the pitch arrives 0.04 s later. Start from model_10298."""

    timing_bias_s: float = 0.04


@configclass
class PitchedBattingHuman5EnvCfg(PitchedBattingHuman4EnvCfg):
    """2026-10-21 diag (Human4 model_10897, 256 pitches): hit rate 0.398; misses are still mostly timing (134/154) and
    spread wide (-0.079 / -0.051 / -0.018 s vs the hits' ~-0.10 s) - the +-0.08 s arrival jitter is not absorbed per
    pitch although t_arrive is observed and |tempo| stays ~0.16. Only change: tempo penalty 0.3 -> 0.1 (w_miss 20
    still makes a drift to tracking expensive, the Human v0 failure). Start from Human4 model_10897."""

    w_tempo: float = 0.1


@configclass
class PitchedBattingHuman6EnvCfg(PitchedBattingHuman4EnvCfg):
    """2026-10-21 user decision: in the game the swing starts from the engine's true arrival time, so no arrival jitter.
    Human4 model_10897 zero-shot with jitter 0: bias 0.04 -> hit rate 0.590, 0.055 -> 0.648, 0.07 -> 0.594 (+-0.08
    jitter: 0.398). The hittable window is only ~+-5 ms wide. Game conditions: jitter 0, bias 0.055 (Human5's tempo
    change failed). Start from Human4 model_10897."""

    timing_noise_s: float = 0.0
    timing_bias_s: float = 0.055


@configclass
class PitchedBattingHuman7EnvCfg(PitchedBattingHuman6EnvCfg):
    """2026-10-22 quality check (Human6 model_11496, 256 pitches, showcase --eval_envs): contact 94.5% but launch
    quartiles -55 / -34 / -16 deg, carry median 1.1 m, exit 23 m/s (>= 28 m/s 4%), topspin ~-5000 rpm; diag: ball
    centre 1.3 cm below the bat axis at contact (median). The inherited launch reward (TeeBattingMimicLaunch: Gaussian,
    sigma 12 deg around 28) is ~1e-14 at -40 deg and the backspin term is clamped at -1: no gradient toward lift.
    Only change: the smooth direction score (any angle, 0.45 at -40 deg, 1 at the ideal 30 deg). Start from Human6."""

    launch_sigma_deg: float = 0.0


@configclass
class PitchedBattingHuman8EnvCfg(PitchedBattingHuman7EnvCfg):
    """2026-10-23 zero-shot on Human7 model_12095 (256 pitches): aim_offset_z 0 / -0.03 / -0.06 / -0.09 m -> launch
    median -37 / -23 / -14 / -4 deg, carry median 1.0 / 1.7 / 2.7 / 5.6 m, hit rate (+-0.03 s) 0.84 / - / 0.79 / 0.80.
    Raising the pitch aim (tee_offset_z 0.10 / 0.13) did more (-1 / +6 deg) - the policy half follows the perceived
    ball. Only change: the batter aims 0.10 m under the ball (its controller, usable in the game). Start from Human7."""

    aim_offset_z: float = -0.10


@configclass
class PitchedBattingHuman9EnvCfg(PitchedBattingHuman7EnvCfg):
    """2026-10-25 ONE OUTS (Human7 model_12095 + aim_offset_z -0.12, 1000 matches): batter wins 8.0%, infield balls 52.5%.
    With the 150.3 m target the distance term is ~0 at ONE OUTS distances: on the 256 evaluated balls an infield out
    scores 1.90 on average, a fair ball past 29 m 2.19, a foul 1.49 - training was paid to hit grounders. With a 29 m
    target: 3.10 / 26.1 / 2.07. Changes: target 29 m (95 ft, ONE OUTS infield edge), and train in the game's aim
    (aim_offset_z -0.12) instead of adding it afterwards. Start from Human7 model_12095."""

    target_distance: float = 29.0
    aim_offset_z: float = -0.12


@configclass
class PitchedBattingHumanPlayEnvCfg(PitchedBattingHumanEnvCfg):
    play_mode: bool = True
    pitch_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
