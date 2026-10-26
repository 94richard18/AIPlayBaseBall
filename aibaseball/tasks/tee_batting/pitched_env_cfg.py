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
    # eval / game: pitches from the RL pitcher's eval_stage --pitch_csv table instead of the analytic library
    pitch_table_csv: str | None = None
    pitch_release_csv: str | None = None  # eval_stage --release_csv: every pitch from its own release state (preferred)
    # training: this share of the pitches shaped like an RL pitcher's releases (direction / spin / release point of the
    # table rows, speed drawn in pitch_mix_speed_kmh, near-zone only); the rest from the analytic library
    pitch_mix_release_csv: str | None = None
    tempo_zero_after_release: bool = False  # eval: tempo action held at 0 once the pitch is released
    # after the reference clip ends (~0.4 s after contact): blend the joint targets over this many seconds to a balanced
    # pose ("contact": the clip's contact frame - feet already in the stride; "start": frame 0), residual and
    # feed-forward faded out. 0 = off (hold the last frame, as before)
    post_ref_blend_s: float = 0.0
    post_ref_pose: str = "contact"
    post_ref_keep_residual: bool = False  # True: blend only the reference; the policy residual is not faded out
    pitch_mix_frac: float = 0.5
    pitch_mix_speed_kmh: tuple = (55.0, 75.0)
    pitch_mix_zone_y: float = 0.5  # m, mix pitches are aimed within |y| <= this
    pitch_mix_zone_z: tuple = (0.3, 1.3)  # m, and this height band
    pitch_table_center_z: float = 0.80  # m, the table's plate z that maps onto the swing's nominal contact height


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
    target: 3.10 / 26.1 / 2.07. Only change (decision session: one change per round; Human8 failed training under an aim
    offset): target 29 m (95 ft, ONE OUTS infield edge). Training aim stays Human7's (0); eval / game use -0.12.
    Start from Human7 model_12095."""

    target_distance: float = 29.0


@configclass
class PitchedBattingHuman10EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-25 Human9 continuation (12300 -> 12694): fair and >= 29 m 11.3 -> ~15% of pitches and ONE OUTS 13.1 ->
    16-17%, but diag hit rate (+-0.03 s, aim -0.12) 0.691 -> 0.48 / 0.54 / 0.59 at 12400 / 12500 / 12600: the 29 m reward
    (fair ball past 29 m ~26) outweighs a miss (20). Only change: miss penalty 20 -> 40. Start from Human9 model_12300."""

    w_miss: float = 40.0


@configclass
class PitchedBattingHuman11EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-25 delivery Human9b model_12500 + aim -0.08 (1024-pitch diag 0.684, ONE OUTS 18.5%) falls after 57% of
    its hits in diag / the match video, but training ends the episode follow_through_s = 0.4 s after contact, so a later
    fall is never penalised. Only change: follow through 0.4 -> 1.2 s (contact ~0.8-1.0 s into the 2.6 s episode, so
    1.2 s fits without a longer episode). Start from Human9b model_12500."""

    follow_through_s: float = 1.2


@configclass
class PitchedBattingHuman12EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-25 Human11 (follow through 1.2 s) failed: diag fall 57 -> 65%. Training never registers the falls
    (bat/fall_rate ~0): a batter starting to tip drifts off the reference first, and that 'lost' termination carries no
    fall penalty; the follow-through window (0.4 / 1.2 s) and the 2.6 s episode end before most falls (diag: 0.94 / 1.25 /
    1.87 s after contact at 10 / 50 / 90%). One intervention, three settings (decision session, option A): lost after
    contact counts as a fall, follow through 2.0 s, episode 3.5 s. Start from Human9b model_12500."""

    lost_after_hit_is_fall: bool = True
    follow_through_s: float = 2.0
    episode_length_s = 3.5


@configclass
class PitchedBattingHuman13EnvCfg(PitchedBattingHuman12EnvCfg):
    """2026-10-25 Human12 zero-shot (scripts/diag_episode_end.py, delivery 12500): fell + lost only 1.0-1.5% of the
    hit episodes - 98% end as time_out because the reference clip ends (ref_end), before any post-contact fall; play
    holds the last pose and goes on, and the batter falls (diag 57%). Change: the training episode also holds the last
    pose (ref_hold_end) with the reference-time input held at 1 (clamp_ref_phase, also in play). Start from 12500."""

    ref_hold_end: bool = True
    clamp_ref_phase: bool = True


@configclass
class PitchedBattingHuman14EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-25 the delivery batter (12500 + aim -0.08) vs the RL pitcher's P3c t18 pitches (~64 km/h, 1.05 s flight):
    diag hit rate 0.140 (0.684 at 88-112 km/h), swings early and under the ball. Change (decision session): half the
    training pitches shaped like P3c's releases (release point + spin, 55-75 km/h, aimed near the zone), half the
    analytic 88-112 km/h. Base Human9, not Human13 (one change per round; Human13 failed). Start from the delivery."""

    pitch_mix_release_csv: str = "assets/game/p3c_identical_t18_release.csv"
    pitch_mix_frac: float = 0.5
    pitch_mix_speed_kmh: tuple = (55.0, 75.0)


@configclass
class PitchedBattingHuman15EnvCfg(PitchedBattingHuman14EnvCfg):
    """2026-10-25/26 slow-pitch diagnosis (delivery 12500, 1024-pitch diag, groups fixed): fast 0.795, slow 0.194; vs P3c
    in-zone 0.327 (aim 0); misses are early swings (reference past contact ~0.04 s at the plate); the tempo action is not
    the cause. Human14 (half slow) in-zone 0.339 after 100 its - the fast half may drown the slow one. Change (decision
    session): all training pitches P3c-shaped 55-75 km/h inside the StrikeZone. Base Human9 (via Human14). From 12500."""

    pitch_mix_frac: float = 1.0
    pitch_mix_zone_y: float = 0.2525
    pitch_mix_zone_z: tuple = (0.50, 1.10)


@configclass
class PitchedBattingHuman16EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-26/27 bat-speed diagnosis (delivery 12500): the reference sweet spot reaches ~31 m/s at contact, the robot
    20.5 m/s (exit speed median 17 m/s; ~25 m/s needed for the outfield); the impact model is fine (head-on 20.5 m/s ->
    29.9 m/s). Effort 1.5 / 2.0x zero-shot and action_clip 0.25-1.0 zero-shot did not help. Change (decision session, one
    change per round, strength stays 1.0x): a sweet-spot speed reward in the last 0.1 s of reference time before contact,
    the speed along the reference sweet spot's velocity capped at the reference speed, over its speed at contact. Window
    sum ~4.5 x w at the reference profile, ~3 x w at the robot's 66% (hit rewards ~2-26, miss 20). Base Human9. From 12500."""

    w_ss_speed: float = 2.0
    ss_speed_window_s: float = 0.1


@configclass
class PitchedBattingHumanPlayEnvCfg(PitchedBattingHumanEnvCfg):
    play_mode: bool = True
    pitch_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
