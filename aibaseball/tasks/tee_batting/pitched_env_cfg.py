from __future__ import annotations

import os

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
    # joint ONE OUTS env check: the pitch crosses at its own plate height (not re-centred by the clip's contact height
    # tee_z + tee_offset_z - pitch_table_center_z), as in one scene where the batter stands on the ground
    real_height: bool = False
    take_hold: bool = False  # match video (play only): a take - zero actions, the reference held in the stance (frame 0)

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
    # 10-30 (planning 14): the crossing prediction from a ballistic least-squares fit over all seen samples since release
    # instead of a one-step finite difference (1.4 m/s velocity noise per axis at 1 cm / 10 ms); delay / noise / bias unchanged
    percept_filter: bool = False
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
class PitchedBattingHuman18EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-27 delivery 12500 at AIB_EFFORT_SCALE 1.5 vs P3c (aim 0): ONE OUTS 19.9%, falls after 43% of hits ~0.8 s
    after the clip ends; a decelerating settle tail did not move them (43.1 vs 43.4%, same timing), so the target past
    the clip is not the cause - training ends the episode at the clip end (ref_end) and never sees the fall. Change
    (decision session): Human13's post-clip training (hold the last pose, reference-time input at 1, Human12's lost after
    contact = fall, follow through 2.0 s, episode 3.5 s) on Human15's P3c-shaped pitches; Human13 ran it at 1.0x on
    88-112 km/h and a crash cut it at 150 its. Run with AIB_EFFORT_SCALE 1.5. From 12500."""

    lost_after_hit_is_fall: bool = True
    follow_through_s: float = 2.0
    episode_length_s = 3.5
    ref_hold_end: bool = True
    clamp_ref_phase: bool = True


@configclass
class PitchedBattingHumanPlayEnvCfg(PitchedBattingHumanEnvCfg):
    play_mode: bool = True
    pitch_ramp_steps: int = 0
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class PitchedBattingHuman19EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-27 Human18 (post-clip training with the held reference + fall penalty) lowered diag falls 42 -> 25% by
    +300 but ONE OUTS 19.9 -> 16.4% and outfield 4.1 -> 0.2%: a softer swing, training falls per hit still 98%; the fall
    penalty (10, plus the lost tracking reward) outweighed the hit reward. Change (user 10-27 19:4x): before the clip end
    Human15 unchanged; past it, on hit episodes, no mimic target (no tracking reward, no lost termination) and no
    fall penalty, only a standing reward capped at bal_frac 0.5 x the episode's hit reward; misses keep Human15's tracking,
    lost termination and fall penalty on the held pose (decision session); no velocity feed-forward on the held pose
    (post_clip_balance). Episode runs on (ref_hold_end, clamp_ref_phase, follow through 2.0 s, episode 3.5 s). Run with
    AIB_EFFORT_SCALE 1.5. From 12500."""

    post_clip_balance: bool = True
    ref_hold_end: bool = True
    clamp_ref_phase: bool = True
    zero_ff_after_ref_end: bool = True
    follow_through_s: float = 2.0
    episode_length_s = 3.5


@configclass
class PitchedBattingHuman20EnvCfg(PitchedBattingHuman9EnvCfg):
    """2026-10-27 Control C (Human15 alone at 1.5x from 12500, no post-clip change): outfield 5-seed mean 4.7 -> 1.4 ->
    0.8% at +100 / +200, ONE OUTS 18.8 -> 16.1 -> 14.7%: the P3c-shaped pitch mix itself costs strength. Change (decision
    session): Human19's post-clip balance on Human9's pitches (88-112 km/h, as Human17, whose 12698 kept outfield 3.5%).
    Run with AIB_EFFORT_SCALE 1.5. From 12500."""

    post_clip_balance: bool = True
    ref_hold_end: bool = True
    clamp_ref_phase: bool = True
    zero_ff_after_ref_end: bool = True
    follow_through_s: float = 2.0
    episode_length_s = 3.5


@configclass
class PitchedBattingHuman21EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-27 Human18-20: every post-clip reward that the swing could see traded swing strength for fewer falls
    (or did nothing). Change (decision session, the pitcher's PR idea): train with --post_clip_teacher 12500 - the frozen
    delivery actor drives every action column until the clip end, the student (a copy of 12500) only after it, hits and
    misses alike - so the swing cannot change. Past the clip end: no tracking, no lost termination, Human15's fall penalty
    kept, a standing reward of 1 x the score per step (uncapped). Human15's pitches. Run with AIB_EFFORT_SCALE 1.5."""

    post_clip_balance: bool = True
    post_clip_stand_w: float = 1.0
    ref_hold_end: bool = True
    clamp_ref_phase: bool = True
    zero_ff_after_ref_end: bool = True
    follow_through_s: float = 2.0
    episode_length_s = 3.5


@configclass
class PitchedBattingHuman22EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-28 arm B power line (decision session): the frozen M2 (12500 at 1.5x) has no outfield ball in 640 joint
    pitches (exit speed median ~16 m/s, ~25 m/s needed). Change: target 29 -> 43 m (mid outfield) and the exit-speed term
    x3 (Power cfg's 20 -> 60, x exit speed / 52.5), fall terms unchanged. Human15's P3c-shaped pitches. Run with
    AIB_EFFORT_SCALE 1.5 (the spec value). From 12500."""

    target_distance: float = 43.0
    w_exit_velo: float = 60.0


@configclass
class PitchedBattingHuman23EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-29 bat-speed cap (diag_bat_contact --link_log on 12500, 277 hits): the upper body is not slow but early - waist
    peaks 0.06 s and shoulder / wrist pitch 0.20-0.26 s before contact (reference: at contact), at every swing-start bias,
    while Human16 / 16b / 22's speed rewards traded hit rate for nothing. Change (decision session): a contact-window joint-
    velocity tracking term (last 0.1 s of reference time before contact, waist + left arm, sigma 8 rad/s, w 1.0) so the
    segment speeds peak at contact. Human15's pitches. Run with AIB_EFFORT_SCALE 1.5. From 12500."""

    w_cvel: float = 1.0
    cvel_window_s: float = 0.1
    sigma_cvel: float = 8.0


@configclass
class PitchedBattingHuman23bEnvCfg(PitchedBattingHuman23EnvCfg):
    """2026-10-29 Human23 +100: cvel_err flat at ~12 rad/s - with sigma 8 the term paid exp(-144/64) ~ 0.1 per step (no
    gradient). Change (decision session, last shot at 1.5x): sigma 20 rad/s, w_cvel 3 (below the hit reward's upper range)."""

    w_cvel: float = 3.0
    sigma_cvel: float = 20.0


@configclass
class PitchedBattingHuman26EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-30 contact precision (user decision A; planning 14): the contact-offset replay of the frozen M2 (35 joint
    hits) put 67% of the exit-speed gap on bat speed and 33% on contact (|dx| median 7.4 cm, 24/35 beyond 5 cm; D +0.8
    [-2.6, +3.7] cm, launch tracks D at r +0.97); centred at today's bat speed exit is 22.8 m/s, and D +1.5 cm gives
    launch ~23 deg with backspin (logs/contact_offset/d15_check.py) - the corner outfield needs centred contact with a
    slight undercut. Change: impact-step contact terms, w 3 x exp(-(dx / 0.04)^2) + w 3 x exp(-((D - 0.015) / 0.012)^2),
    on top of the existing hit terms. The swing plane is folded into D (sweet-spot velocity elevation already +12 deg
    median). Human15's pitches. Run with AIB_EFFORT_SCALE 1.5. From 12500."""

    w_contact_dx: float = 3.0
    w_contact_D: float = 3.0


@configclass
class PitchedBattingHuman26bEnvCfg(PitchedBattingHuman26EnvCfg):
    """2026-10-30 Human26 +100 failed: sigma_dx 0.04 paid ~0.002 at the ~10 cm |dx| median (no gradient) and the flat D
    bonus was farmed by more contact (in-zone hit 0.445 -> 0.643, |dx| / D unchanged). One retry (planning 14): contact
    term 4 x g_dx + 3 x g_dx x g_D, g_dx = exp(-(dx / 0.12)^2) (~0.5 at 10 cm), g_D unchanged. From 12500."""

    w_contact_dx: float = 4.0
    sigma_contact_dx: float = 0.12
    w_contact_D: float = 3.0
    contact_D_times_dx: bool = True


@configclass
class PitchedBattingHuman27EnvCfg(PitchedBattingHuman15EnvCfg):
    """2026-10-30 contact accuracy (user: sweet-spot contact; planning 14). Timing x contact station (bias sweep, diag1024):
    |dx| ~10 cm at every swing-start bias, within-bias corr(dx, timing) <= 0.24; the miss vector is along the bat axis
    (10.2 cm hits / 12.4 cm misses) and axis_off tracks the pitch's plate_y at slope -0.80 (r -0.86): the sweet spot lands
    at a ~fixed spot, the reach barely follows the pitch. Human26 / 26b (contact bonuses on hits only) were farmed by more
    contact. Change: an aim term on EVERY swing (hit or miss) at the closest approach, 6 x exp(-(axis_off / 0.08)^2) x
    exp(-(gap / 0.05)^2); the hit reward becomes 20 x quality, quality = exp(-(dx / 0.08)^2) x exp(-((D - 0.010) / 0.015)^2);
    flat per-hit terms off (distance, success, exit speed, backspin, the mimic contact approach w_contact); the launch and
    spray terms (exit-speed scaled) and w_miss 20 stay. Obs unchanged. Run with AIB_EFFORT_SCALE 1.5. From 12500."""

    w_aim: float = 6.0
    w_hit_q: float = 20.0
    w_distance: float = 0.0
    w_success: float = 0.0
    w_exit_velo: float = 0.0
    w_backspin: float = 0.0
    w_contact: float = 0.0


@configclass
class PitchedBattingHuman27bEnvCfg(PitchedBattingHuman27EnvCfg):
    """Human27b (planning 14, 10-30; only if Human27 fails +100 with axis_off unmoved and the plate_y slope still ~-0.8):
    Human27 + the arm-relax switch - in the last 0.15 s before the predicted arrival (until the hit) the shoulder / elbow /
    wrist joints and the hand / bat key points count x0.5 in the pose / velocity / key tracking errors; legs, torso and
    root tracking unchanged. From 12500 @ 1.5x."""

    arm_relax_s: float = 0.15
    arm_relax_factor: float = 0.5
    # planning 14 (10-30, after Human27 +100 FAIL: sigma_aim_ax 0.08 paid 0.008 at the ~17.5 cm start - only the gap factor
    # had a gradient). Rule: sigma >= 1.2 x the quantity's starting median (12500 diag1024: |axis_off| 0.152, gap 0.056,
    # |dx| 0.112, |D - 0.010| 0.026 -> sigma_hit_q_D 0.032 instead of 14's 0.025).
    sigma_aim_ax: float = 0.20
    sigma_aim_gap: float = 0.10
    sigma_hit_q_dx: float = 0.15
    sigma_hit_q_D: float = 0.032
    launch_spray_times_q: bool = True



@configclass
class PitchedBattingHuman29EnvCfg(PitchedBattingHuman27bEnvCfg):
    """Human29 (planning 14, 10-30): Human27b on the FILTERED percept (percept_filter: ballistic least-squares crossing
    prediction over all seen samples since release; delay 0.10 s / noise 1 cm / bias 2 cm unchanged). The raw one-step
    velocity estimate wobbled the predicted crossing by ~12 cm mean (25 cm p90) at the last step, so 12500 - and the
    reward-only lines Human23 / 26 / 27 on top of it - learned on an input that carried no aim signal. Aim sigmas >= 1.2 x
    the start medians of the filter-on zero-shot diag (AIB_H29_SIGMA_AX / _GAP set by logs/run_human29.sh; defaults
    0.20 / 0.10 as Human27b). From 12500 @ 1.5x."""

    percept_filter: bool = True
    sigma_aim_ax: float = float(os.environ.get("AIB_H29_SIGMA_AX", "0.20"))
    sigma_aim_gap: float = float(os.environ.get("AIB_H29_SIGMA_GAP", "0.10"))


_ARMAIM_JOINTS = [f"{s}_{j}" for s in ("l", "r") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")] + \
    [f"{s}_wrist_{j}" for s in ("l", "r") for j in ("roll", "pitch", "yaw")] + ["waist_yaw"]
_SWING_12500 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "rsl_rl", "aib1_tee_batting_mimic",
                                            "2026-10-25_13-56-51_human_pitched9b", "model_12500.pt"))


@configclass
class ArmAimEnvCfg(PitchedBattingHuman15EnvCfg):
    """Human28 "ArmAim" (planning 14, user 10-30 "動打者架構"; arm_aim_env.ArmAimEnv): a separate residual policy on 15 joints
    (shoulders + elbows + wrists of both arms, waist_yaw), bound +-0.4 rad (step 0, scripts/armaim_authority.py: at contact,
    top hand held, sweet-spot height held, +42 / -13 cm), over the FROZEN swing policy 12500 @ 1.5x on Human15's pitches.
    Reward once per swing: 6 x exp(-(axis_off / 0.20)^2) x exp(-(gap / 0.10)^2) at the closest approach; 20 x q at contact,
    q = exp(-(dx / 0.15)^2) x exp(-((D - 0.010) / 0.032)^2); miss 20 and the falls as Human15; - 0.01 x sum(tanh(a)^2) in the
    window. No tracking / speed / distance terms. Run with AIB_EFFORT_SCALE 1.5. From scratch."""

    swing_checkpoint: str = _SWING_12500
    aim_joints: list = _ARMAIM_JOINTS
    aim_bound: float = 0.4
    aim_window_s: float = 0.30  # reference-clock seconds before the clip's contact: the window opens (the swing start)
    aim_decay_tau_s: float = 0.05  # after the window: the offset decays with this time constant
    aim_scripted: str | None = None  # scripts/armaim_scripted_gain.py JSON: the offset comes from it, the policy is ignored
    aim_scripted_sign: float = 1.0  # -1: control row (offset sign flipped)
    aim_scripted_true: bool = False  # falsifier only: dy from the true plate y (privileged) instead of the percept
    aim_scripted_z: bool = False  # falsifier row 5: + the height offset (gain_zplus / zminus of the gain JSON)
    aim_scripted_D_fit: tuple = (-0.1437, -0.048, 0.195)  # 12500 hits: D = a0 + ay * plate_y + az * plate_z (r2 0.36)
    aim_scripted_D_target: float = 0.010
    aim_scripted_dry: bool = False  # offset computed + logged, NOT applied (paired control for the realised share)
    aim_scripted_dump: str | None = None  # npz: per-pitch realised sums (env, episode, num / den per group)
    aim_obs_comp: bool = False  # the frozen swing policy sees q - offset on the offset joints (row 6)
    aim_obs_comp_full: bool = False  # + the bat pose it sees moved back by the Jacobian-mapped offset (row 8)
    aim_hold: bool = False  # ArmAim2: the policy's action is sampled once at the window's first step and held
    aim_delta: float = 0.5  # aim_hold + aim_scripted: the held action adds aim_delta x tanh(a) (bound units) to the scripted aim
    w_res: float = 0.01
    w_aim: float = 6.0
    sigma_aim_ax: float = 0.20
    sigma_aim_gap: float = 0.10
    w_hit_q: float = 20.0
    sigma_hit_q_dx: float = 0.15
    hit_q_D_target: float = 0.010
    sigma_hit_q_D: float = 0.032
    w_distance: float = 0.0
    w_success: float = 0.0
    w_exit_velo: float = 0.0
    w_launch: float = 0.0
    w_backspin: float = 0.0
    w_spray: float = 0.0
    w_contact_dx: float = 0.0
    w_contact_D: float = 0.0
    w_contact: float = 0.0



@configclass
class ArmAim2EnvCfg(ArmAimEnvCfg):
    """ArmAim2 (planning 14, 10-30, after the +300 FAIL: per-step noise averaged out by PD, no gradient): the offset is the
    scripted aim (logs/armaim_auth/scripted_gain.json, gain(side) x |dy|) plus a correction the policy picks ONCE at the
    window's first step and holds (aim_delta x tanh(a), +-0.2 rad); actor head zeroed at the start (= the scripted aim),
    std fixed 0.1, entropy 0 (AIB_FIX_STD=1, ArmAim2PPORunnerCfg). Rewards as ArmAim."""

    aim_scripted: str | None = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "armaim_auth",
                                                            "scripted_gain.json"))
    aim_hold: bool = True
    aim_delta: float = 0.5
    w_res: float = 0.01


@configclass
class ArmAimPlayEnvCfg(PitchedBattingHumanPlayEnvCfg):
    """Eval / diag (diag_bat_contact --armaim): ArmAim on the play env (ball flight, 9 s episodes)."""

    swing_checkpoint: str = _SWING_12500
    aim_joints: list = _ARMAIM_JOINTS
    aim_bound: float = 0.4
    aim_window_s: float = 0.30
    aim_decay_tau_s: float = 0.05
    aim_scripted: str | None = None
    aim_scripted_sign: float = 1.0
    aim_scripted_true: bool = False
    aim_scripted_z: bool = False
    aim_scripted_D_fit: tuple = (-0.1437, -0.048, 0.195)
    aim_scripted_D_target: float = 0.010
    aim_scripted_dry: bool = False  # offset computed + logged, NOT applied (paired control for the realised share)
    aim_scripted_dump: str | None = None  # npz: per-pitch realised sums (env, episode, num / den per group)
    aim_obs_comp: bool = False  # the frozen swing policy sees q - offset on the offset joints (row 6)
    aim_obs_comp_full: bool = False  # + the bat pose it sees moved back by the Jacobian-mapped offset (row 8)
    aim_hold: bool = False
    aim_delta: float = 0.5
    w_res: float = 0.01
    w_aim: float = 6.0
    sigma_aim_ax: float = 0.20
    sigma_aim_gap: float = 0.10
