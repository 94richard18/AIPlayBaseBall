from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from aibaseball.physics.constants import BallSpec
from aibaseball.robot.aib1_cfg import make_elastic_pitcher_cfg, make_pitcher_cfg

BALL = BallSpec()
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "motions"))
KMH = 1 / 3.6


@configclass
class PitchEnvCfg(DirectRLEnvCfg):
    # ---------------------------------------------------------------- timing
    episode_length_s = 2.0  # capped by the end of the reference
    # 400 Hz policy, 800 Hz physics (fingers interpolated per physics step): release timing needs ~1 ms
    # precision and the ball moves ~5.6 cm per physics step at release
    decimation = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 800,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**22, gpu_total_aggregate_pairs_capacity=2**22),
    )

    # ---------------------------------------------------------------- spaces
    action_space = 45  # 29 body joints (residual on the reference) + 16 right-hand finger joints
    observation_space = 191
    state_space = 0

    # ---------------------------------------------------------------- scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = make_pitcher_cfg()
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, max_linear_velocity=1000.0,
                                                         max_angular_velocity=100000.0),  # deg/s (~16,700 rpm)
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
            # leather on a rubber-padded robot fingertip
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.92)),
        ),
    )

    # ---------------------------------------------------------------- reference motion
    motion_file: str = os.path.join(MOTION_DIR, "pitch_2916-4.npz")  # capture + synthesised 0.8 s follow-through
    phase_duration_s: float = 1.1167  # phase observation = t / this (length of the capture alone; keeps trained policies valid)
    # aim the whole delivery (rotation about the vertical axis through the rubber); the retargeted
    # hand releases ~9.5 deg toward first base, so the reference is turned that much toward third base
    ref_yaw_deg: float = -8.0
    speed: float = 1.0
    speed_end: float = 1.0
    speed_ramp_steps: int = 0
    rsi_prob: float = 0.2  # mostly full pitches from the set position (what is evaluated)
    rsi_margin_before_release: float = 0.12
    rsi_min_s: float = 0.0  # earliest reference-state start (s); later starts skip the already-learned wind-up
    residual_scale: float = 0.5
    vel_feedforward: float = 1.0
    post_release_feedforward: float = 1.0  # velocity feed-forward scale after the release
    # balance mode after the release: the reference stays the PD base pose, but nothing rewards following it
    # (_stand_reward); the reference COM / contact terms switch off
    post_release_balance: bool = False
    w_stand_support: float = 0.0
    w_stand_spin: float = 0.0
    sigma_stand_spin: float = 15.0  # kg m^2/s (PD-only from the reference release: ~31-37 when tipping over)
    w_stand_upright: float = 0.0
    w_stand_feet: float = 0.0
    w_stand_still: float = 0.0
    w_stand_pivot: float = 0.0
    # lead-foot placement SIDEWAYS (across the pitching line, env y) and the COM in line with the lead foot (0 = off).
    # The distance-only w_lead_plant band let the foot land ~0.38 m toward the throwing-arm side; the COM then sat 0.22 m
    # outside the lead foot and no ground force could stop the body after the release (centroidal LP, 2/16 states)
    w_lead_lateral: float = 0.0  # per step x exp(-(lateral error / sigma)^2) inside lead_lateral_window
    sigma_lead_lateral: float = 0.10  # m
    lead_lateral_window: tuple = (0.6, 0.3)  # s: from this reference time to reference release + this
    w_lead_land: float = 0.0  # once, at the lead foot's first touchdown x exp(-(lateral error / sigma)^2)
    sigma_lead_land: float = 0.08  # m
    lead_touch_after_s: float = 0.6  # s, reference time: first lead-foot touchdown counted from here (not the leg lift)
    # True: the lead-foot touchdown = the first contact after the lead foot was off the ground (any time after
    # lead_lift_after_s), not "a new contact after lead_touch_after_s". The policy's own stride lands at ~0.60 s: a
    # touchdown at 0.59 s never counted and every later release was judged early (1200: 63/64)
    lead_land_after_lift: bool = False
    lead_lift_after_s: float = 0.1
    # (lead_land_after_lift) the touchdown counts only with the lead foot this far ahead of the back foot toward the
    # plate. Reference: 0.10 m in the set position, 0.83 m at 0.6 s, 1.33 m at the strike. At 1500 the policy lifted the
    # lead foot, put it straight back down and released at 0.485 s (its real stride lands at 0.603 s)
    lead_stride_min_m: float = 0.0
    w_com_lateral: float = 0.0  # strike -> release: (COM - lead foot) sideways like the athlete's (~0)
    sigma_com_lateral: float = 0.06  # m
    # physics instead of the athlete's spot: capture point (COM + COM velocity x sqrt(h / g)) sideways over the planted
    # lead foot, wherever the robot puts that foot
    w_cp_lateral: float = 0.0  # lead foot planted, lead_touch_after_s -> reference release + 0.3 s
    sigma_cp_lateral: float = 0.06  # m
    sigma_cp_lateral_wide: float = 0.30  # m: 20050 sat at 0.43 m, where the 6 / 8 cm bands alone give no gradient
    w_land_cp: float = 0.0  # once, at the lead foot's first touchdown
    sigma_land_cp: float = 0.08  # m
    # ---- reward v2 (docs/reward_redesign_stage1.md): outcome first, physics guidance on the robot's own feet
    reward_v2: bool = False
    free_after_s: float = 0.5  # s, reference time: imitation (style) only before this; legs get free_leg_gain after it
    free_leg_gain: float = 1.0  # residual gain of both legs from free_after_s to the release
    freeze_pd_after_release: bool = False  # PD base = the reference pose at the release (not the follow-through)
    # > 0: after the release the PD base blends over this time from the release pose to a standing pose (upper body:
    # the set position; legs: the athlete's stride at its release)
    stand_pd_blend_s: float = 0.0
    # legs of that standing pose: "ref_release" = the athlete's at its release (lead hip flexed ~90 deg: with the
    # robot's shorter stride it lifted the planted lead foot), "hold" = the robot's own leg angles at its release
    stand_pd_legs: str = "ref_release"
    w_throw: float = 100.0  # once at the release x clamp((ball-to-plate distance decrease) / throw_full_m, 0, 1)
    throw_full_m: float = 5.0
    w_stand_v2: float = 200.0  # once at release + follow_through_s if standing, x (0.5 + 0.5 x throw score)
    stand_max_com_v: float = 0.2  # m/s
    stand_max_tilt_deg: float = 35.0  # trunk from vertical
    stand_min_head_z: float = 1.3  # m
    stand_max_sink_acc: float = 0.5  # m/s^2: trunk not accelerating downward (upward is fine; both feet must be down)
    w_fall_v2: float = 200.0  # once, at a fall (the episode ends)
    w_style: float = 0.1  # per step before free_after_s x normalized reference tracking
    w_hold_v2: float = 0.05  # per step before the release: ball in the grip
    w_cp_v2: float = 0.1  # per step from the robot's own lead-foot touchdown: capture point inside the support
    w_spin_v2: float = 0.05  # per step after the release: horizontal centroidal angular momentum small
    w_up_v2: float = 0.05  # per step after the release: head above 1 m
    w_torque: float = 0.001  # per step x mean (torque / limit)^2
    w_slip_v2: float = 0.1  # per step x horizontal speed of the planted feet (m/s)
    # survival shaping (plan A, 2026-10-12): a fall that every pitch gets carries no advantage signal
    w_alive_v2: float = 0.0  # total over follow_through_s for staying up after the release (spread per step)
    fall_graded: bool = False  # a fall after the release costs w_fall_v2 x (0.5 + 0.5 x share of follow-through left)
    post_release_pen_scale: float = 1.0  # action rate / torque / impact / slip after the release x this
    # a release before the robot's own lead-foot touchdown is no pitch (from scratch the policy threw from the set
    # position at 0.07 s and walked through the reference without the ball): ends the episode, costs w_early_release
    release_after_lead_strike: bool = False
    w_early_release: float = 0.0
    # settling (2026-10-13): from settle_after_s after the release, per step toward each standing criterion (soft), so the
    # one-shot stand bonus at follow_through_s gets found. Total over the window ~ w_settle_v2
    w_settle_v2: float = 0.0
    settle_after_s: float = 0.5
    w_lead_hip_roll: float = 0.0  # lead hip roll like the reference inside lead_hip_roll_window (0 = off)
    sigma_lead_hip_roll: float = 0.15  # rad
    lead_hip_roll_window: tuple = (0.5, 0.85)  # s, reference time
    action_scale: dict = {"hip": 0.5, "knee": 0.5, "ankle": 0.4, "waist": 1.0, "shoulder": 1.5, "elbow": 1.5, "wrist": 1.2}
    finger_action_scale: float = 0.4  # rad around the squeezed four-seam grip pose
    grip_squeeze: float = 0.15  # default finger targets this far past contact = grip force (0.3 ejected the ball)

    # ---------------------------------------------------------------- field (OBP lab frame: +x toward home plate)
    plate_distance: float = 18.44  # m, pitching rubber -> home plate
    zone_center_y: float = 0.0
    zone_bottom: float = 0.45  # above the plate-level ground
    # pitcher's mound (MLB: rubber 10 in above the plate, 1 in / ft slope); the field is mound_height below its top.
    # Fitted to the reference soles (scripts/analyze_ref_dynamics.py): the pivot foot stands at z = +0.058 on the
    # rubber (flat) and the lead foot lands at z = -0.03, x = 1.6 m. On a flat floor the lead foot struck the ground early.
    mound_top_z: float = 0.058
    mound_height: float = 0.254
    mound_slope: float = 1.0 / 12.0
    mound_slope_start_x: float = 0.54
    # side to side (MLB mound: 18 ft = 5.49 m across). Was 2.4 m, so a foot stepping out sideways dropped off the edge;
    # neighbouring envs' mounds (4 m apart) overlap with the same surface, so the overlap is seamless.
    mound_width: float = 5.5
    # baseball spikes dig into the mound clay. The reference's braking at foot strike needs a friction coefficient of
    # ~0.9-1.4; at 1.0 the lead foot slid (~0.9 m/s) and lifted right after landing.
    mound_static_friction: float = 1.8
    mound_dynamic_friction: float = 1.6
    zone_height: float = 0.95
    zone_width: float = 0.65
    target_speed: float = 120.0 * KMH  # 33.3 m/s (lowered from 150 km/h for the human-strength pitcher)

    # ---------------------------------------------------------------- release / drop
    release_distance: float = 0.05  # ball centre this far from the grip point => released
    min_release_speed: float = 15.0  # slower "releases" are drops
    pad_speed_tolerance: float = 1.03  # ball speed at release <= 1.03 x fastest finger-pad speed (momentum audit)
    follow_through_s: float = 0.80  # must stay balanced after release (no diving follow-through)

    # ---------------------------------------------------------------- rewards
    w_track: float = 0.5  # lowered: the reference hand path near release points off the plate
    w_pose: float = 0.45
    w_vel: float = 0.10
    w_key: float = 0.30
    w_root: float = 0.15
    sigma_pose: float = 0.25
    sigma_vel: float = 5.0
    sigma_key: float = 0.10
    sigma_root: float = 0.10
    sigma_rot: float = 0.30
    max_key_err: float = 0.45
    w_hold: float = 0.3  # per step before the reference release: ball still in the grip
    w_speed: float = 20.0  # x min(release speed / target, 1.0)
    w_speed_bonus: float = 10.0  # release speed >= target_speed
    speed_soft_cap: float = 135.0 * KMH  # above this, speed is penalised (trade speed for control)
    w_overspeed: float = 2.0  # per 3 m/s above the soft cap
    w_strike: float = 20.0  # crosses the plate inside the zone
    w_zone: float = 20.0  # x exp(-(distance to zone centre / 0.4)^2)
    w_aim: float = 20.0  # x exp(-(distance / 2 m)^2)
    w_aim_wide: float = 20.0  # x exp(-(distance / 8 m)^2): signal while still far off
    w_both: float = 20.0  # strike AND >= target_speed
    w_drop: float = 5.0
    w_fall: float = 5.0
    w_fall_after_release: float = 30.0  # falling within the follow-through window
    # follow-through (no reference after the release): balance reward replaces most of the tracking reward
    w_balance: float = 1.0  # per step: upright x pelvis not dropping x calming angular velocity
    w_post_release_track: float = 1.0  # the reference now has a real follow-through (was 0.2 with a frozen last frame)
    # post-release rewards x (release speed / target)^p: with a real follow-through to track, standing up after a
    # soft toss out-earned the release rewards and the release speed fell from 124 to 63 km/h
    post_release_speed_pow: float = 4.0
    post_release_pelvis_z: float = 0.75  # m; lower than this is penalised by the balance term
    post_release_head_z: float = 1.10  # m; head lower than this is penalised by the balance term
    # fallen = body (nearly) on the ground; z = 0 is the rubber, the lead foot stands ~0.1 m lower. Higher thresholds
    # ended deep but recoverable follow-through bends before the policy could learn to recover from them.
    fallen_head_z: float = 0.45
    fallen_pelvis_z: float = 0.30
    post_release_residual_gain: float = 2.0  # residual authority after the release (6.0 did not stop the falls)
    lead_leg_post_release_gain: float | None = None  # own post-release gain for the lead leg (None = the same)
    lead_knee_post_release_gain: float | None = None  # own post-release gain for the lead knee (None = the lead leg's)
    back_leg_post_release_gain: float | None = None  # own post-release gain for the back (pivot) leg
    trunk_post_release_gain: float | None = None  # own post-release gain for the waist joints
    # recovery practice: episodes that start from stored real release states (follow-through only)
    recovery_prob: float = 0.3
    recovery_buffer_size: int = 4096
    recovery_min_states: int = 256
    recovery_states_file: str = ""  # load release states from a file (balance skill) instead of collecting them
    # late recovery starts: also store states from inside the follow-through (each step, per env in the window, with
    # this probability; they share the buffer with the release states)
    late_store_prob: float = 0.0
    late_start_window: tuple = (0.3, 0.7)  # s after the release
    late_store_min_head_z: float = 1.0  # only states that are still up
    w_action_rate: float = 0.002
    late_release_s: float = 0.0  # > 0: no release by the reference release + this ends the episode (penalty w_late_release)
    w_late_release: float = 30.0
    w_lost: float = 0.0  # losing the reference before the release (ends the episode) - else it is a free escape
    scripted_release_s: float | None = None  # fingers forced open at the reference release + this (None = policy)
    # lead foot planted at the capture's spot from its landing on, and release timing (0 = off)
    w_lead_plant: float = 0.0  # per step x exp(-(lead foot - reference)^2 / sigma^2), from lead_plant_t on
    lead_plant_t: float = 0.83  # s, capture: the lead foot lands at 0.83 s
    sigma_lead_plant: float = 0.08  # m
    lead_path_t: float = 0.40  # s, wide-band lead-foot tracking from here on (the swing)
    sigma_lead_wide: float = 0.30  # m
    w_release_time: float = 0.0  # at the release x exp(-((t - reference release) / sigma)^2)
    sigma_release_time: float = 0.03  # s
    # centre of mass and footing like the athlete (0 = off); reference from mocap/centroidal.py
    w_com: float = 0.0  # per step x exp(-(COM error / sigma)^2) x exp(-(COM velocity error / sigma_v)^2)
    sigma_com: float = 0.10  # m
    sigma_com_vel: float = 0.6  # m/s
    w_contact: float = 0.0  # per step x share of feet whose ground contact matches the athlete's
    contact_force_n: float = 30.0  # N on a foot = planted
    # balance state in the observation (+16): foot contacts, feet relative to the pelvis, COM position / velocity and
    # capture point relative to the lead foot, all in the pelvis heading frame (the policy and critic saw none of it)
    balance_obs: bool = False
    contact_bodies: str = ".*_foot"  # contact-sensor bodies (regex)
    contact_history: int = 16  # physics steps (20 ms) a foot counts as planted after a contact
    w_slip: float = 0.0  # per step x speed (m/s) of planted feet
    w_support: float = 0.0  # per step (from the lead-foot landing on) x capture point over the planted feet
    support_radius: float = 0.12  # m around a foot centre counts as support
    sigma_support: float = 0.15  # m
    sigma_support_wide: float = 0.60  # m
    # pitching mechanics (kinetic chain) rewards, 0 = off; see scripts/diag_pitch.py (sections 4, 5)
    w_arm_launch: float = 0.0  # throwing hand relative to the shoulder like the athlete's (launch position at foot strike)
    arm_window: tuple = (0.75, 0.98)  # s, reference time
    w_chain: float = 0.0  # pelvis / trunk rotation speeds, shoulder internal rotation and elbow extension speeds
    chain_window: tuple = (0.70, 1.05)
    sigma_seg_w: float = 4.0  # rad/s (pelvis, trunk about the vertical)
    sigma_arm_w: float = 15.0  # rad/s (shoulder internal rotation, elbow)
    w_drive: float = 0.0  # pivot-leg hip / knee angles like the athlete's from drive_window[0] to the release
    drive_window: tuple = (0.45, 0.98)
    sigma_drive: float = 0.35  # rad (root-sum-square over 3 joints)
    w_brace: float = 0.0  # lead-leg brace: COM not sinking faster than the athlete's from foot strike to release + 0.15 s
    w_trunk_release: float = 0.0  # trunk orientation like the athlete's around the release (no diving)
    # soft, planted lead-foot landing (penalties, 0 = off)
    w_soft_descent: float = 0.0  # x (downward lead-foot speed above soft_landing_speed)^2 just before touch-down
    soft_landing_speed: float = 1.0  # m/s
    w_lead_stay: float = 0.0  # lead foot off the ground / moving up while the athlete's lead foot is planted
    w_impact: float = 0.0  # x (lead-foot contact force above impact_limit_bw) in body weights
    impact_limit_bw: float = 3.0
    w_back_down: float = 0.0  # back foot off the mound (+ its height / 5 cm) once the athlete's is down after the release
    w_capture_step: float = 0.0  # back foot planted after the release near the capture point (catch the forward momentum)
    sigma_capture_step: float = 0.25  # m
    w_back_place: float = 0.0  # back foot at the reference back-foot position after the release (task space, wide + narrow)
    w_lead_knee: float = 0.0  # lead knee no more bent than the reference after the release (straight post)

    # ---------------------------------------------------------------- play / evaluation
    play_mode: bool = False  # ball flies with PhysX + aerodynamic forces; zone drawn; results printed
    terminate_on_cross: bool = True


@configclass
class PitchPlayEnvCfg(PitchEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    episode_length_s = 4.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class PitchElasticEnvCfg(PitchEnvCfg):
    """Human-strength pitcher with a tendon-like elastic throwing arm; obs += 7 spring deflections."""

    robot: ArticulationCfg = make_elastic_pitcher_cfg(sea_dt=1 / 800)
    observation_space = 198


@configclass
class PitchElasticPlayEnvCfg(PitchElasticEnvCfg):
    play_mode: bool = True
    rsi_prob: float = 0.0
    episode_length_s = 4.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class PitchBalanceEnvCfg(PitchElasticEnvCfg):
    """Follow-through balance skill: a separate policy that takes over at the release.

    Every episode starts from a recorded real release of the pitching policy (scripts/collect_release_states.py);
    the reward is staying upright and settling, the follow-through reference is only a weak guide.
    """

    recovery_states_file: str = os.path.join(ROOT_DIR, "data", "pitch_release_states.pt")
    recovery_prob: float = 1.0
    recovery_min_states: int = 1
    follow_through_s: float = 1.5
    episode_length_s = 3.5
    w_post_release_track: float = 0.1
    post_release_speed_pow: float = 0.0
    post_release_residual_gain: float = 4.0  # legs +-1 rad around the follow-through reference: room for a step
    w_balance: float = 1.0
    w_fall_after_release: float = 30.0


@configclass
class PitchBalancePlayEnvCfg(PitchBalanceEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)


@configclass
class PitchHandoverEnvCfg(PitchElasticEnvCfg):
    """Pitching skill only: the follow-through belongs to the balance skill (AIB-PitchBalance-v0), which takes
    over at the release, so episodes end shortly after it and nothing after the release is rewarded."""

    follow_through_s: float = 0.1
    recovery_prob: float = 0.0
    w_balance: float = 0.0
    w_post_release_track: float = 0.0
    w_fall_after_release: float = 0.0


@configclass
class PitchStandFirstEnvCfg(PitchElasticEnvCfg):
    """Stage 1 on the mound: stand after the pitch first, speed second (the user's order).

    The follow-through is rewarded whatever the release speed, a fall costs twice a good pitch, the lead foot is
    pulled onto the capture's landing spot and the release toward the capture's timing; the speed reward saturates
    at 100 km/h (stage 2 raises it back to 120 km/h).
    """

    target_speed: float = 100.0 * KMH
    follow_through_s: float = 1.5
    episode_length_s = 3.5
    # standing after a soft lob pays less (with a reference-only follow-through the policy delayed the release to 1.28 s
    # and lobbed at 23 km/h: holding the ball keeps control, releasing hands the body to the reference)
    post_release_speed_pow: float = 2.0
    late_release_s: float = 0.12
    # escapes must cost more than releasing and falling (-60): holding the ball (-30) and losing tracking (free) were
    # preferred, release fell to 32% and 11/32 pitches lost tracking before the release
    w_late_release: float = 100.0
    w_lost: float = 100.0
    # the policy kept holding the ball (full pitches released at 1.27 s / 25 km/h even with a -100 late penalty):
    # stage 1 is about standing, so the fingers open on the athlete's schedule
    scripted_release_s: float | None = 0.0
    # the back foot hovered 3-7 cm above the mound after the follow-through (reference joint angles alone do not reach the
    # ground from the robot's lower, more tilted body); the robot ended on one leg and fell ~1 s after the release
    # 2.0 (capture-step finish: 1.0 was too little to swing the foot past the lead foot). With the pro finish the policy
    # used the 2x authority to kick the back foot ~1 m up (reference 0.4 m) and kept it 0.6-1 m behind: back to 1.0 so
    # the reference swing (up, then forward beside the lead foot) leads
    back_leg_post_release_gain: float | None = 1.0
    w_back_place: float = 0.5  # 2.0 under imitation; a light guide in balance mode
    w_back_down: float = 2.0
    # pro finish (the user's reference videos): balance over a straight lead leg, the back leg up behind as a counterweight,
    # then down beside the lead foot. The capture step (back foot straight through, planted near the capture point) is off:
    # the back foot is meant to stay up until ~0.7 s after the release now
    w_capture_step: float = 0.0
    w_lead_knee: float = 0.0  # 1.0 under imitation (no knee bent past the reference); the knee is free in balance mode
    # half the trunk authority back after the release: with a reference-only trunk it kept pitching forward
    # (to ~100-110 deg) in every follow-through variant and the forward rotation was never braked
    trunk_post_release_gain: float | None = 1.0  # 0.5 under imitation
    w_post_release_track: float = 0.2  # 1.0 under imitation: now only a weak pose prior
    w_balance: float = 1.0
    w_fall_after_release: float = 60.0
    # no residual after the release: the follow-through is the reference alone. Any post-release residual drove the
    # trunk / back leg into a forward pitch that lifted the planted lead foot (falls +0.36 s after the release);
    # reference-only kept the lead foot planted 82% of the time and delayed falls to +0.89 s. The policy has to arrive
    # at the release in a state the reference follow-through can stand on.
    post_release_residual_gain: float = 1.0  # 0.0 under imitation (balance mode: every joint can act)
    w_lead_plant: float = 0.0  # 1.0 (athlete's spot); the user, 2026-10-11: the lead foot need not land where the athlete's did
    w_release_time: float = 20.0
    w_com: float = 0.5
    w_contact: float = 0.5
    w_slip: float = 0.3
    w_support: float = 0.5
    # the new flat-foot reference: every full pitch lost tracking at 0.84-0.95 s (lead foot not yet down) and ended just
    # before the release, so landing, release and follow-through were never experienced
    max_key_err: float = 0.8
    # mid-pitch starts from the end of the stride (0.6 s) to just before the release: the wind-up is learned (strike
    # timing within 1 ms), the follow-through is not. With the ~30% release-state starts, ~28% full pitches remain
    # 0.6 (17350-17950): full pitches fell to ~28% of training and the full-pitch standing rate declined (17650 6/16 ->
    # 17950 0/16) while the training failure rate stayed at 0.29; 0.4 keeps ~42% full pitches
    # 0.1 (20050): the late-stride reference starts landed the lead foot on the athlete's line (training touchdown error
    # ~0.10 m) and hid that full pitches kept it 0.24 m off; full pitches ~63% of training now
    rsi_prob: float = 0.1
    rsi_min_s: float = 0.6
    # kinetic chain (two pitching-mechanics articles): arm up at foot strike, proximal-to-distal timing, pivot-leg
    # drive, a braced lead leg (damped, not softened: the knee flipped between +-250 Nm and the foot bounced)
    # + elastic trunk (waist yaw / pitch SEAs): obs += 2 spring deflections
    # leg PD stiffness x0.6 (what-if: half the stiffness cut the lead-foot bounce by a third; depenetration was not it)
    robot: ArticulationCfg = make_elastic_pitcher_cfg(sea_dt=1 / 800, leg_damping={"hip": 150.0, "knee": 150.0,
                                                                                   "ankle": 60.0}, waist_sea=True,
                                                      leg_stiffness={"hip": 1200.0, "knee": 1200.0, "ankle": 600.0})
    # 200 + 16 balance inputs (2026-10-10 plan step 1: push-recovery work feeds COM / capture point to the networks;
    # without them the policy cannot tell how far the COM has run past the lead foot)
    balance_obs: bool = True
    observation_space = 216
    w_arm_launch: float = 1.0
    w_chain: float = 1.0
    w_drive: float = 0.5
    w_brace: float = 0.5
    w_trunk_release: float = 0.5
    # soft landing (the lead foot hit at -2.5 m/s / ~8 BW and bounced up +0.75 m/s; 44% stood after the release)
    w_soft_descent: float = 0.5
    w_lead_stay: float = 2.0  # 0.5 did not stop the learned lead-foot lift after the release
    # the lead leg only tracked the reference after the release (0.0): with full residual authority the policy lifted the
    # planted foot (COM then ran past it). Plan step 2 (balance obs in place): hip + ankle get 0.4 back for ankle /
    # hip strategy (at 17950 the lead foot rocked onto its heel, toe 15-20 deg up, with no way to press it down); the
    # knee stays reference-only (the lift came through the knee)
    lead_leg_post_release_gain: float = 1.0  # 0.4 hip/ankle, 0.0 knee under imitation
    lead_knee_post_release_gain: float | None = 1.0
    # plan step 3: at 18250 the falls happen +0.6..1.0 s after the release, after the back foot is down (the robot folds
    # forward into a crouch), but every practice start was the release itself. ~0.01 per step stores about as many
    # in-follow-through states (+0.3..0.7 s) as release states, so half the recovery starts begin part-way through
    late_store_prob: float = 0.01
    # 2026-10-11: balance after the release instead of imitation. With the robot started from the reference's own state
    # at 0.95 s and pure PD on the reference, 0/64 stood (trunk dove to 72 deg, the lead foot lifted, forward spin
    # 31-37 kg m^2/s vs ~4 in the reference, fell +1.36 s): the synthesized follow-through is not dynamically feasible,
    # and imitating it pinned the policy to a falling motion. Full residual authority, no velocity feed-forward, the
    # reference pose is only the PD base and a weak prior
    post_release_balance: bool = True
    post_release_feedforward: float = 0.0
    w_stand_support: float = 1.0
    w_stand_spin: float = 1.0
    w_stand_upright: float = 1.0
    w_stand_feet: float = 1.0
    w_stand_still: float = 0.5
    # lead foot sideways on the athlete's line, COM over it (approved 2026-10-11): standing must be possible at the release.
    # The user, 2026-10-11: the lead foot need not follow the athlete -> athlete-relative terms off (were 2.0 / 10.0 /
    # 1.0 / hip roll 1.0); the capture point over the robot's own lead foot instead
    w_lead_lateral: float = 0.0
    w_lead_land: float = 0.0
    w_com_lateral: float = 0.0
    w_cp_lateral: float = 2.0
    w_land_cp: float = 10.0
    # 20050: no progress (0.24 m). The foot is off BEFORE touchdown: the lead hip roll swings 15-23 deg past the athlete's
    # at 0.6-0.7 s (low torque: chosen), then 300 Nm (the limit) cannot bring it back by 0.8 s. Track the foot's sideways
    # path through the swing (out to -0.84 m at 0.6 s, back to -0.51 m) and the lead hip roll itself
    lead_lateral_window: tuple = (0.3, 0.3)
    w_lead_hip_roll: float = 0.0
    w_stand_pivot: float = 0.3  # the user's pivot idea, as a light guide (pelvis heading toward the finish heading)
    # releases under min_release_speed (base 15 m/s = 54 km/h) count as drops and END the episode at -5, while a fall after
    # the release costs -60. Stage 1 releases at ~34 km/h (median, 18100-18550), so half the training episodes
    # (drop_rate 0.50) ended at the release: full pitches never practised the follow-through, and a slow toss was the
    # safe way out of it. Only real drops (< 18 km/h) end the episode now; slow pitches stand for less (speed_pow 2)
    min_release_speed: float = 5.0
    w_impact: float = 0.2


@configclass
class PitchSpeedEnvCfg(PitchStandFirstEnvCfg):
    """Stage 2: the pitcher stays up (stage 1); now the speed reward saturates at 120 km/h again."""

    target_speed: float = 120.0 * KMH


@configclass
class PitchStand2EnvCfg(PitchStandFirstEnvCfg):
    """Stage 1, reward v2 (docs/reward_redesign_stage1.md, approved 2026-10-11): any motion that throws the ball toward
    the plate and comes to rest standing is a good motion. Outcome rewards (ball progress toward the plate, standing at
    release + 1.5 s, falls), physics guidance on the robot's own feet (capture point in the support, spin, height) and
    imitation only before 0.5 s. Three attempts on the old 30+-term reward could not move the lead foot (always
    ~0.25 m toward the throwing-arm side, 2-3/64 release states stoppable); per-step imitation (hundreds of points)
    outweighed the fall penalty (60)."""

    reward_v2: bool = True
    free_leg_gain: float = 2.0
    freeze_pd_after_release: bool = True
    # escapes must cost at least as much as a fall: not releasing, dropping the ball
    late_release_s: float = 0.3
    w_late_release: float = 200.0
    w_drop: float = 200.0
    min_release_speed: float = 2.0  # only a ball that slips out counts as a drop (speed is not rewarded in stage 1)
    max_key_err: float = 99.0  # no tracking-loss termination: the motion is free after 0.5 s
    rsi_prob: float = 0.0  # mid-pitch reference starts put the feet where the athlete's were
    recovery_prob: float = 0.3
    # quality penalties scaled to <= ~10 per episode (measured on 20200: action rate -36, impact -42, slip -38, i.e. as
    # large as the throw reward): x 1/4
    w_action_rate: float = 0.0005
    w_impact: float = 0.05
    w_slip_v2: float = 0.025


@configclass
class PitchStand3EnvCfg(PitchStand2EnvCfg):
    """Stage 1, reward v2 + survival shaping (plan A, approved 2026-10-12). After 300 iterations of v2 every pitch still
    fell (falls moved earlier, +0.57 -> +0.40 s): the -200 fall was the same for every pitch (no advantage), the +200
    stand was never sampled, and the per-step penalties after the release made falling early cheaper. Now: +100 spread
    over 1.5 s for staying up after the release, a fall costs more the earlier it comes (200 at the release, 100 at the
    end), post-release penalties halved."""

    w_alive_v2: float = 100.0
    fall_graded: bool = True
    post_release_pen_scale: float = 0.5
    release_after_lead_strike: bool = True  # added 2026-10-12 after the from-scratch run released at 0.07 s
    w_early_release: float = 200.0


@configclass
class PitchStand4EnvCfg(PitchStand3EnvCfg):
    """Stage 1, + settling shaping (2026-10-13). At 1050 (noise capped) 59/64 pitches were still up 1.5 s after the
    release but 0/64 at rest: trunk tilt > 35 deg 59/64, COM still moving 54/64. The stand bonus is checked once, at
    1.5 s, and was never sampled; per step from 0.5 s after the release: trunk upright, COM still, both feet down, head
    up (each 0..1, averaged)."""

    w_settle_v2: float = 100.0
    lead_land_after_lift: bool = True


@configclass
class PitchStand5EnvCfg(PitchStand4EnvCfg):
    """Stage 1, + a release counts only after a real stride (2026-10-13). At 1500 the policy tapped the lead foot down
    in place and released at 0.485 s, before its own stride landed (0.603 s); still up 1.5 s after the release fell
    from 40/64 (1350) to 2/64."""

    lead_stride_min_m: float = 0.5


@configclass
class PitchStand6EnvCfg(PitchStand5EnvCfg):
    """Stage 1, + a standing PD base after the release (2026-10-13). With freeze_pd_after_release the PD base stayed
    at the reference pose of the robot's own release time - at 0.45-0.56 s that is the athlete's leg lift (lead knee
    76 deg, trunk 27-35 deg forward). The trunk came up to 5 deg at +0.35 s and was pulled back to 33-50 deg by +0.5 s
    in every report since 1050; v8 1500: 0/64 still up, falls +1.06 s after the release."""

    stand_pd_blend_s: float = 0.4


@configclass
class PitchStand7EnvCfg(PitchStand6EnvCfg):
    """Stage 1, standing PD base with the robot's own leg angles (2026-10-13). v10 1800 with zero residual after the
    release (PD base alone): the athlete's release legs (lead hip -91 deg) lifted the lead foot 50 cm in 0.5 s and
    the trunk went 29 -> 57 deg, falls at +0.66 s."""

    stand_pd_legs: str = "hold"
