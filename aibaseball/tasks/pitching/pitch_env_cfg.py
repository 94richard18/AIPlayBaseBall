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
    residual_scale: float = 0.5
    vel_feedforward: float = 1.0
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
    # recovery practice: episodes that start from stored real release states (follow-through only)
    recovery_prob: float = 0.3
    recovery_buffer_size: int = 4096
    recovery_min_states: int = 256
    recovery_states_file: str = ""  # load release states from a file (balance skill) instead of collecting them
    w_action_rate: float = 0.002
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
    post_release_speed_pow: float = 0.0
    w_post_release_track: float = 1.0
    w_balance: float = 1.0
    w_fall_after_release: float = 60.0
    post_release_residual_gain: float = 2.0
    w_lead_plant: float = 1.0
    w_release_time: float = 20.0
    w_com: float = 0.5
    w_contact: float = 0.5
    w_slip: float = 0.3
    w_support: float = 0.5
    # the new flat-foot reference: every full pitch lost tracking at 0.84-0.95 s (lead foot not yet down) and ended just
    # before the release, so landing, release and follow-through were never experienced
    max_key_err: float = 0.8
    rsi_prob: float = 0.4
    # kinetic chain (two pitching-mechanics articles): arm up at foot strike, proximal-to-distal timing, pivot-leg
    # drive, a braced lead leg (damped, not softened: the knee flipped between +-250 Nm and the foot bounced)
    # + elastic trunk (waist yaw / pitch SEAs): obs += 2 spring deflections
    # leg PD stiffness x0.6 (what-if: half the stiffness cut the lead-foot bounce by a third; depenetration was not it)
    robot: ArticulationCfg = make_elastic_pitcher_cfg(sea_dt=1 / 800, leg_damping={"hip": 150.0, "knee": 150.0,
                                                                                   "ankle": 60.0}, waist_sea=True,
                                                      leg_stiffness={"hip": 1200.0, "knee": 1200.0, "ankle": 600.0})
    observation_space = 200
    w_arm_launch: float = 1.0
    w_chain: float = 1.0
    w_drive: float = 0.5
    w_brace: float = 0.5
    w_trunk_release: float = 0.5
    # soft landing (the lead foot hit at -2.5 m/s / ~8 BW and bounced up +0.75 m/s; 44% stood after the release)
    w_soft_descent: float = 0.5
    w_lead_stay: float = 0.5
    w_impact: float = 0.2


@configclass
class PitchSpeedEnvCfg(PitchStandFirstEnvCfg):
    """Stage 2: the pitcher stays up (stage 1); now the speed reward saturates at 120 km/h again."""

    target_speed: float = 120.0 * KMH
