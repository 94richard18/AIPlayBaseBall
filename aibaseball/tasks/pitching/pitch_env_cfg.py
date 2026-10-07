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
    motion_file: str = os.path.join(MOTION_DIR, "pitch_2916-4.npz")
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
    zone_bottom: float = 0.45
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
    w_post_release_track: float = 0.2  # remaining weight of tracking the frozen last reference frame
    post_release_pelvis_z: float = 0.75  # m; lower than this is penalised by the balance term
    post_release_residual_gain: float = 2.0  # residual action authority after the release (step / brace)
    # recovery practice: episodes that start from stored real release states (follow-through only)
    recovery_prob: float = 0.3
    recovery_buffer_size: int = 4096
    recovery_min_states: int = 256
    w_action_rate: float = 0.002

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
