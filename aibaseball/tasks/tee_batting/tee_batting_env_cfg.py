from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from aibaseball.physics.constants import TARGET_DISTANCE, BallSpec
from aibaseball.robot.aib1_cfg import make_aib1_cfg

BALL = BallSpec()


@configclass
class TeeBattingEnvCfg(DirectRLEnvCfg):
    # ---------------------------------------------------------------- timing
    # 400 Hz physics (bat-ball contact handled analytically with swept detection), 100 Hz policy.
    episode_length_s = 1.5
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**22,
        ),
    )

    # ---------------------------------------------------------------- spaces
    action_space = 29  # legs 12 + waist 3 + arms 14 (fingers hold the grip pose)
    extra_actions: int = 0  # non-joint actions appended at the end (e.g. swing tempo)
    observation_space = 110
    state_space = 0

    # ---------------------------------------------------------------- scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = make_aib1_cfg()
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL.radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.92)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
    )
    tee: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tee",
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=1.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # ---------------------------------------------------------------- task layout (world, Z up, +X = center field)
    tee_height: float = 0.80  # ball centre height [m]
    tee_height_noise: float = 0.03
    tee_xy_noise: float = 0.03
    batter_offset: tuple[float, float] = (-0.20, 0.90)  # robot root xy relative to the tee
    joint_pos_noise: float = 0.02
    target_distance: float = TARGET_DISTANCE
    fair_half_angle_deg: float = 45.0

    # ---------------------------------------------------------------- action scaling [rad per unit action]
    action_scale: dict = {
        "hip": 0.5, "knee": 0.5, "ankle": 0.4,
        "waist": 1.0, "shoulder": 1.5, "elbow": 1.5, "wrist": 1.2,
    }
    action_clip: float = 2.0

    # ---------------------------------------------------------------- termination
    min_pelvis_height: float = 0.55
    max_tilt_cos: float = 0.5  # terminate when the pelvis tilts > 60 deg

    # ---------------------------------------------------------------- reward weights
    w_distance: float = 10.0  # x (carry / target)^2 (fair balls)
    w_success: float = 10.0  # carry >= target and fair
    w_exit_velo: float = 4.0  # x exit speed / 52.5 m/s
    w_launch: float = 4.0  # x EV/52.5 x ((1 + cos(launch dir, 30 deg to center field)) / 2)^2
    w_backspin: float = 0.0  # x EV/52.5 x clamp(backspin_rpm / 2000, -1, 1)
    launch_target_deg: float = 28.0
    launch_sigma_deg: float = 0.0  # > 0: Gaussian launch-angle score instead of the smooth direction score
    foul_factor: float = 0.2
    w_approach: float = 0.0  # potential: -distance(sweet spot, ball); off: it rewarded cheap taps
    w_bat_speed: float = 0.5  # (sweet-spot speed toward the field / 45)^2 near the ball
    w_action_rate: float = 0.002
    w_fall: float = 5.0
    w_fall_after_hit: float = 10.0
    follow_through_s: float = 0.4  # episode continues after contact; the batter must stay up
    w_miss: float = 1.0

    # ---------------------------------------------------------------- perception of the ball (step B)
    ball_obs_bias_std: float = 0.0  # m, per-episode estimation error
    ball_obs_noise_std: float = 0.0  # m, per-step noise
    ball_obs_delay_s: float = 0.0  # s, visuomotor latency
    ball_obs_clip: float = 100.0  # m, clip of the ball-relative observation vectors (pitched balls start ~17 m away)

    # ---------------------------------------------------------------- contact
    contact_substeps: int = 16  # swept-test samples per physics step (bat moves <= ~11 cm per step)

    # ---------------------------------------------------------------- flight model for rewards
    flight_dt: float = 0.04

    # ---------------------------------------------------------------- play / evaluation mode
    # If True the episode continues after contact and the ball flies in the scene (aero model)
    # until it lands; contact reaction is applied to the bat.
    play_mode: bool = False
    terminate_on_land: bool = True  # play mode: end the episode once the ball lands


@configclass
class TeeBattingPlayEnvCfg(TeeBattingEnvCfg):
    play_mode: bool = True
    episode_length_s = 9.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=4.0, replicate_physics=True)
    tee_xy_noise: float = 0.0
    tee_height_noise: float = 0.0
    joint_pos_noise: float = 0.0
