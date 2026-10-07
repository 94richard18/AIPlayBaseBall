"""Isaac Lab ArticulationCfg for the AIB-1 humanoid (requires a running Isaac Sim app)."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .humanoid import ACTUATOR_GROUPS, HUMAN_ACTUATOR_GROUPS
from .stance import Stance

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "aib1")
USD_PATH = os.path.join(ASSET_DIR, "aib1.usd")
STANCE_PATH = os.path.join(ASSET_DIR, "batting_stance.json")
HUMAN_USD_PATH = os.path.join(ASSET_DIR + "_human", "aib1.usd")  # same robot, human arm masses (de Leva)

GROUP_JOINTS = {
    "hip": [".*_hip_.*"],
    "knee": [".*_knee"],
    "ankle": [".*_ankle_.*"],
    "waist": ["waist_.*"],
    "shoulder": [".*_shoulder_.*"],
    "elbow": [".*_elbow"],
    "wrist": [".*_wrist_.*"],
    "finger": [".*_(index|middle|ring|pinky)_.*"],
    "thumb": [".*_thumb_.*"],
}

# Joints the batting policy controls; fingers hold the grip pose.
BODY_JOINT_EXPR = [".*_hip_.*", ".*_knee", ".*_ankle_.*", "waist_.*", ".*_shoulder_.*", ".*_elbow", ".*_wrist_.*"]

# A right-handed batter faces the plate (-Y world): yaw -90 deg about Z.
BATTER_YAW = -math.pi / 2


def _actuators(effort_scale: float = 1.0, groups: dict | None = None) -> dict[str, ImplicitActuatorCfg]:
    groups = groups or ACTUATOR_GROUPS
    acts = {}
    for name, exprs in GROUP_JOINTS.items():
        g = groups[name]
        acts[name] = ImplicitActuatorCfg(
            joint_names_expr=exprs,
            effort_limit_sim=g.effort * (effort_scale if name not in ("finger", "thumb") else 1.0),
            velocity_limit_sim=g.velocity,
            stiffness=g.stiffness,
            damping=g.damping,
            armature=g.armature,
            friction=0.0,
        )
    return acts


def make_aib1_cfg(root_pos=(0.0, 0.0, 1.0), yaw: float = BATTER_YAW, effort_scale: float | None = None,
                  human_strength: bool = False) -> ArticulationCfg:
    """Batter (with bat). `human_strength`: human-level joint torque / speed limits (HUMAN_ACTUATOR_GROUPS) and
    human arm masses (like the pitcher)."""
    if effort_scale is None:  # experiments: AIB_EFFORT_SCALE=2.0
        effort_scale = float(os.environ.get("AIB_EFFORT_SCALE", "1.0"))
    stance = Stance.load(STANCE_PATH)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HUMAN_USD_PATH if human_strength else USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(root_pos[0], root_pos[1], stance.root_height + 0.005),
            rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            joint_pos=dict(stance.joint_pos),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=_actuators(effort_scale, HUMAN_ACTUATOR_GROUPS if human_strength else None),
    )


# ----------------------------------------------------------------------------- pitcher variant (no bat)
PITCHER_DIR = os.path.join(os.path.dirname(ASSET_DIR), "aib1_pitcher")
PITCHER_USD = os.path.join(PITCHER_DIR, "aib1_pitcher.usd")
PITCHER_POSE = os.path.join(PITCHER_DIR, "default_pose.json")
PITCHER_GRIP = os.path.join(PITCHER_DIR, "grip.json")


def make_pitcher_cfg(fix_root: bool = False, effort_scale: float = 1.0, human_strength: bool = True) -> ArticulationCfg:
    """Pitcher variant; by default with human-level joint torque / speed limits (HUMAN_ACTUATOR_GROUPS)."""
    import json

    with open(PITCHER_POSE, encoding="utf-8") as f:
        pose = json.load(f)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PITCHER_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=2,
                sleep_threshold=0.0, stabilization_threshold=0.0, fix_root_link=fix_root,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, pose["root_height"] + 0.005), joint_pos=dict(pose["joint_pos"]), joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators=_actuators(effort_scale, HUMAN_ACTUATOR_GROUPS if human_strength else None),
    )


# ----------------------------------------------------------------------------- elastic throwing arm (tendon model)
# right shoulder (3) / elbow / wrist (3) as series elastic actuators with human-level motor (muscle) limits;
# the joints themselves may move faster than the motors while a stretched tendon recoils.
SEA_PARAMS = {  # group: (spring Nm/rad, joint speed limit rad/s)
    "shoulder": (150.0, 250.0),
    "elbow": (120.0, 100.0),
    "wrist": (40.0, 80.0),
}
SEA_JOINT_EXPR = {"shoulder": ["r_shoulder_.*"], "elbow": ["r_elbow"], "wrist": ["r_wrist_.*"]}


def _pitcher_actuators(sea_dt: float) -> dict:
    from .sea import SeriesElasticActuatorCfg

    acts = _actuators(1.0, HUMAN_ACTUATOR_GROUPS)
    for group in SEA_PARAMS:  # left arm keeps the implicit human-level PD actuators
        acts[group] = acts[group].replace(joint_names_expr=[e.replace(".*_", "l_") for e in GROUP_JOINTS[group]])
    for group, (k_s, joint_vel) in SEA_PARAMS.items():
        g = HUMAN_ACTUATOR_GROUPS[group]
        acts[f"r_{group}_sea"] = SeriesElasticActuatorCfg(
            joint_names_expr=SEA_JOINT_EXPR[group], dt=sea_dt, motor_effort=g.effort, motor_velocity=g.velocity,
            spring_stiffness=k_s, stiffness=0.0, damping=0.0, armature=g.armature, friction=0.0,
            effort_limit_sim=1000.0, velocity_limit_sim=joint_vel,
        )
    return acts


def make_elastic_pitcher_cfg(sea_dt: float) -> ArticulationCfg:
    """Human-strength pitcher whose throwing arm has tendon-like series elasticity."""
    return make_pitcher_cfg().replace(actuators=_pitcher_actuators(sea_dt))
