"""ONE OUTS joint scene (user 2026-10-28, Route 2): the P3c pitcher and the batter in one env, one ball.

One episode = one pitch. The pitcher is the P3c deliverable (ho2 7600 on the TW2PL task, release 2.5 ms early, plate yaw
18 deg, identical start); the batter (make_aib1_cfg human strength, effort 1.5, implicit PD) stands at the plate in the
plate frame. Ball: PhysX while in the pitcher's hand, then the batter's RK4 drag + Magnus flight from the live release
state (the model the batter was trained under and the pitcher's aero tiers use), mirrored into the same PhysX body."""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from aibaseball.robot.aib1_cfg import make_aib1_cfg
from aibaseball.tasks.pitching.pitch_env_cfg import Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLEnvCfg


@configclass
class JointPitchEnvCfg(Stage1dK70GW2MBASVRT1SYW110R5H5TW2PLEnvCfg):
    # P3c: the gate eval's --set list as defaults
    start_joint_noise: float = 0.0
    start_noise_ramp_steps: int = 0
    release_action_shift_s: float = 0.0025
    plate_yaw_deg: float = 18.0
    ball_aero_sim: bool = False  # the flight is the integrator's (JointPitchEnv), not PhysX + aero forces

    # not replicated: the ball-batter collision filter is set per env on the USD stage
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=False)
    batter: ArticulationCfg = make_aib1_cfg(human_strength=True, effort_scale=1.5).replace(prim_path="/World/envs/env_.*/Batter")
    # batter root in the plate frame (origin: plate centre at field level, +x away from the mound, +y to the plate
    # batter session 10-28: its frame is this plate frame turned 180 deg (pitch along -x, +y third base); clip root0 at
    # x -0.19..-0.38, y +0.87..+0.94 there, i.e. x +0.19..+0.38, y -0.87..-0.94 here; yaw = the clip's + pi
    batter_pose_plate: tuple = (0.3, -0.9, math.pi / 2)
    # gate P3c identical starts: release reference time 0.9975 s for most envs (P 0: episode time = reference time)
    release_t_pred_s: float = 0.9975
    batter_every: int = 4  # batter control every this many pitcher control steps (400 Hz / 4 = 100 Hz, as trained)
    batter_hold: bool = True  # M1: the batter holds its default pose under PD (no policy)
    # after the hand-off: "mirror" = the integrated state written into the PhysX ball every physics step (one body for the
    # camera and the judge); "park" = the PhysX ball parked under the field once, a marker shows the flight (C1 compares
    # the batter's pelvis / foot height in both)
    ball_after_handoff: str = "park"  # planning 10-28: park (nothing functional reads the body after the release)
    # integrated flight ends (the PhysX ball takes over again) on the ground or this far behind the plate
    flight_end_behind_plate_m: float = 3.0
    # film (M3): a 200 m grass-green ground instead of the grey grid - visual only, the physics material is the same
    film_grass: bool = False
    # film (M3): a round visual-only mound (5.5 m across) instead of the wide box copy; the collider is unchanged
    film_round_mound: bool = False
    # film (M3): HDRI sky dome from Nucleus (falls back to the colour dome if it cannot be spawned)
    film_sky: bool = False
    # film (M3, user 10-28): a wall behind the plate with the judge's strike zone (StrikeZone: plate half width + ball
    # radius, 0.50-1.10 m) painted on it - visual only (no collider; the judge reads the plate-plane crossing); the
    # flight marker stops at its face. Distance = the integrated flight's end behind the plate.
    film_wall: bool = False
    film_wall_size_m: tuple = (4.0, 2.5)  # width (centred on the zone), height

    def __post_init__(self):
        super().__post_init__()
        pd, zc = self.plate_distance, self.zone_center_y
        th = -math.radians(self.plate_yaw_deg)  # plate frame -> env frame: Rz(-plate_yaw)
        bx, by, byaw = self.batter_pose_plate
        px, py = pd + bx, zc + by
        x, y = math.cos(th) * px - math.sin(th) * py, math.sin(th) * px + math.cos(th) * py
        field_z = self.mound_top_z - self.mound_height if self.mound_height > 0 else 0.0
        yaw = th + byaw
        s = self.batter.init_state
        self.batter.init_state = s.replace(pos=(x, y, s.pos[2] + field_z), rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
