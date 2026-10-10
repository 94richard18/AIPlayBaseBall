"""Pitching a fastball (target_speed, default 120 km/h) into an 18.44 m away strike zone with the five-finger AIB-1.

* The ball is a free PhysX rigid body held only by finger contacts and friction (four-seam grip).
  Grip force comes from finger PD targets set past the contact surface (`grip_squeeze`); the policy
  controls the 16 right-hand finger joints, so holding, the release instant and the spin are learned.
* The body imitates a retargeted OBP fastball (residual PD targets + velocity feed-forward + RSI);
  `speed` can play the reference faster than the athlete.
* On release, the trajectory to the plate is integrated with the drag + Magnus model; the reward
  uses release speed and where the ball crosses the plate plane (strike zone 65 x 95 cm).
* Play mode: the ball flies in PhysX with aerodynamic forces applied every physics step.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_rotate_inverse

from aibaseball.mocap.centroidal import mound_z, reference_centroidal
from aibaseball.mocap.motion import PITCH_KEY_BODIES, MotionRef
from aibaseball.physics import PhysicsSpec
from aibaseball.physics.ball_flight import aero_acceleration, simulate_to_plane
from aibaseball.robot.aib1_cfg import BODY_JOINT_EXPR, GROUP_JOINTS, PITCHER_GRIP
from aibaseball.robot.ball_grip import FINGER_JOINTS, BallGrip

from .pitch_env_cfg import PitchEnvCfg

# finger joints squeezed past contact to create grip force
SQUEEZE_JOINTS = ("index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "ring_pip", "thumb_cmc_flex",
                  "thumb_mcp")


class PitchEnv(DirectRLEnv):
    cfg: PitchEnvCfg

    def __init__(self, cfg: PitchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        self.phys = PhysicsSpec()
        self.ref = MotionRef(cfg.motion_file, dev)
        self.ref.rotate_z(math.radians(cfg.ref_yaw_deg))
        self.release_ref = self.ref.contact_time  # reference release time (stored as "contact")
        # reference COM / planted feet (robot masses, fitted mound), rotated like the reference
        cen = reference_centroidal(cfg.motion_file, lambda x: mound_z(x, cfg.mound_top_z, cfg.mound_height,
                                                                       cfg.mound_slope, cfg.mound_slope_start_x))
        yaw = math.radians(cfg.ref_yaw_deg)
        Rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
                          device=dev)
        self.ref_com = torch.tensor(cen["com"], dtype=torch.float32, device=dev) @ Rz.T
        self.ref_com_vel = torch.tensor(cen["com_vel"], dtype=torch.float32, device=dev) @ Rz.T
        self.ref_contact = torch.tensor(cen["contact"], dtype=torch.float32, device=dev)  # (T, 2): left, right on the ground
        self.ref_still = torch.tensor(cen["still"], dtype=torch.float32, device=dev)  # planted and not sliding
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self.ref_hand_rel = f32(cen["hand_rel"]) @ Rz.T  # throwing hand - throwing shoulder
        self.ref_torso_up = f32(cen["torso_up"]) @ Rz.T
        self.ref_seg_w = torch.stack([f32(cen["pelvis_wz"]), f32(cen["torso_wz"])], -1)  # rad/s about the vertical
        lead_on = cen["contact"][:, 0] & (cen["time"] > 0.3)
        self.ref_lead_strike = float(cen["time"][lead_on.argmax()])

        # joints
        ids, names = self.robot.find_joints(BODY_JOINT_EXPR, preserve_order=True)
        self.body_ids, self.body_names = torch.tensor(ids, device=dev), names
        f_ids, _ = self.robot.find_joints([f"r_{j}" for j in FINGER_JOINTS], preserve_order=True)
        self.finger_ids = torch.tensor(f_ids, device=dev)
        name_to_ref = {nm: i for i, nm in enumerate(self.ref.joint_names)}
        self.ref_cols = torch.tensor([name_to_ref[nm] for nm in self.body_names], device=dev)
        scale = torch.zeros(len(names), device=dev)
        for group, s in cfg.action_scale.items():
            for i, nm in enumerate(names):
                if any(re.fullmatch(e, nm) for e in GROUP_JOINTS[group]):
                    scale[i] = s
        assert (scale > 0).all()
        self.action_scale = scale
        lim = self.robot.data.soft_joint_pos_limits[0]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]

        grip = BallGrip.load(PITCHER_GRIP)
        self.ball_local = torch.tensor(grip.ball_center_hand, device=dev).expand(n, 3)
        self.grip_q = self.robot.data.default_joint_pos[0, self.finger_ids].clone()
        squeeze = torch.tensor([cfg.grip_squeeze if j in SQUEEZE_JOINTS else 0.0 for j in FINGER_JOINTS], device=dev)
        self.hold_q = torch.clamp(self.grip_q + squeeze, self.q_lo[self.finger_ids], self.q_hi[self.finger_ids])

        self.hand_id = self.robot.find_bodies("r_hand")[0][0]
        # finger pads that can push the ball: (body id, pad point in the body frame)
        from aibaseball.robot.ball_grip import TIPS as GRIP_TIPS

        self.pad_ids = [self.robot.find_bodies(GRIP_TIPS[k][0])[0][0] for k in ("index", "middle", "ring")]
        self.pad_ids.append(self.robot.find_bodies("r_thumb_distal")[0][0])
        pads = [GRIP_TIPS[k][1] for k in ("index", "middle", "ring")] + [[0.0, 0.0, 0.0]]
        self.pad_local = torch.tensor(pads, dtype=torch.float32, device=dev).unsqueeze(0).expand(n, 4, 3)
        self.pad_hist = torch.zeros(n, 8, device=dev)  # max pad speed over the last 8 physics steps
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]
        self.torso_id = self.robot.find_bodies("torso")[0][0]
        self.key_ids = [self.robot.find_bodies(b)[0][0] for b in PITCH_KEY_BODIES]
        self.foot_ids = [self.robot.find_bodies(f"{s}_foot")[0][0] for s in "lr"]
        self.foot_sensor_ids = [self.feet.find_bodies(f"{s}_foot")[0][0] for s in "lr"]
        self.body_mass = self.robot.root_physx_view.get_masses().to(dev)  # (n, bodies)
        self.sh_id = self.robot.find_bodies("r_shoulder_pitch_link")[0][0]
        jn = self.robot.joint_names
        self.chain_joints = [jn.index("r_shoulder_yaw"), jn.index("r_elbow")]  # internal rotation, elbow extension
        bn = list(self.body_names)
        self.drive_cols = [bn.index(j) for j in ("r_hip_pitch", "r_hip_yaw", "r_knee")]  # pivot-leg drive
        self.chain_cols = [bn.index("r_shoulder_yaw"), bn.index("r_elbow")]
        self.lead_leg_cols = [i for i, j in enumerate(bn) if j.startswith("l_") and any(k in j for k in ("hip", "knee", "ankle"))]
        self.back_leg_cols = [i for i, j in enumerate(bn) if j.startswith("r_") and any(k in j for k in ("hip", "knee", "ankle"))]
        self.trunk_cols = [i for i, j in enumerate(bn) if j.startswith("waist_")]

        # buffers
        self.actions = torch.zeros(n, cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.q_target = self.robot.data.default_joint_pos.clone()
        self.finger_prev = self.q_target[:, self.finger_ids].clone() if hasattr(self, "finger_ids") else None
        self.finger_next = self.finger_prev
        self.substep = 0
        self.qd_target = torch.zeros_like(self.q_target)
        self.t0 = torch.zeros(n, device=dev)
        self.speed = float(cfg.speed)
        self.released = torch.zeros(n, dtype=torch.bool, device=dev)
        self.new_release = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.release_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.rel_vel = torch.zeros(n, 3, device=dev)
        self.rel_omega = torch.zeros(n, 3, device=dev)
        self.rel_pos = torch.zeros(n, 3, device=dev)
        self.release_reward = torch.zeros(n, device=dev)
        self.crossed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fallen = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lost = torch.zeros(n, dtype=torch.bool, device=dev)
        # release-state buffer: states captured right after real releases, used to start "recovery" episodes
        # that only practise the follow-through (one release per pitch is too few balance-recovery samples)
        cap, nj = cfg.recovery_buffer_size, self.robot.num_joints
        self.rb_root = torch.zeros(cap, 13, device=dev)  # root state relative to the env origin
        self.rb_qpos = torch.zeros(cap, nj, device=dev)
        self.rb_qvel = torch.zeros(cap, nj, device=dev)
        self.rb_t = torch.zeros(cap, device=dev)  # reference time at the release
        self.rb_gain = torch.zeros(cap, device=dev)  # post-release reward gain of that release
        self.post_gain = torch.ones(n, device=dev)  # (release speed / target)^p: standing up after a soft toss pays little
        self.rb_sea = None  # (motor_pos, motor_vel) per SEA actuator
        self.rb_speed = torch.zeros(cap, device=dev)  # release speed (m/s)
        self.rb_count, self.rb_head = 0, 0
        if cfg.recovery_states_file:
            self.load_release_states(cfg.recovery_states_file)
        self.recovery_ep = torch.zeros(n, dtype=torch.bool, device=dev)
        zc = self.field_z + cfg.zone_bottom + 0.5 * cfg.zone_height
        self.zone_center = torch.tensor([cfg.zone_center_y, zc], device=dev)
        self.stats = {k: torch.zeros((), device=dev) for k in (
            "release_rate", "drop_rate", "fall_rate", "release_kmh", "strike_rate", "zone_dist_m", "success_rate",
            "spin_rpm", "backspin_rpm", "release_time_err", "lead_foot_err_m", "com_err_m", "contact_match", "foot_slip_mps",
            "capture_out_m", "back_place_err_m", "capture_step_err_m", "lead_descent_mps", "lead_lift_share", "lead_impact_bw", "arm_launch_err_m", "trunk_err_deg", "com_drop_mps", "drive_err_rad", "chain_score", "track_reward", "key_err_m", "hold_gap_mm",
            "post_release_fail", "recovery_fall", "recovery_share")}
        self.stats["speed"] = torch.tensor(self.speed, device=dev)
        self.stats["eject_excess"] = torch.zeros((), device=dev)
        self.raw_speed = torch.zeros(n, device=dev)
        self.play_log: list[dict] = []

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        c = self.cfg
        self.field_z = c.mound_top_z - c.mound_height if c.mound_height > 0 else 0.0  # plate-level ground
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        ground = GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9))
        if self.cfg.play_mode:
            ground.size = (200.0, 200.0)
            ground.color = (0.16, 0.36, 0.14)
        spawn_ground_plane("/World/ground", ground, translation=(0.0, 0.0, self.field_z))
        self._spawn_mound("/World/envs/env_0")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/" + self.cfg.contact_bodies,
                                                   update_period=0.0, history_length=self.cfg.contact_history))
        self.scene.sensors["feet"] = self.feet
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)
        if self.cfg.play_mode:
            self._spawn_zone_markers()

    def _spawn_mound(self, env_path: str):
        """Static mound per env: a flat top around the rubber (z = mound_top_z) and a 1:12 slope down to the field."""
        c = self.cfg
        h = c.mound_height
        if h <= 0:
            return
        # spikes on mound clay: "max" combine so the mound's friction is the contact's friction
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=c.mound_static_friction, dynamic_friction=c.mound_dynamic_friction,
                                             friction_combine_mode="max")
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.24))
        x0, w = c.mound_slope_start_x, 2.4
        top = sim_utils.CuboidCfg(size=(x0 + 1.2, w, h), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=mat, visual_material=look)
        top.func(f"{env_path}/MoundTop", top, translation=((x0 - 1.2) / 2, 0.0, c.mound_top_z - h / 2))
        run = h / c.mound_slope
        th = math.atan(c.mound_slope)
        length, thick = math.hypot(run, h), 0.3
        slope = sim_utils.CuboidCfg(size=(length, w, thick), collision_props=sim_utils.CollisionPropertiesCfg(),
                                    physics_material=mat, visual_material=look)
        mid = (x0 + run / 2, c.mound_top_z - h / 2)  # middle of the top surface
        n = (math.sin(th), math.cos(th))  # its normal
        slope.func(f"{env_path}/MoundSlope", slope, translation=(mid[0] - n[0] * thick / 2, 0.0, mid[1] - n[1] * thick / 2),
                   orientation=(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0))

    def _spawn_zone_markers(self):
        """Strike-zone frame (yellow) at the plate plane, home plate (white), crossing points (red)."""
        c = self.cfg
        self.zone_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/zone",
            markers={
                "edge": sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0), emissive_color=(0.8, 0.6, 0.0))),
                "plate": sim_utils.CuboidCfg(size=(0.43, 0.43, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.95, 0.95))),
                "rubber": sim_utils.CuboidCfg(size=(0.15, 0.61, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.95, 0.95))),
            },
        ))
        self.cross_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/cross",
            markers={"hit": sim_utils.SphereCfg(radius=BALL_R, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.15, 0.1)))},
        ))
        self.cross_markers.set_visibility(False)
        pts, idx = [], []
        x = c.plate_distance
        y0, y1 = c.zone_center_y - c.zone_width / 2, c.zone_center_y + c.zone_width / 2
        z0 = self.field_z + c.zone_bottom
        z1 = z0 + c.zone_height
        for k in range(40):
            a = k / 39
            pts += [[x, y0 + a * (y1 - y0), z0], [x, y0 + a * (y1 - y0), z1], [x, y0, z0 + a * (z1 - z0)],
                    [x, y1, z0 + a * (z1 - z0)]]
            idx += [0, 0, 0, 0]
        pts.append([x + 0.215, c.zone_center_y, self.field_z + 0.005])
        idx.append(1)
        pts.append([-0.075, 0.0, c.mound_top_z + 0.005])
        idx.append(2)
        origin = self.scene.env_origins[0]
        P = torch.tensor(pts, device=self.device) + origin
        self.zone_markers.visualize(translations=P, marker_indices=torch.tensor(idx, device=self.device))

    # ------------------------------------------------------------------ reference helpers
    def _ref_time(self, extra: int = 0) -> torch.Tensor:
        return self.t0 + self.speed * (self.episode_length_buf.float() + extra) * self.step_dt

    def _ref(self, t):
        r = self.ref.sample(t)
        r["joint_pos"] = r["joint_pos"][:, self.ref_cols]
        r["joint_vel"] = r["joint_vel"][:, self.ref_cols] * self.speed
        r["root_lin_vel"] = r["root_lin_vel"] * self.speed
        r["root_ang_vel"] = r["root_ang_vel"] * self.speed
        o = self.scene.env_origins
        r["root_pos"] = r["root_pos"] + o
        r["key_pos"] = r["key_pos"] + o.unsqueeze(1)
        return r

    def _ref_centroid(self, t: torch.Tensor):
        """Reference COM position / velocity (world) and planted feet (n, 2: left, right) at reference time t."""
        i0, i1, a = self.ref._idx(t)
        com = self.ref_com[i0] * (1 - a) + self.ref_com[i1] * a + self.scene.env_origins
        vel = (self.ref_com_vel[i0] * (1 - a) + self.ref_com_vel[i1] * a) * self.speed
        near = a < 0.5
        contact = torch.where(near, self.ref_contact[i0], self.ref_contact[i1]) > 0.5
        self._ref_still_now = torch.where(near, self.ref_still[i0], self.ref_still[i1]) > 0.5
        return com, vel, contact

    def _com(self):
        d = self.robot.data
        w = self.body_mass.unsqueeze(-1)
        M = self.body_mass.sum(-1, keepdim=True)
        return (d.body_com_pos_w * w).sum(1) / M, (d.body_com_lin_vel_w * w).sum(1) / M

    def _head_z(self) -> torch.Tensor:
        """Height of the head centre (0.62 m up the torso link axis)."""
        q = self.robot.data.body_link_quat_w[:, self.torso_id]
        up = quat_apply(q, torch.tensor([0.0, 0.0, 0.62], device=self.device).expand(q.shape[0], 3))
        return self.robot.data.body_link_pos_w[:, self.torso_id, 2] + up[:, 2]

    def _grip_point(self):
        d = self.robot.data
        return d.body_link_pos_w[:, self.hand_id] + quat_apply(d.body_link_quat_w[:, self.hand_id], self.ball_local)

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        if c.speed_ramp_steps > 0:
            self.speed = c.speed + (c.speed_end - c.speed) * min(1.0, self.common_step_counter / c.speed_ramp_steps)
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-2.0, 2.0)
        nxt = self._ref(self._ref_time(1))
        # after the release (no reference follow-through: the OBP trial ends 0.145 s later) the policy gets
        # more authority to step / brace and recover its balance
        # per joint: the lead leg can keep its own (smaller) post-release gain - with 2x it learned to lift the planted
        # lead foot after the release (planted 26% of the time vs 82% with the reference pose alone)
        post = torch.full((len(self.body_names),), c.post_release_residual_gain, device=self.device)
        if c.lead_leg_post_release_gain is not None:
            post[self.lead_leg_cols] = c.lead_leg_post_release_gain
        if c.back_leg_post_release_gain is not None:
            post[self.back_leg_cols] = c.back_leg_post_release_gain
        if c.trunk_post_release_gain is not None:
            post[self.trunk_cols] = c.trunk_post_release_gain
        scale = torch.where(self.released.unsqueeze(-1), c.residual_scale * post.unsqueeze(0),
                            torch.full_like(post, c.residual_scale).unsqueeze(0))
        body = nxt["joint_pos"] + self.actions[:, :29] * self.action_scale * scale
        self.q_target[:, self.body_ids] = torch.clamp(body, self.q_lo[self.body_ids], self.q_hi[self.body_ids])
        self.qd_target[:, self.body_ids] = nxt["joint_vel"] * c.vel_feedforward
        fing = self.hold_q + self.actions[:, 29:] * c.finger_action_scale
        if c.scripted_release_s is not None:  # fingers open on the reference's schedule (no holding on to the ball)
            opening = (self._ref_time() >= self.release_ref + c.scripted_release_s).unsqueeze(-1)
            fing = torch.where(opening, (self.hold_q - c.finger_action_scale).expand_as(fing), fing)
        self.finger_prev = self.q_target[:, self.finger_ids].clone()
        self.finger_next = torch.clamp(fing, self.q_lo[self.finger_ids], self.q_hi[self.finger_ids])
        self.substep = 0
        self.new_release[:] = False
        self.release_reward[:] = 0.0

    def _apply_action(self):
        if self.cfg.play_mode:
            self._apply_aero()
        # finger targets ramp linearly over the physics sub-steps (sub-control-step release timing)
        self.substep += 1
        a = min(1.0, self.substep / self.cfg.decimation)
        self.q_target[:, self.finger_ids] = self.finger_prev + a * (self.finger_next - self.finger_prev)
        self._record_pad_speed()
        self.robot.set_joint_position_target(self.q_target)
        self.robot.set_joint_velocity_target(self.qd_target)

    def _record_pad_speed(self):
        d = self.robot.data
        q = d.body_link_quat_w[:, self.pad_ids]
        r = quat_apply(q.reshape(-1, 4), self.pad_local.reshape(-1, 3)).view(-1, 4, 3)
        v = d.body_link_lin_vel_w[:, self.pad_ids] + torch.cross(d.body_link_ang_vel_w[:, self.pad_ids], r, dim=-1)
        self.pad_hist = torch.roll(self.pad_hist, 1, dims=1)
        self.pad_hist[:, 0] = v.norm(dim=-1).max(dim=-1).values

    def _apply_aero(self):
        """Play mode: drag + Magnus on released balls (PhysX integrates gravity and contacts)."""
        flying = self.released & ~self.crossed
        v, w = self.ball.data.root_lin_vel_w, self.ball.data.root_ang_vel_w
        f = aero_acceleration(v, w, self.phys) * self.phys.ball.mass * flying.unsqueeze(-1).float()
        fb = quat_rotate_inverse(self.ball.data.root_quat_w, f).unsqueeze(1)
        self.ball.set_external_force_and_torque(fb, torch.zeros_like(fb))

    # ------------------------------------------------------------------ release detection / outcome
    def _check_release(self):
        gap = (self.ball.data.root_pos_w - self._grip_point()).norm(dim=-1)
        new = (~self.released) & (gap > self.cfg.release_distance)
        if new.any():
            ids = new.nonzero(as_tuple=False).squeeze(-1)
            v = self.ball.data.root_lin_vel_w[ids]
            # momentum audit: a ball cannot leave faster than the finger pads that push it. Any surplus is a
            # contact-solver ejection (squeezed sphere) -> removed, and logged as an artefact fraction.
            allowed = self.cfg.pad_speed_tolerance * self.pad_hist[ids].max(dim=-1).values
            speed = v.norm(dim=-1)
            excess = ((speed - allowed) / speed.clamp_min(1e-6)).clamp_min(0.0)
            v = torch.where((speed > allowed).unsqueeze(-1), v * (allowed / speed.clamp_min(1e-6)).unsqueeze(-1), v)
            self.ball.write_root_velocity_to_sim(torch.cat([v, self.ball.data.root_ang_vel_w[ids]], -1), ids)
            self.stats["eject_excess"] = 0.95 * self.stats["eject_excess"] + 0.05 * excess.mean()
            self.raw_speed[ids] = speed
            self.released[ids] = True
            self.new_release[ids] = True
            self.release_step[ids] = self.episode_length_buf[ids]
            self.rel_vel[ids] = v
            self.rel_omega[ids] = self.ball.data.root_ang_vel_w[ids]
            self.rel_pos[ids] = self.ball.data.root_pos_w[ids]
            self.post_gain[ids] = (v.norm(dim=-1) / self.cfg.target_speed).clamp(max=1.0) ** self.cfg.post_release_speed_pow
            slow = v.norm(dim=-1) < self.cfg.min_release_speed
            self.dropped[ids] = slow | (v[:, 0] <= 0)
            if not self.cfg.play_mode and not getattr(self, "rb_frozen", False):
                good = ids[~self.dropped[ids] & ~self.recovery_ep[ids]]
                if len(good) > 0:
                    self._store_release_states(good)
        return gap

    def _seas(self):
        return [a for k, a in self.robot.actuators.items() if k.endswith("_sea")]

    def _store_release_states(self, ids):
        d = self.robot.data
        cap = self.cfg.recovery_buffer_size
        k = len(ids)
        slots = (torch.arange(k, device=self.device) + self.rb_head) % cap
        root = d.root_state_w[ids].clone()
        root[:, :3] -= self.scene.env_origins[ids]
        self.rb_root[slots] = root
        self.rb_qpos[slots] = d.joint_pos[ids]
        self.rb_qvel[slots] = d.joint_vel[ids]
        self.rb_t[slots] = self._ref_time()[ids]
        self.rb_gain[slots] = self.post_gain[ids]
        self.rb_speed[slots] = self.rel_vel[ids].norm(dim=-1)
        seas = self._seas()
        if seas:
            if self.rb_sea is None:
                self.rb_sea = [(torch.zeros(cap, a.num_joints, device=self.device),
                                torch.zeros(cap, a.num_joints, device=self.device)) for a in seas]
            for (bp, bv), a in zip(self.rb_sea, seas):
                bp[slots] = a.motor_pos[ids]
                bv[slots] = a.motor_vel[ids]
        self.rb_head = (self.rb_head + k) % cap
        self.rb_count = min(cap, self.rb_count + k)

    def save_release_states(self, path: str):
        """Write the release-state buffer (starts for the separate follow-through balance skill)."""
        k = self.rb_count
        out = dict(root=self.rb_root[:k], qpos=self.rb_qpos[:k], qvel=self.rb_qvel[:k], t=self.rb_t[:k],
                   speed=self.rb_speed[:k], sea=[(p[:k], v[:k]) for p, v in (self.rb_sea or [])])
        torch.save({key: (val.cpu() if torch.is_tensor(val) else [(p.cpu(), v.cpu()) for p, v in val])
                    for key, val in out.items()}, path)

    def load_release_states(self, path: str):
        d = torch.load(path)
        k = min(len(d["t"]), self.cfg.recovery_buffer_size)
        dev = self.device
        self.rb_root[:k], self.rb_qpos[:k], self.rb_qvel[:k] = d["root"][:k].to(dev), d["qpos"][:k].to(dev), d["qvel"][:k].to(dev)
        self.rb_t[:k], self.rb_speed[:k] = d["t"][:k].to(dev), d["speed"][:k].to(dev)
        self.rb_gain[:k] = 1.0
        if d["sea"]:
            self.rb_sea = []
            for p, v in d["sea"]:
                bp = torch.zeros(self.cfg.recovery_buffer_size, p.shape[1], device=dev)
                bv = torch.zeros_like(bp)
                bp[:k], bv[:k] = p[:k].to(dev), v[:k].to(dev)
                self.rb_sea.append((bp, bv))
        self.rb_count, self.rb_head = k, 0
        self.rb_frozen = True

    def _start_recovery(self, env_ids):
        """Start these envs right after a stored real release (ball already gone): follow-through only."""
        n = len(env_ids)
        idx = torch.randint(0, self.rb_count, (n,), device=self.device)
        root = self.rb_root[idx].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(self.rb_qpos[idx], self.rb_qvel[idx], env_ids=env_ids)
        for (bp, bv), a in zip(self.rb_sea or [], self._seas()):
            a.motor_pos[env_ids] = bp[idx]
            a.motor_vel[env_ids] = bv[idx]
            a.needs_sync[env_ids] = False
        self.t0[env_ids] = self.rb_t[idx]
        self.post_gain[env_ids] = self.rb_gain[idx]
        self.q_target[env_ids] = self.rb_qpos[idx]
        self.qd_target[env_ids] = 0.0
        open_q = self.robot.data.joint_pos[env_ids][:, self.finger_ids]
        self.finger_prev[env_ids] = open_q
        self.finger_next[env_ids] = open_q
        self.released[env_ids] = True  # the ball is gone: no release reward, follow-through window starts now
        self.release_step[env_ids] = 0
        far = self.scene.env_origins[env_ids] + torch.tensor([10.0, 0.0, 1.0], device=self.device)
        ball = torch.cat([far, torch.tensor([1.0, 0, 0, 0], device=self.device).expand(n, 4),
                          torch.zeros(n, 6, device=self.device)], -1)
        self.ball.write_root_state_to_sim(ball, env_ids)

    def _release_outcome(self, ids):
        c = self.cfg
        o = self.scene.env_origins[ids]
        p_local = self.rel_pos[ids] - o
        # a few balls per step: the CPU is ~5x faster than many tiny GPU kernel launches (Windows WDDM)
        cross, _, t_cross, reached = simulate_to_plane(p_local.cpu(), self.rel_vel[ids].cpu(), self.rel_omega[ids].cpu(),
                                                       self.phys, c.plate_distance, ground_z=self.field_z)
        cross, t_cross, reached = cross.to(self.device), t_cross.to(self.device), reached.to(self.device)
        yz = cross[:, 1:]
        half = torch.tensor([c.zone_width / 2, c.zone_height / 2], device=self.device) + self.phys.ball.radius
        off = (yz - self.zone_center).abs()
        strike = reached & (off <= half).all(-1)
        # miss distance; balls short of the plate also count the missing x distance
        short = (c.plate_distance - cross[:, 0]).clamp_min(0.0)
        dist = torch.sqrt(((yz - self.zone_center) ** 2).sum(-1) + short**2)
        speed = self.rel_vel[ids].norm(dim=-1)
        return dict(cross=cross, strike=strike, dist=dist, speed=speed, reached=reached, t_cross=t_cross)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        hq = d.body_link_quat_w[:, self.hand_id]
        ball_rel = quat_rotate_inverse(hq, self.ball.data.root_pos_w - self._grip_point())
        ball_vrel = quat_rotate_inverse(hq, self.ball.data.root_lin_vel_w - d.body_link_lin_vel_w[:, self.hand_id])
        t = self._ref_time()
        nxt = self._ref(self._ref_time(1))
        q = d.joint_pos
        obs = torch.cat([
            q[:, self.body_ids] - nxt["joint_pos"],
            d.joint_vel[:, self.body_ids] * 0.05,
            q[:, self.finger_ids] - self.grip_q,
            d.joint_vel[:, self.finger_ids] * 0.05,
            d.projected_gravity_b,
            d.root_lin_vel_b * 0.5,
            d.root_ang_vel_b * 0.25,
            d.root_link_pos_w[:, 2:3] - 1.0,
            ball_rel * 10.0,
            ball_vrel * 0.1,
            self.ball.data.root_ang_vel_w * 0.01,
            self.released.float().unsqueeze(-1),
            (t / self.cfg.phase_duration_s).unsqueeze(-1),
            nxt["joint_pos"] - q[:, self.body_ids],
            quat_rotate_inverse(d.root_link_quat_w, nxt["root_pos"] - d.root_link_pos_w),
            quat_rotate_inverse(d.root_link_quat_w, nxt["key_pos"][:, -1] - self.ball.data.root_pos_w),
            self.actions,
        ] + self._sea_obs(), -1)
        return {"policy": torch.nan_to_num(obs)}

    def _sea_obs(self) -> list:
        """Elastic-arm variant: stretched tendon state (spring deflection, rad) of the throwing arm."""
        seas = [a for k, a in self.robot.actuators.items() if k.endswith("_sea")]
        return [torch.cat([a.deflection for a in seas], -1) * 2.0] if seas else []

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.robot.data
        r = self._ref(self._ref_time())
        q = d.joint_pos[:, self.body_ids]
        pose_err = ((q - r["joint_pos"]) ** 2).mean(-1)
        vel_err = ((d.joint_vel[:, self.body_ids] - r["joint_vel"]) ** 2).mean(-1)
        key = torch.cat([d.body_link_pos_w[:, self.key_ids], self.ball.data.root_pos_w.unsqueeze(1)], 1)
        key_err = ((key - r["key_pos"]) ** 2).sum(-1).mean(-1)
        root_err = ((d.root_link_pos_w - r["root_pos"]) ** 2).sum(-1)
        rot_err = quat_error_magnitude(d.root_link_quat_w, r["root_quat"]) ** 2
        track = (c.w_pose * torch.exp(-pose_err / c.sigma_pose ** 2) + c.w_vel * torch.exp(-vel_err / c.sigma_vel ** 2)
                 + c.w_key * torch.exp(-key_err / c.sigma_key ** 2)
                 + c.w_root * torch.exp(-root_err / c.sigma_root ** 2 - rot_err / c.sigma_rot ** 2))
        self._key_err = key_err.sqrt()
        # lead foot planted where and when the capture puts it (the pitcher used to keep it in the air past the release)
        # wide + narrow band: a foot 0.5 m off gets no signal from the 8 cm band alone; the wide band follows the swing
        plant_on = self._ref_time() >= c.lead_plant_t
        path_on = self._ref_time() >= c.lead_path_t
        lead_err = (d.body_link_pos_w[:, self.key_ids[2]] - r["key_pos"][:, 2]).norm(dim=-1)
        plant = (plant_on.float() * torch.exp(-((lead_err / c.sigma_lead_plant) ** 2))
                 + path_on.float() * torch.exp(-((lead_err / c.sigma_lead_wide) ** 2)))

        gap = (self.ball.data.root_pos_w - self._grip_point()).norm(dim=-1)
        before = self._ref_time() < self.release_ref
        hold = (~self.released & before).float() * torch.exp(-((gap / 0.02) ** 2))

        # after the release: stay up. Heights, not pelvis tilt: a pitcher's pelvis is tilted 45-65 deg around the
        # release (the reference itself) while perfectly balanced, so a tilt term punished the real motion.
        pelvis_z = d.body_link_pos_w[:, self.pelvis_id, 2]
        head_z = self._head_z()
        ang_v = d.root_ang_vel_b.norm(dim=-1)
        balance = (torch.exp(-((head_z - c.post_release_head_z).clamp(max=0.0) / 0.20) ** 2)
                   * torch.exp(-((pelvis_z - c.post_release_pelvis_z).clamp(max=0.0) / 0.15) ** 2)
                   * torch.exp(-(ang_v / 8.0) ** 2))
        # centre of mass and footing like the athlete: COM path (incl. the braking at foot strike), each foot on the
        # ground exactly when the athlete's is, no sliding while planted, capture point over the planted feet
        com, com_v = self._com()
        rc, rv, rcontact = self._ref_centroid(self._ref_time())
        com_err = (com - rc).norm(dim=-1)
        com_rew = (torch.exp(-((com_err / c.sigma_com) ** 2))
                   * torch.exp(-(((com_v - rv).norm(dim=-1) / c.sigma_com_vel) ** 2)))
        # a planted foot chatters (contact on/off every few physics steps): planted = touched within the history window
        forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values  # (n, 2)
        planted = forces > c.contact_force_n
        contact_match = (planted == rcontact).float().mean(-1)
        foot_v = d.body_link_lin_vel_w[:, self.foot_ids, :2].norm(dim=-1)
        slip = ((planted & self._ref_still_now).float() * foot_v).sum(-1)  # the dragging pivot foot may slide
        o = self.scene.env_origins
        n_env = com.shape[0]
        fwd = torch.tensor([0.04, 0.0, 0.0], device=self.device).expand(n_env, 2, 3)
        feet_xy = d.body_link_pos_w[:, self.foot_ids, :2] + quat_apply(d.body_link_quat_w[:, self.foot_ids], fwd)[..., :2]
        x_loc = com[:, 0] - o[:, 0]
        g_z = torch.where(x_loc < c.mound_slope_start_x, torch.full_like(x_loc, c.mound_top_z),
                          (c.mound_top_z - (x_loc - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
        h = (com[:, 2] - g_z).clamp_min(0.3)
        cp = com[:, :2] + com_v[:, :2] * torch.sqrt(h / 9.81).unsqueeze(-1)
        d_feet = (cp.unsqueeze(1) - feet_xy).norm(dim=-1) - c.support_radius  # (n, 2)
        ab = feet_xy[:, 1] - feet_xy[:, 0]
        u = (((cp - feet_xy[:, 0]) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0, 1)
        d_seg = (cp - (feet_xy[:, 0] + u.unsqueeze(-1) * ab)).norm(dim=-1) - c.support_radius
        d_sup = torch.where(planted.all(-1), d_seg, torch.where(planted[:, 0], d_feet[:, 0],
                            torch.where(planted[:, 1], d_feet[:, 1], torch.ones_like(d_seg)))).clamp_min(0.0)
        support_on = self._ref_time() >= c.lead_plant_t
        support = support_on.float() * 0.5 * (torch.exp(-((d_sup / c.sigma_support) ** 2))
                                               + torch.exp(-((d_sup / c.sigma_support_wide) ** 2)))
        footing = c.w_com * com_rew + c.w_contact * contact_match + c.w_support * support - c.w_slip * slip

        # pitching mechanics (kinetic chain): arm in the launch position at foot strike, pelvis / trunk / arm speeds on
        # the athlete's timeline, pivot-leg drive, lead-leg brace (COM not dropping), trunk not diving at the release
        tr = self._ref_time()
        i0, i1, a_ = self.ref._idx(tr)
        lerp = lambda x: x[i0] * (1 - a_) + x[i1] * a_ if x.dim() > 1 else x[i0] * (1 - a_.squeeze(-1)) + x[i1] * a_.squeeze(-1)  # noqa: E731
        hand_rel = d.body_link_pos_w[:, self.hand_id] - d.body_link_pos_w[:, self.sh_id]
        arm_err = (hand_rel - lerp(self.ref_hand_rel)).norm(dim=-1)
        arm_on = ((tr >= c.arm_window[0]) & (tr <= c.arm_window[1])).float()
        arm = arm_on * 0.5 * (torch.exp(-(arm_err / 0.10) ** 2) + torch.exp(-(arm_err / 0.30) ** 2))
        seg_ref = lerp(self.ref_seg_w)
        seg = torch.stack([d.root_ang_vel_w[:, 2], d.body_ang_vel_w[:, self.torso_id, 2]], -1)
        jv_ref = r["joint_vel"][:, self.chain_cols]
        jv = d.joint_vel[:, self.chain_joints]
        chain_on = ((tr >= c.chain_window[0]) & (tr <= c.chain_window[1])).float()
        chain = chain_on * 0.25 * (torch.exp(-((seg - seg_ref) / c.sigma_seg_w) ** 2).sum(-1)
                                   + torch.exp(-((jv - jv_ref) / c.sigma_arm_w) ** 2).sum(-1))
        q_drive = d.joint_pos[:, self.body_ids[self.drive_cols]]
        drive_err = ((q_drive - r["joint_pos"][:, self.drive_cols]) ** 2).sum(-1)
        drive_on = ((tr >= c.drive_window[0]) & (tr <= self.release_ref)).float()
        drive = drive_on * torch.exp(-drive_err / c.sigma_drive ** 2)
        brace_on = ((tr >= self.ref_lead_strike) & (tr <= self.release_ref + 0.15)).float()
        drop = (rv[:, 2] - com_v[:, 2]).clamp_min(0.0)  # COM sinking faster than the athlete's
        brace = brace_on * torch.exp(-(drop / 0.4) ** 2)
        up = quat_apply(d.body_link_quat_w[:, self.torso_id], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(
            n_env, 3))
        trunk_err = torch.acos((up * lerp(self.ref_torso_up)).sum(-1).clamp(-1, 1))
        trunk_on = ((tr >= self.release_ref - 0.1) & (tr <= self.release_ref + 0.15)).float()
        trunk = trunk_on * torch.exp(-(trunk_err / 0.3) ** 2)
        mech = (c.w_arm_launch * arm + c.w_chain * chain + c.w_drive * drive + c.w_brace * brace
                + c.w_trunk_release * trunk)
        # soft, planted lead-foot landing (the foot came down at -2.5 m/s, hit ~8 body weights and bounced back up):
        # slow descent when about to land, no lifting / upward speed once the athlete's lead foot is down, no impacts
        lf = self.foot_ids[0]
        lf_x = d.body_link_pos_w[:, lf, 0] - o[:, 0]
        ground_lf = torch.where(lf_x < c.mound_slope_start_x, torch.full_like(lf_x, c.mound_top_z),
                                (c.mound_top_z - (lf_x - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
        lf_gap = d.body_link_pos_w[:, lf, 2] - 0.08 - ground_lf  # sole (flat foot) above the mound
        lf_vz = d.body_link_lin_vel_w[:, lf, 2]
        approach = (~planted[:, 0]) & (lf_gap < 0.12) & (tr > 0.6) & (tr < self.release_ref)
        descent = approach.float() * (((-lf_vz) - c.soft_landing_speed).clamp_min(0.0) ** 2)
        lead_down = rcontact[:, 0] & (tr >= self.ref_lead_strike)
        lift = lead_down.float() * ((~planted[:, 0]).float() + (lf_vz.clamp_min(0.0) / 0.3) ** 2)
        bw = self.body_mass.sum(-1) * 9.81
        lead_force = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids[0]].norm(dim=-1).max(1).values
        impact = (lead_force / bw - c.impact_limit_bw).clamp_min(0.0)
        # back foot down once the athlete's is (end of the follow-through): it hovered 7-11 cm above the mound
        back_due = self.released & rcontact[:, 1] & (tr > self.release_ref + 0.2)
        rf = self.foot_ids[1]
        rf_x = d.body_link_pos_w[:, rf, 0] - o[:, 0]
        ground_rf = torch.where(rf_x < c.mound_slope_start_x, torch.full_like(rf_x, c.mound_top_z),
                                (c.mound_top_z - (rf_x - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
        rf_gap = (d.body_link_pos_w[:, rf, 2] - 0.08 - ground_rf).clamp_min(0.0)
        back_up = back_due.float() * ((~planted[:, 1]).float() + (rf_gap / 0.05).clamp(max=3.0))
        landing = c.w_soft_descent * descent + c.w_lead_stay * lift + c.w_impact * impact + c.w_back_down * back_up
        # capture step: the back foot, once down after the release, where the forward momentum can be caught
        step_on = self.released & planted[:, 1] & (tr > self.release_ref + 0.25) & (tr < self.release_ref + 1.0)
        step_err = (feet_xy[:, 1] - cp).norm(dim=-1)
        capture_step = step_on.float() * torch.exp(-((step_err / c.sigma_capture_step) ** 2))
        # back foot WHERE the athlete's goes (task space): through past the lead foot and down. Reference joint angles on
        # the robot's lower, more bent body left it level with the lead foot and 11 cm up; the policy held it back further
        place_on = (self.released & (tr > self.release_ref + 0.25)).float()
        place_err = (d.body_link_pos_w[:, self.foot_ids[1]] - r["key_pos"][:, 3]).norm(dim=-1)
        back_place = place_on * 0.5 * (torch.exp(-((place_err / 0.10) ** 2)) + torch.exp(-((place_err / 0.35) ** 2)))

        track = torch.where(self.released, c.w_post_release_track * self.post_gain * track, track)
        rew = (c.w_track * track + c.w_hold * hold + self.release_reward + c.w_lead_plant * plant + footing + mech
               - landing + c.w_capture_step * capture_step + c.w_back_place * back_place
               + c.w_balance * self.released.float() * self.post_gain * balance
               - c.w_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1)
               - c.w_drop * (self.new_release & self.dropped).float()
               - c.w_fall * self.fallen.float()
               # a real fall within the follow-through window costs as much as a good pitch earns (the policy
               # used to dive off the mound: "lost" ended those episodes first, without any penalty)
               - c.w_fall_after_release * (self.released & self.fallen).float()
               - c.w_late_release * getattr(self, "_late", torch.zeros_like(self.released)).float()
               - c.w_lost * self._lost.float())

        a = 0.01
        self.stats["track_reward"] = (1 - a) * self.stats["track_reward"] + a * track.mean()
        self.stats["key_err_m"] = (1 - a) * self.stats["key_err_m"] + a * key_err.sqrt().mean()
        for k, val in (("com_err_m", com_err.mean()), ("contact_match", contact_match.mean()),
                       ("foot_slip_mps", slip.mean())):
            self.stats[k] = (1 - a) * self.stats[k] + a * val
        for k, val, on in (("back_place_err_m", place_err, place_on), ("capture_step_err_m", step_err, step_on.float()), ("lead_descent_mps", -lf_vz, approach.float()), ("lead_lift_share", (~planted[:, 0]).float(),
                            lead_down.float()), ("lead_impact_bw", lead_force / bw, planted[:, 0].float()),
                           ("arm_launch_err_m", arm_err, arm_on), ("trunk_err_deg", trunk_err * 57.3, trunk_on),
                           ("com_drop_mps", drop, brace_on), ("drive_err_rad", drive_err.sqrt(), drive_on)):
            if on.any():
                self.stats[k] = (1 - a) * self.stats[k] + a * val[on > 0].mean()
        for k, val in (("chain_score", (chain / chain_on.clamp_min(1e-6))[chain_on > 0].mean() if chain_on.any()
                        else self.stats["chain_score"]),):
            self.stats[k] = (1 - a) * self.stats[k] + a * val
        if support_on.any():
            self.stats["capture_out_m"] = (1 - a) * self.stats["capture_out_m"] + a * d_sup[support_on].mean()
        if plant_on.any():
            self.stats["lead_foot_err_m"] = (1 - a) * self.stats["lead_foot_err_m"] + a * lead_err[plant_on].mean()
        held = ~self.released
        if held.any():
            self.stats["hold_gap_mm"] = (1 - a) * self.stats["hold_gap_mm"] + a * gap[held].mean() * 1000
        done = self.reset_terminated | self.reset_time_outs
        if done.any():
            b = 0.02
            self.stats["release_rate"] = (1 - b) * self.stats["release_rate"] + b * (self.released & ~self.dropped)[done].float().mean()
            self.stats["drop_rate"] = (1 - b) * self.stats["drop_rate"] + b * self.dropped[done].float().mean()
            self.stats["fall_rate"] = (1 - b) * self.stats["fall_rate"] + b * self.fallen[done].float().mean()
            # honest follow-through metric: episodes that ended by falling OR losing the reference after release
            fail = (self.released & self.fallen)[done].float().mean()
            rec = done & self.recovery_ep
            if rec.any():
                self.stats["recovery_fall"] = (1 - b) * self.stats["recovery_fall"] + b * self.fallen[rec].float().mean()
            self.stats["recovery_share"] = (1 - b) * self.stats["recovery_share"] + b * self.recovery_ep[done].float().mean()
            self.stats["post_release_fail"] = (1 - b) * self.stats["post_release_fail"] + b * fail
        self.stats["speed"] = torch.tensor(self.speed, device=self.device)
        self.extras["log"] = {f"pitch/{k}": v.clone() for k, v in self.stats.items()}
        return rew

    # ------------------------------------------------------------------ dones
    def _get_dones(self):
        c = self.cfg
        self._check_release()
        good = self.new_release & ~self.dropped
        if good.any():
            ids = good.nonzero(as_tuple=False).squeeze(-1)
            out = self._release_outcome(ids)
            speed_n = (out["speed"] / c.target_speed).clamp(0, 1.0)  # no extra credit above target_speed
            fast = out["speed"] >= c.target_speed
            # overspeed costs control (faster arm -> larger direction error per ms of release timing)
            over = ((out["speed"] - c.speed_soft_cap) / 3.0).clamp_min(0.0)
            r = (c.w_speed * speed_n + c.w_speed_bonus * fast.float() + c.w_strike * out["strike"].float()
                 - c.w_overspeed * over
                 + c.w_zone * torch.exp(-((out["dist"] / 0.4) ** 2)) + c.w_aim * torch.exp(-((out["dist"] / 2.0) ** 2))
                 + c.w_aim_wide * torch.exp(-((out["dist"] / 8.0) ** 2))
                 + c.w_both * (fast & out["strike"]).float())
            t_rel = self.t0[ids] + self.speed * self.release_step[ids].float() * self.step_dt
            r = r + c.w_release_time * torch.exp(-(((t_rel - self.release_ref) / c.sigma_release_time) ** 2))
            self.release_reward[ids] = r
            w = self.rel_omega[ids]
            v = self.rel_vel[ids]
            vh = torch.nn.functional.normalize(torch.cat([v[:, :2], torch.zeros_like(v[:, :1])], -1), dim=-1)
            back_axis = torch.cross(vh, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(vh), dim=-1)
            backspin = (w * back_axis).sum(-1) * 60 / (2 * math.pi)
            a = 0.05
            for k, val in (("release_kmh", out["speed"].mean() * 3.6), ("strike_rate", out["strike"].float().mean()),
                           ("zone_dist_m", out["dist"].mean()), ("success_rate", (fast & out["strike"]).float().mean()),
                           ("spin_rpm", w.norm(dim=-1).mean() * 60 / (2 * math.pi)), ("backspin_rpm", backspin.mean()),
                           ("release_time_err", (t_rel - self.release_ref).mean())):
                self.stats[k] = (1 - a) * self.stats[k] + a * val
            if c.play_mode:
                for j, i in enumerate(ids.tolist()):
                    rec = dict(env=i, release_kmh=float(out["speed"][j] * 3.6), raw_kmh=float(self.raw_speed[i] * 3.6),
                               plate_y=float(out["cross"][j, 1]), plate_z=float(out["cross"][j, 2]),
                               strike=bool(out["strike"][j]), flight_time=float(out["t_cross"][j]),
                               spin_rpm=float(w[j].norm() * 60 / (2 * math.pi)), backspin_rpm=float(backspin[j]))
                    self.play_log.append(rec)
                    ok = "STRIKE" if rec["strike"] else "ball"
                    print(f"[pitch] env {i}: {rec['release_kmh']:5.1f} km/h (raw {rec['raw_kmh']:5.1f}) | plate y {rec['plate_y']:+.2f} m, "
                          f"z {rec['plate_z']:.2f} m -> {ok} | spin {rec['spin_rpm']:.0f} rpm (backspin "
                          f"{rec['backspin_rpm']:+.0f}) | flight {rec['flight_time']:.3f} s", flush=True)

        d = self.robot.data
        # fallen = pelvis or head near the ground (the old "pelvis tilt > 60 deg" fired on the reference posture)
        self.fallen = (d.body_link_pos_w[:, self.pelvis_id, 2] < self.cfg.fallen_pelvis_z) | (self._head_z() < self.cfg.fallen_head_z)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        follow = int(round(c.follow_through_s / self.step_dt))
        if c.play_mode:
            ball = self.ball.data.root_pos_w - self.scene.env_origins
            newly = self.released & ~self.crossed & (ball[:, 0] >= c.plate_distance)
            if newly.any():
                self.cross_markers.set_visibility(True)
                self.cross_markers.visualize(translations=self.ball.data.root_pos_w[newly])
            self.crossed |= newly | (self.released & (ball[:, 2] < self.field_z + 0.05))
            terminated = self.crossed & torch.tensor(c.terminate_on_cross, device=self.device)
            return terminated, time_out
        lost = self._key_err > c.max_key_err if hasattr(self, "_key_err") else torch.zeros_like(self.released)
        # After the release the reference only holds its last frame, so a real follow-through must drift away
        # from it: tracking loss ends the episode only before the release; afterwards only real falls count.
        lost = lost & ~self.released
        self._lost = lost
        done_follow = self.released & (self.episode_length_buf - self.release_step >= follow)
        # the reference holds its last (post-release) pose; keep going through the follow-through window
        ref_end = self._ref_time(1) >= self.ref.duration + c.follow_through_s
        # late release: holding the ball past the reference release (+ late_release_s) ends the pitch
        self._late = (~self.released) & (self._ref_time() > self.release_ref + c.late_release_s) if c.late_release_s > 0             else torch.zeros_like(self.released)
        terminated = self.fallen | (self.released & self.dropped) | done_follow | lost | self._late
        return terminated, time_out | ref_end

    # ------------------------------------------------------------------ reset (reference state initialisation)
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n, dev, c = len(env_ids), self.device, self.cfg
        rsi = torch.rand(n, device=dev) < c.rsi_prob
        t_max = max(0.0, self.release_ref - c.rsi_margin_before_release)
        self.t0[env_ids] = torch.where(rsi, torch.rand(n, device=dev) * t_max, torch.zeros(n, device=dev))
        r = self.ref.sample(self.t0[env_ids])
        o = self.scene.env_origins[env_ids]
        root = torch.cat([r["root_pos"] + o, r["root_quat"], r["root_lin_vel"] * self.speed, r["root_ang_vel"] * self.speed], -1)
        self.robot.write_root_state_to_sim(root, env_ids)
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        jp[:, self.body_ids] = r["joint_pos"][:, self.ref_cols]
        jv[:, self.body_ids] = r["joint_vel"][:, self.ref_cols] * self.speed
        jp[:, self.finger_ids] = self.grip_q
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.q_target[env_ids] = jp
        self.q_target[env_ids.unsqueeze(-1), self.finger_ids] = self.hold_q
        self.finger_prev[env_ids] = self.hold_q
        self.finger_next[env_ids] = self.hold_q
        self.qd_target[env_ids] = jv
        # ball in the grip: reference ball position / velocity (last key point)
        t = self.t0[env_ids]
        r2 = self.ref.sample(t + 1e-3)
        ball_v = (r2["key_pos"][:, -1] - r["key_pos"][:, -1]) / 1e-3 * self.speed
        ball = torch.cat([r["key_pos"][:, -1] + o, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4), ball_v,
                          torch.zeros(n, 3, device=dev)], -1)
        self.ball.write_root_state_to_sim(ball, env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        for buf in (self.released, self.new_release, self.dropped, self.crossed):
            buf[env_ids] = False
        self.release_step[env_ids] = 0
        self.recovery_ep[env_ids] = False
        if not c.play_mode and self.rb_count >= c.recovery_min_states and c.recovery_prob > 0:
            rec = env_ids[torch.rand(n, device=dev) < c.recovery_prob]
            if len(rec) > 0:
                self._start_recovery(rec)
                self.recovery_ep[rec] = True


BALL_R = PhysicsSpec().ball.radius
