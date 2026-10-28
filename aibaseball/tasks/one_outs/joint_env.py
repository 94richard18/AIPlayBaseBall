"""ONE OUTS joint scene (user 2026-10-28, Route 2; design approved by the planning session): the P3c pitcher (StageEnv,
obs / release / SEA untouched) and the batter (second articulation "batter", implicit PD) in one env.

Ball: the PhysX body while it is in the pitcher's hand (grip, release detection). From the step the release is detected,
its live state is handed to an RK4 drag + Magnus flight (aibaseball.physics.ball_flight._deriv, the batter's model) and
the result is written back into the same PhysX body every physics step, so the camera, the judge and the pitcher's
buffers read one ball. The batter side (batter session, BatterSide) may own the flight instead (swept bat contact):
set env.batter_side; it then provides ball_world(dt) each physics step. Ball-batter PhysX contact is filtered (a
contact would count twice). The integrated flight ends on the ground or flight_end_behind_plate_m behind the plate,
and PhysX takes the ball back.

One episode = one pitch; no env.reset() between pitches (every later reset drifts, 2026-10-28): one process = 64 envs
= 64 first-after-load pitches."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from aibaseball.physics.ball_flight import _aero_consts, _deriv
from aibaseball.tasks.pitching.stage_env import StageEnv

from .joint_env_cfg import JointPitchEnvCfg


class JointPitchEnv(StageEnv):
    cfg: JointPitchEnvCfg

    def __init__(self, cfg: JointPitchEnvCfg, render_mode: str | None = None, **kwargs):
        self.batter_side = None  # set before the first reset (the env's construction already reset once: call env.reset())
        self.ball_marker = None
        self.marker_envs = None  # park: envs whose flight marker is drawn (None = all)
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.j_fly = torch.zeros(n, dtype=torch.bool, device=dev)  # the integrator owns the ball
        self.j_done = torch.zeros(n, dtype=torch.bool, device=dev)  # flight over (ground / behind the plate)
        self.j_pos = torch.zeros(n, 3, device=dev)  # world frame
        self.j_vel = torch.zeros(n, 3, device=dev)
        self.j_omega = torch.zeros(n, 3, device=dev)
        self.j_prev = torch.zeros(n, 3, device=dev)
        self.j_t = torch.zeros(n, device=dev)  # time since the hand-off
        # hand-off (batter side): plate-frame release state and the episode time it was taken
        self.handoff_pos = torch.zeros(n, 3, device=dev)  # plate frame, z above the field
        self.handoff_vel = torch.zeros(n, 3, device=dev)
        self.handoff_omega = torch.zeros(n, 3, device=dev)
        self.handoff_t = torch.full((n,), float("nan"), device=dev)
        # the batter's swing clock starts before the release: the gate's release reference time, as episode time
        self.release_t_pred = torch.full((n,), cfg.release_t_pred_s / self.speed, device=dev)
        self.j_cross = torch.full((n, 2), float("nan"), device=dev)  # (y - zone centre, z above the field) at the plate
        self._aero_c = _aero_consts(self.phys, self.j_vel)
        self.j_bat_z = torch.tensor([[1e9, -1e9]], device=dev).repeat(n, 1)  # batter root z min / max since the load
        self._j_k, self._ctrl = 0, False  # pitcher control steps since the load; this step is a batter control step

    # ------------------------------------------------------------------ batter side clocks (batter session 10-28):
    # control() at 100 Hz before the pitcher's action, ball_world(dt) every physics step (_fly), post() after the step
    def _pre_physics_step(self, actions: torch.Tensor):
        self._ctrl = self.batter_side is not None and self._j_k % self.cfg.batter_every == 0
        self._j_k += 1
        if self._ctrl:
            self.batter_side.control()
        super()._pre_physics_step(actions)

    def step(self, action):
        out = super().step(action)
        if self._ctrl:
            self.batter_side.post()
        return out

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        # PitchEnv._setup_scene with the batter articulation added before the clone
        c = self.cfg
        self.field_z = c.mound_top_z - c.mound_height if c.mound_height > 0 else 0.0
        self.robot = Articulation(c.robot)
        self.batter = Articulation(c.batter)
        self.ball = RigidObject(c.ball)
        ground = GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9))
        if c.play_mode or c.film_grass:  # colour only (the user's spec: grass in the video); friction unchanged
            ground.size = (200.0, 200.0)
            ground.color = (0.16, 0.36, 0.14)
        spawn_ground_plane("/World/ground", ground, translation=(0.0, 0.0, self.field_z))
        if c.film_grass:  # the tinted grid read as teal tiles (M3 film): a plain grass sheet 2 mm above it, visual only
            sheet = sim_utils.CuboidCfg(size=(200.0, 200.0, 0.002), visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.42, 0.16), roughness=0.9))  # no collision / rigid-body props: nothing physical
            sheet.func("/World/grass", sheet, translation=(0.0, 0.0, self.field_z + 0.002))
        self._spawn_mound("/World/envs/env_0")
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self._filter_ball_batter()
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["batter"] = self.batter
        self.scene.rigid_objects["ball"] = self.ball
        self.feet = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/" + c.contact_bodies,
                                                   update_period=0.0, history_length=c.contact_history))
        self.scene.sensors["feet"] = self.feet
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        sky = None
        if c.film_sky:  # M3 (planning 14): an HDRI sky from Nucleus; any failure falls back to the colour dome
            try:
                from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

                sky = sim_utils.DomeLightCfg(intensity=1000.0, texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr")
                sky.func("/World/Light", sky)
                print("[joint] film_sky: HDRI dome light", flush=True)
            except Exception as e:  # noqa: BLE001 - a missing asset must not stop the film
                print(f"[joint] film_sky failed ({type(e).__name__}: {e}) - colour dome instead", flush=True)
                sky = None
        if sky is None:
            light.func("/World/Light", light)
        if c.play_mode:
            self._spawn_zone_markers()

    def _spawn_mound(self, env_path: str):
        """The colliding mound as in PitchEnv; film_round_mound: the visual copy (two 5.5 m wide boxes, a long plank seen
        from centre field) hidden and replaced by a round visual-only mesh. Height = min(collider profile along the pitch
        line, round mound: 5.5 m across, centre 0.84 m in front of the rubber, flat to 1.3 m, then down to the field), so
        on the pitch line it never rises above the collider (the round slope is steeper) and the front edge stays at the
        collider's; the rubber and the back foot (x >= -0.3 m) are on the flat. Turned with plate_yaw_deg like the copy."""
        super()._spawn_mound(env_path)
        c = self.cfg
        h = c.mound_height
        if not c.film_round_mound or h <= 0:
            return
        import numpy as np
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        for name in ("MoundTopShown", "MoundSlopeShown"):
            p = stage.GetPrimAtPath(f"{env_path}/{name}")
            if p:
                UsdGeom.Imageable(p).MakeInvisible()
        x0, s = c.mound_slope_start_x, c.mound_slope
        cx, R, r0 = x0 + h / s - 2.75, 2.75, 1.3
        xs, ys = np.arange(cx - R - 0.1, cx + R + 0.1001, 0.1), np.arange(-R - 0.1, R + 0.1001, 0.1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        prof = np.where(X <= x0, h, h - s * (X - x0))
        rho = np.hypot(X - cx, Y)
        Z = np.clip(np.minimum(prof, h - h / (R - r0) * np.clip(rho - r0, 0.0, None)), 0.0, None) + 0.003
        yaw = -math.radians(getattr(c, "plate_yaw_deg", 0.0))
        Xr, Yr = math.cos(yaw) * X - math.sin(yaw) * Y, math.sin(yaw) * X + math.cos(yaw) * Y
        nx, ny = X.shape
        pts = [Gf.Vec3f(float(a), float(b), float(z + self.field_z)) for a, b, z in zip(Xr.ravel(), Yr.ravel(), Z.ravel())]
        idx = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                k = i * ny + j
                idx += [k, k + ny, k + ny + 1, k + 1]
        mesh = UsdGeom.Mesh.Define(stage, f"{env_path}/MoundRound")
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * ((nx - 1) * (ny - 1))))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        look = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.24), roughness=0.9)
        look.func(f"{env_path}/MoundRoundLook", look)
        sim_utils.bind_visual_material(f"{env_path}/MoundRound", f"{env_path}/MoundRoundLook")

    def _filter_ball_batter(self):
        """No PhysX contact between each env's ball and its batter (the flight model owns the bat contact)."""
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for i in range(self.scene.num_envs):
            ball = stage.GetPrimAtPath(f"/World/envs/env_{i}/Ball")
            rel = UsdPhysics.FilteredPairsAPI.Apply(ball).CreateFilteredPairsRel()
            rel.AddTarget(f"/World/envs/env_{i}/Batter")

    # ------------------------------------------------------------------ batter + ball each physics step
    def _apply_action(self):
        super()._apply_action()
        if self.cfg.batter_hold and self.batter_side is None:
            self.batter.set_joint_position_target(self.batter.data.default_joint_pos)
            self.batter.set_joint_velocity_target(torch.zeros_like(self.batter.data.default_joint_pos))
        self._fly(self.physics_dt)
        bz = self.batter.data.root_pos_w[:, 2]  # C1: does the ball (mirror / park) disturb the batter's footing
        self.j_bat_z = torch.stack([torch.minimum(self.j_bat_z[:, 0], bz), torch.maximum(self.j_bat_z[:, 1], bz)], -1)

    def _fly(self, dt: float):
        start = self.released & ~self.j_fly & ~self.j_done
        if start.any():  # hand-off: the PhysX ball's state right now
            d = self.ball.data
            self.j_pos[start], self.j_vel[start] = d.root_pos_w[start], d.root_lin_vel_w[start]
            self.j_omega[start] = d.root_ang_vel_w[start]
            self.j_fly |= start
            self.j_t[start] = 0.0
            self.handoff_pos[start] = self.to_plate(self.j_pos[start] - self.scene.env_origins[start])
            self.handoff_pos[start, 2] -= self.field_z
            self.handoff_vel[start] = self.to_plate(self.j_vel[start])
            self.handoff_omega[start] = self.to_plate(self.j_omega[start])
            self.handoff_t[start] = self.episode_length_buf[start].float() * self.step_dt
        if self.batter_side is not None:  # every physics step, every env: it also writes the batter's PD targets
            p, v, w = self.batter_side.ball_world(dt)
        if not self.j_fly.any():
            return
        self.j_prev = self.j_pos.clone()
        if self.batter_side is None:
            p, v, w = self._rk4(self.j_pos, self.j_vel, self.j_omega, dt)
        m = self.j_fly.unsqueeze(-1)
        self.j_pos, self.j_vel, self.j_omega = torch.where(m, p, self.j_pos), torch.where(m, v, self.j_vel), torch.where(m, w, self.j_omega)
        self.j_t += dt * self.j_fly.float()
        # first plate-plane crossing (plate frame x = plate_distance), interpolated
        a = self.to_plate(self.j_prev - self.scene.env_origins)
        b = self.to_plate(self.j_pos - self.scene.env_origins)
        pd = self.cfg.plate_distance
        cr = self.j_fly & torch.isnan(self.j_cross[:, 0]) & (a[:, 0] < pd) & (b[:, 0] >= pd)
        if cr.any():
            f = ((pd - a[cr, 0]) / (b[cr, 0] - a[cr, 0]).clamp_min(1e-9)).unsqueeze(-1)
            q = a[cr] + f * (b[cr] - a[cr])
            self.j_cross[cr, 0] = q[:, 1] - self.cfg.zone_center_y
            self.j_cross[cr, 1] = q[:, 2] - self.field_z
        end = self.j_fly & ((b[:, 2] <= self.field_z + self.phys.ball.radius) | (b[:, 0] > pd + self.cfg.flight_end_behind_plate_m))
        if self.cfg.ball_after_handoff == "mirror":
            ids = self.j_fly.nonzero(as_tuple=False).squeeze(-1)
            if len(ids):  # into the PhysX body (pose + velocities: the next physics step starts from them)
                st = torch.cat([self.j_pos[ids], self.ball.data.root_quat_w[ids], self.j_vel[ids], self.j_omega[ids]], -1)
                self.ball.write_root_state_to_sim(st, ids)
        else:  # "park" (the batter env's play mode: a ball teleported every physics step once broke the robot's ground
            # contact): the PhysX ball goes 5 m under the field once, a marker shows the flight
            if start.any():
                ids = start.nonzero(as_tuple=False).squeeze(-1)
                st = torch.zeros(len(ids), 13, device=self.device)
                st[:, :3] = self.scene.env_origins[ids] + torch.tensor([0.0, 0.0, self.field_z - 5.0], device=self.device)
                st[:, 3] = 1.0
                self.ball.write_root_state_to_sim(st, ids)
            self._show_marker()
        self.j_fly &= ~end
        self.j_done |= end

    def _show_marker(self):
        """park: the integrated ball as a marker (marker_envs: the envs to draw, the film sets its one env)."""
        if self.ball_marker is None:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self.ball_marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/joint_ball", markers={
                "ball": sim_utils.SphereCfg(radius=self.phys.ball.radius, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.95, 0.92)))}))
        ids = torch.arange(self.num_envs, device=self.device) if self.marker_envs is None else torch.as_tensor(
            self.marker_envs, device=self.device)
        on = (self.j_fly | self.j_done)[ids]
        p = torch.where(on.unsqueeze(-1), self.j_pos[ids], self.j_pos[ids] * 0.0 + torch.tensor([0.0, 0.0, -50.0], device=self.device))
        self.ball_marker.visualize(translations=p)

    def plate_front_w(self) -> torch.Tensor:
        """(n, 3) world point of the judged plate plane (plate frame x = plate_distance, y = zone_center_y) at field level:
        the plane the pitcher's tiers and j_cross are measured on (batter session pivot for its frame)."""
        th = -math.radians(self.cfg.plate_yaw_deg)
        pd, zc = self.cfg.plate_distance, self.cfg.zone_center_y
        p = torch.tensor([math.cos(th) * pd - math.sin(th) * zc, math.sin(th) * pd + math.cos(th) * zc, self.field_z],
                         device=self.device)
        return self.scene.env_origins + p

    def _rk4(self, p, v, w, dt):
        c = self._aero_c
        a1, w1 = _deriv(v, w, self.phys, c)
        a2, w2 = _deriv(v + 0.5 * dt * a1, w + 0.5 * dt * w1, self.phys, c)
        a3, w3 = _deriv(v + 0.5 * dt * a2, w + 0.5 * dt * w2, self.phys, c)
        a4, w4 = _deriv(v + dt * a3, w + dt * w3, self.phys, c)
        new_p = p + dt / 6 * (v + 2 * (v + 0.5 * dt * a1) + 2 * (v + 0.5 * dt * a2) + (v + dt * a3))
        new_v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        new_w = w + dt / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        return new_p, new_v, new_w

    # ------------------------------------------------------------------ reset (first load only, see the module doc)
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.batter._ALL_INDICES
        root = self.batter.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.batter.write_root_state_to_sim(root, env_ids)
        jp = self.batter.data.default_joint_pos[env_ids]
        self.batter.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        self.batter.set_joint_position_target(jp, env_ids=env_ids)
        self.batter.set_joint_velocity_target(torch.zeros_like(jp), env_ids=env_ids)
        if self.batter_side is not None:  # the batter's own reset (frame 0 of its clip)
            self.batter_side.reset(env_ids)
        if hasattr(self, "j_fly"):
            for b in (self.j_fly, self.j_done):
                b[env_ids] = False
            self.j_cross[env_ids] = float("nan")
            self.handoff_t[env_ids] = float("nan")
