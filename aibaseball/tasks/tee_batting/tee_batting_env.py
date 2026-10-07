"""Tee batting with the AIB-1 humanoid.

Physics pipeline per 400 Hz physics step:
  1. PhysX steps the robot (articulation + two-hand loop grip, feet friction on the ground).
  2. Swept ball-bat contact test against the bat motion over that step (no tunnelling).
  3. On contact, a 3D rigid-body impulse (speed-dependent COR, sweet-spot vibration loss,
     Coulomb friction) gives the ball its exit velocity and spin.
  4. Training: the carry distance is computed immediately with the drag + Magnus flight model
     and the episode ends.  Play mode: the ball flies in the scene with the same model and the
     reaction impulse is applied to the bat.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from aibaseball.physics import BatGeometry, BatSpec, PhysicsSpec, detect_contact, resolve_impact
from aibaseball.physics.ball_flight import GraphedFlight, _aero_consts, _deriv
from aibaseball.robot.aib1_cfg import BODY_JOINT_EXPR, GROUP_JOINTS

from .tee_batting_env_cfg import TeeBattingEnvCfg

X_AXIS = (1.0, 0.0, 0.0)


class TeeBattingEnv(DirectRLEnv):
    cfg: TeeBattingEnvCfg

    def __init__(self, cfg: TeeBattingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        n = self.num_envs
        self.phys = PhysicsSpec()
        self.bat_spec = BatSpec()
        self.bat_geom = BatGeometry(self.bat_spec, dev)

        # joints controlled by the policy, in a fixed order
        self.body_joint_ids, self.body_joint_names = self.robot.find_joints(BODY_JOINT_EXPR, preserve_order=True)
        assert len(self.body_joint_ids) == self.cfg.action_space - self.cfg.extra_actions, self.body_joint_names
        self.body_joint_ids = torch.tensor(self.body_joint_ids, device=dev)
        scale = torch.zeros(len(self.body_joint_names), device=dev)
        for group, s in self.cfg.action_scale.items():
            for i, name in enumerate(self.body_joint_names):
                if any(re.fullmatch(e, name) for e in GROUP_JOINTS[group]):
                    scale[i] = s
        assert (scale > 0).all(), "every controlled joint needs an action scale"
        self.action_scale = scale
        limits = self.robot.data.soft_joint_pos_limits[0, self.body_joint_ids]
        self.q_lo, self.q_hi = limits[:, 0], limits[:, 1]

        self.bat_id = self.robot.find_bodies("bat")[0][0]
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]

        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.joint_targets = self.robot.data.default_joint_pos.clone()

        # task state
        self.tee_pos = torch.zeros(n, 3, device=dev)  # world
        # perceived ball position (step B): per-episode bias + per-step noise, delayed by ball_obs_delay_s
        self.obs_delay_steps = int(round(cfg.ball_obs_delay_s / (cfg.sim.dt * cfg.decimation)))
        self.ball_obs_hist = torch.zeros(n, self.obs_delay_steps + 1, 3, device=dev)
        self.ball_obs_bias = torch.zeros(n, 3, device=dev)
        self.ball_obs_fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hit_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.new_hit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_knob = torch.zeros(n, 3, device=dev)
        self.prev_axis = torch.zeros(n, 3, device=dev)
        self.prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.launch_vel = torch.zeros(n, 3, device=dev)
        self.launch_omega = torch.zeros(n, 3, device=dev)
        self.launch_pos = torch.zeros(n, 3, device=dev)
        self.contact_x = torch.zeros(n, device=dev)
        self.hit_reward = torch.zeros(n, device=dev)
        self.carry = torch.zeros(n, device=dev)
        self.landed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dist_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ball_pos = torch.zeros(n, 3, device=dev)  # play mode: in-flight ball state
        self.ball_vel = torch.zeros(n, 3, device=dev)
        self.ball_omega = torch.zeros(n, 3, device=dev)
        self.pending_impulse = torch.zeros(n, 3, device=dev)
        self.pending_point = torch.zeros(n, 3, device=dev)
        self.has_pending = False

        self.flight = GraphedFlight(n, self.phys, dev, dt=self.cfg.flight_dt, block=10)
        self.aero_c = _aero_consts(self.phys, self.tee_pos)

        self.field_dir = torch.tensor([1.0, 0.0, 0.4], device=dev)
        self.field_dir = self.field_dir / self.field_dir.norm()
        # ideal launch direction: center field, 30 deg up (smooth shaping signal for any launch angle)
        self.ideal_dir = torch.tensor([math.cos(math.radians(30)), 0.0, math.sin(math.radians(30))], device=dev)

        # running statistics for logging (always the same keys for rsl_rl)
        self.stats = {k: torch.zeros((), device=dev) for k in
                      ("hit_rate", "fall_rate", "carry", "carry_max", "exit_velo", "launch_angle", "backspin_rpm",
                       "bat_speed", "success_rate", "fair_rate")}
        self.best_carry = 0.0
        self.play_log: list[dict] = []

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        self.tee = RigidObject(self.cfg.tee)
        ground = GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9))
        if self.cfg.play_mode:  # big turf-colored field so a 150 m drive stays on the ground plane
            ground.size = (500.0, 500.0)
            ground.color = (0.16, 0.36, 0.14)
        spawn_ground_plane("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.rigid_objects["tee"] = self.tee
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)
        if self.cfg.play_mode:
            self.landing_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/landing",
                markers={"spot": sim_utils.SphereCfg(radius=1.0, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.1)))},
            ))
            self.landing_markers.set_visibility(False)  # shown once a ball has landed
            # The ball shown in play mode is a purely visual marker. Teleporting the (kinematic) physics ball
            # every physics step broke the robot's ground contact after resets (it sank through the floor),
            # so in play mode the physics ball is parked underground and never moved during an episode.
            self.ball_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/ball", markers={"ball": sim_utils.SphereCfg(
                    radius=self.cfg.ball.spawn.radius, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.95, 0.92)))}))
            self._spawn_distance_arcs()

    def _spawn_distance_arcs(self):
        """Visual distance arcs (fair territory, +-45 deg) at 50 / 100 m (white) and the 150.3 m target (yellow)."""
        arcs = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/distance_arcs",
            markers={
                "mark": sim_utils.CylinderCfg(radius=0.5, height=0.04, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.95, 0.95))),
                "target": sim_utils.CylinderCfg(radius=0.8, height=0.05, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0), emissive_color=(0.6, 0.5, 0.0))),
            },
        ))
        pts, idx = [], []
        origin = self.scene.env_origins[0].cpu()
        for radius, kind, n in ((50.0, 0, 31), (100.0, 0, 61), (self.cfg.target_distance, 1, 121)):
            for k in range(n):
                th = math.radians(-45 + 90 * k / (n - 1))
                pts.append([origin[0] + radius * math.cos(th), origin[1] + radius * math.sin(th), 0.02])
                idx.append(kind)
        self.distance_arcs = arcs
        arcs.visualize(translations=torch.tensor(pts, device=self.device), marker_indices=torch.tensor(idx, device=self.device))

    # ------------------------------------------------------------------ bat kinematics helpers
    def _bat_state(self):
        d = self.robot.data
        knob = d.body_link_pos_w[:, self.bat_id]
        quat = d.body_link_quat_w[:, self.bat_id]
        axis = quat_apply(quat, torch.tensor(X_AXIS, device=self.device).expand(self.num_envs, 3))
        cm = d.body_com_pos_w[:, self.bat_id]
        v_cm = d.body_com_lin_vel_w[:, self.bat_id]
        w = d.body_com_ang_vel_w[:, self.bat_id]
        return knob, axis, cm, v_cm, w, quat

    def _sweet_spot(self, knob, axis, cm, v_cm, w):
        p = knob + axis * self.bat_geom.sweet_spot_x
        v = v_cm + torch.cross(w, p - cm, dim=-1)
        return p, v

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        default = self.robot.data.default_joint_pos[:, self.body_joint_ids]
        tgt = default + self.actions * self.action_scale
        self.joint_targets[:, self.body_joint_ids] = torch.clamp(tgt, self.q_lo, self.q_hi)
        self.new_hit[:] = False
        self.hit_reward[:] = 0.0

    def _apply_action(self):
        self._apply_pending_impulse()
        self._check_impact()
        if self.cfg.play_mode:
            self._advance_ball(self.physics_dt)
        self.robot.set_joint_position_target(self.joint_targets)

    # ------------------------------------------------------------------ impact
    def _incoming_ball(self):
        """(position now, position at the previous physics step or None, velocity, spin) of the un-hit ball."""
        return self.tee_pos, None, torch.zeros_like(self.tee_pos), torch.zeros_like(self.tee_pos)

    def _check_impact(self):
        knob, axis, cm, v_cm, w, quat = self._bat_state()
        active = self.prev_valid & ~self.hit
        b_now, b_prev, b_vel, b_omega = self._incoming_ball()
        c = detect_contact(self.bat_geom, b_now, self.phys.ball.radius,
                           self.prev_knob, self.prev_axis, knob, axis, n_sub=self.cfg.contact_substeps, ball_prev=b_prev)
        c.hit &= active
        if c.hit.any():  # single host sync per physics step
            h = c.hit.nonzero(as_tuple=False).squeeze(-1)
            sub = type(c)(*(t[c.hit] for t in (c.hit, c.frac, c.x_on_bat, c.normal, c.point)))
            ball_c = sub.point + self.phys.ball.radius * sub.normal  # ball centre at first contact
            res = resolve_impact(self.bat_geom, self.phys, sub, ball_c, b_vel[h], b_omega[h],
                                 cm[h], axis[h], v_cm[h], w[h])
            self.hit[h] = True
            self.new_hit[h] = True
            self.hit_step[h] = self.episode_length_buf[h]
            self.launch_vel[h] = res.ball_vel
            self.launch_omega[h] = res.ball_omega
            self.launch_pos[h] = ball_c
            self.contact_x[h] = sub.x_on_bat
            if self.cfg.play_mode:
                self.pending_impulse[h] = -res.impulse_on_ball
                self.pending_point[h] = sub.point
                self.has_pending = True
                self.ball_pos[h] = ball_c
                self.ball_vel[h] = res.ball_vel
                self.ball_omega[h] = res.ball_omega
        self.prev_knob[:] = knob
        self.prev_axis[:] = axis
        self.prev_valid[:] = True

    def _apply_pending_impulse(self):
        """Play mode: bat reaction impulse spread over one physics step (forces are in the body frame)."""
        if not self.has_pending:
            return
        knob, axis, cm, _, _, quat = self._bat_state()
        f_w = self.pending_impulse / self.physics_dt
        t_w = torch.cross(self.pending_point - cm, f_w, dim=-1)
        f_b = quat_rotate_inverse(quat, f_w).unsqueeze(1)
        t_b = quat_rotate_inverse(quat, t_w).unsqueeze(1)
        self.robot.set_external_force_and_torque(f_b, t_b, body_ids=[self.bat_id])
        self.pending_impulse.zero_()
        self.has_pending = bool(f_w.abs().sum() > 0)  # one more call clears the wrench

    def _advance_ball(self, dt: float):
        """Play mode: integrate in-flight balls (RK4, drag + Magnus) and update the visual ball marker."""
        self.ball_marker.visualize(translations=torch.where(self.hit.unsqueeze(-1), self.ball_pos, self._true_ball_pos()))
        flying = self.hit & ~self.landed
        if flying.any():
            c = self.aero_c
            v, w = self.ball_vel, self.ball_omega
            a1, w1 = _deriv(v, w, self.phys, c)
            a2, w2 = _deriv(v + 0.5 * dt * a1, w + 0.5 * dt * w1, self.phys, c)
            a3, w3 = _deriv(v + 0.5 * dt * a2, w + 0.5 * dt * w2, self.phys, c)
            a4, w4 = _deriv(v + dt * a3, w + dt * w3, self.phys, c)
            new_p = self.ball_pos + dt / 6 * (v + 2 * (v + 0.5 * dt * a1) + 2 * (v + 0.5 * dt * a2) + (v + dt * a3))
            new_v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            new_w = w + dt / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
            land = flying & (new_p[:, 2] <= self.phys.ball.radius)
            m = flying.unsqueeze(-1)
            self.ball_pos = torch.where(m, new_p, self.ball_pos)
            self.ball_vel = torch.where(m, new_v, self.ball_vel)
            self.ball_omega = torch.where(m, new_w, self.ball_omega)
            if land.any():
                for i in land.nonzero(as_tuple=False).squeeze(-1).tolist():
                    d = (self.ball_pos[i, :2] - self.launch_pos[i, :2]).norm().item()
                    self.carry[i] = d
                    self._log_play_result(i, d)
                self.landed |= land
                self._update_landing_markers()

    def _park(self, pos: torch.Tensor) -> torch.Tensor:
        """Play mode keeps the physics ball underground (the visible ball is a marker)."""
        if not self.cfg.play_mode:
            return pos
        out = pos.clone()
        out[:, 2] = -5.0
        return out

    def _log_play_result(self, i: int, carry: float):
        v = self.launch_vel[i]
        ev = v.norm().item()
        la = math.degrees(math.atan2(v[2].item(), v[:2].norm().item()))
        d = self.ball_pos[i, :2] - self.launch_pos[i, :2]
        spray = math.degrees(math.atan2(d[1].item(), d[0].item()))
        vh = torch.nn.functional.normalize(torch.stack([v[0], v[1], torch.zeros_like(v[0])]), dim=0)
        back = float((self.launch_omega[i] * torch.cross(vh, torch.tensor([0.0, 0.0, 1.0], device=v.device), dim=0)).sum())
        rec = dict(env=i, carry_m=carry, backspin_rpm=back * 60 / (2 * math.pi), exit_velo_mps=ev, exit_velo_mph=ev / 0.44704, launch_deg=la, spray_deg=spray,
                   spin_rpm=self.launch_omega[i].norm().item() * 60 / (2 * math.pi), contact_x=self.contact_x[i].item())
        self.play_log.append(rec)
        ok = "SUCCESS" if carry >= self.cfg.target_distance and abs(spray) <= self.cfg.fair_half_angle_deg else ""
        print(f"[play] env {i}: carry {carry:6.1f} m | EV {ev:5.1f} m/s ({ev/0.44704:5.1f} mph) | LA {la:5.1f} deg | "
              f"spray {spray:+5.1f} deg | spin {rec['spin_rpm']:5.0f} rpm (backspin {rec['backspin_rpm']:+5.0f}) {ok}", flush=True)

    def _update_landing_markers(self):
        if self.landed.any():
            pos = self.ball_pos[self.landed].clone()
            pos[:, 2] = 0.02
            self.landing_markers.set_visibility(True)
            self.landing_markers.visualize(translations=pos, scales=torch.full((len(pos), 3), 0.3, device=self.device))

    # ------------------------------------------------------------------ observations
    def _true_ball_pos(self) -> torch.Tensor:
        """Where the ball really is (on the tee; a moving ball in later stages)."""
        return self.tee_pos

    def _perceived_ball(self) -> torch.Tensor:
        """Ball position as the batter perceives it: true + episode bias + noise, `obs_delay_steps` old."""
        c = self.cfg
        meas = self._true_ball_pos() + self.ball_obs_bias + torch.randn_like(self.tee_pos) * c.ball_obs_noise_std
        fresh = self.ball_obs_fresh
        if fresh.any():  # new episode: no older percepts yet -> fill the history with the first one
            self.ball_obs_hist[fresh] = meas[fresh].unsqueeze(1)
            self.ball_obs_fresh[:] = False
        self.ball_obs_hist = torch.roll(self.ball_obs_hist, 1, dims=1)
        self.ball_obs_hist[:, 0] = meas
        return self.ball_obs_hist[:, -1]

    def _get_observations(self) -> dict:
        d = self.robot.data
        root_q = d.root_link_quat_w
        root_p = d.root_link_pos_w
        knob, axis, cm, v_cm, w, _ = self._bat_state()
        ss_p, ss_v = self._sweet_spot(knob, axis, cm, v_cm, w)
        jp = d.joint_pos[:, self.body_joint_ids] - d.default_joint_pos[:, self.body_joint_ids]
        jv = d.joint_vel[:, self.body_joint_ids]
        phase = (self.episode_length_buf.float() * self.step_dt / self.max_episode_length_s).unsqueeze(-1)
        ball_seen = self._perceived_ball()
        obs = torch.cat([
            jp,
            jv * 0.05,
            d.projected_gravity_b,
            d.root_lin_vel_b * 0.5,
            d.root_ang_vel_b * 0.25,
            (root_p[:, 2:3] - 0.94),
            quat_rotate_inverse(root_q, ball_seen - root_p).clamp(-self.cfg.ball_obs_clip, self.cfg.ball_obs_clip),
            quat_rotate_inverse(root_q, ss_p - ball_seen).clamp(-self.cfg.ball_obs_clip, self.cfg.ball_obs_clip),
            quat_rotate_inverse(root_q, ss_v) * 0.05,
            quat_rotate_inverse(root_q, axis),
            phase,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ rewards
    def _launch_score(self, v, ev, la):
        """0..1: smooth direction score (any angle), or a sharper Gaussian around the target launch angle."""
        if self.cfg.launch_sigma_deg > 0:
            return torch.exp(-(((la - self.cfg.launch_target_deg) / self.cfg.launch_sigma_deg) ** 2))
        return ((1.0 + (v / ev.clamp_min(1e-6).unsqueeze(-1) * self.ideal_dir).sum(-1)) / 2) ** 2

    def _hit_rewards(self):
        """Compute launch metrics + carry for balls hit during this step (training mode)."""
        ids = self.new_hit.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        v, w, p = self.launch_vel[ids], self.launch_omega[ids], self.launch_pos[ids]
        res = self.flight.run(p, v, w)
        carry = res.carry
        disp = res.landing_pos[:, :2] - p[:, :2]
        spray = torch.rad2deg(torch.atan2(disp[:, 1], disp[:, 0]))
        fair = (spray.abs() <= self.cfg.fair_half_angle_deg) & (disp[:, 0] > 0)
        ev = v.norm(dim=-1)
        la = torch.rad2deg(torch.atan2(v[:, 2], v[:, :2].norm(dim=-1)))
        success = fair & (carry >= self.cfg.target_distance)
        cfg = self.cfg
        # Weak contact must be worth ~nothing: distance is quadratic, launch angle is gated by exit speed.
        ev_n = ev.clamp(0, 70) / 52.5
        # backspin = spin component about the horizontal axis perpendicular to travel (v_h x z): lift
        v_h = torch.nn.functional.normalize(torch.cat([v[:, :2], torch.zeros_like(v[:, :1])], -1), dim=-1)
        back_axis = torch.cross(v_h, torch.tensor([0.0, 0.0, 1.0], device=v.device).expand_as(v_h), dim=-1)
        backspin = (w * back_axis).sum(-1) * 60 / (2 * math.pi)
        r = (cfg.w_distance * (carry.clamp(0, 200) / cfg.target_distance) ** 2 * torch.where(fair, 1.0, cfg.foul_factor)
             + cfg.w_success * success.float()
             + cfg.w_exit_velo * ev_n
             + cfg.w_launch * ev_n * self._launch_score(v, ev, la)
             + cfg.w_backspin * ev_n * (backspin / 2000.0).clamp(-1.0, 1.0))
        self.hit_reward[ids] = r
        self.carry[ids] = carry
        # stats (EMA so keys exist every step)
        a = 0.05
        for k, val in (("carry", carry.mean()), ("exit_velo", ev.mean()), ("launch_angle", la.mean()),
                       ("backspin_rpm", backspin.mean()), ("success_rate", success.float().mean()),
                       ("fair_rate", fair.float().mean())):
            self.stats[k] = (1 - a) * self.stats[k] + a * val
        self.stats["carry_max"] = torch.maximum(self.stats["carry_max"] * 0.999, (carry * fair).max())
        self.best_carry = max(self.best_carry, float((carry * fair).max()))

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        knob, axis, cm, v_cm, w, _ = self._bat_state()
        ss_p, ss_v = self._sweet_spot(knob, axis, cm, v_cm, w)
        dist = (ss_p - self._true_ball_pos()).norm(dim=-1)
        not_hit_before = ~self.hit | self.new_hit
        approach = torch.where(not_hit_before & self.dist_valid, self.prev_dist - dist, torch.zeros_like(dist))
        self.prev_dist[:] = dist
        self.dist_valid[:] = True
        speed_toward = (ss_v * self.field_dir).sum(-1).clamp(0.0, 50.0) / 45.0
        bat_speed_r = speed_toward**2 * torch.exp(-((dist / 0.4) ** 2)) * not_hit_before.float()

        if not cfg.play_mode:
            self._hit_rewards()
        rew = (self.hit_reward
               + cfg.w_approach * approach
               + cfg.w_bat_speed * bat_speed_r
               - cfg.w_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        # falling before contact, or during the follow-through (no "dive at the ball" swings)
        rew = rew - torch.where(self.hit, cfg.w_fall_after_hit, cfg.w_fall) * self.fallen.float()
        timeout_miss = self.reset_time_outs & ~self.hit
        rew = rew - cfg.w_miss * timeout_miss.float()

        # logging
        a = 0.02
        done = self.reset_terminated | self.reset_time_outs
        if done.any():
            self.stats["hit_rate"] = (1 - a) * self.stats["hit_rate"] + a * (self.hit[done].float().mean())
            self.stats["fall_rate"] = (1 - a) * self.stats["fall_rate"] + a * (self.fallen[done].float().mean())
        if self.new_hit.any():
            self.stats["bat_speed"] = 0.95 * self.stats["bat_speed"] + 0.05 * ss_v[self.new_hit].norm(dim=-1).mean()
        self.extras["log"] = {f"bat/{k}": v.clone() for k, v in self.stats.items()}
        return rew

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_impact()  # cover the last physics sub-step of this control step
        d = self.robot.data
        pelvis_z = d.body_link_pos_w[:, self.pelvis_id, 2]
        self.fallen = (pelvis_z < self.cfg.min_pelvis_height) | (d.projected_gravity_b[:, 2] > -self.cfg.max_tilt_cos)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.play_mode:
            terminated = self.landed.clone() if self.cfg.terminate_on_land else torch.zeros_like(self.landed)
        else:
            follow_steps = int(round(self.cfg.follow_through_s / self.step_dt))
            done_follow = self.hit & (self.episode_length_buf - self.hit_step >= follow_steps)
            terminated = self.fallen | done_follow
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        self.ball_obs_bias[env_ids] = torch.randn(n, 3, device=self.device) * self.cfg.ball_obs_bias_std
        self.ball_obs_fresh[env_ids] = True
        dev = self.device
        origins = self.scene.env_origins[env_ids]

        # tee / ball
        tee = origins.clone()
        tee[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * self.cfg.tee_xy_noise
        tee[:, 2] = self.cfg.tee_height + (torch.rand(n, device=dev) * 2 - 1) * self.cfg.tee_height_noise
        self.tee_pos[env_ids] = tee
        unit_q = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
        self.ball.write_root_pose_to_sim(torch.cat([self._park(tee), unit_q], -1), env_ids)
        tee_body = tee.clone()
        tee_body[:, 2] = tee[:, 2] - self.phys.ball.radius - 0.5  # 1 m cylinder: top touches the ball
        self.tee.write_root_pose_to_sim(torch.cat([tee_body, unit_q], -1), env_ids)

        # robot
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0] += origins[:, 0] + self.cfg.batter_offset[0]
        root[:, 1] += origins[:, 1] + self.cfg.batter_offset[1]
        self.robot.write_root_state_to_sim(root, env_ids)
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        noise = (torch.rand(n, len(self.body_joint_ids), device=dev) * 2 - 1) * self.cfg.joint_pos_noise
        jp[:, self.body_joint_ids] += noise
        jv = torch.zeros_like(jp)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.joint_targets[env_ids] = self.robot.data.default_joint_pos[env_ids]

        # buffers
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.hit[env_ids] = False
        self.new_hit[env_ids] = False
        self.landed[env_ids] = False
        self.prev_valid[env_ids] = False
        self.carry[env_ids] = 0.0
        self.dist_valid[env_ids] = False
        self.ball_pos[env_ids] = tee
        self.ball_vel[env_ids] = 0.0
        self.ball_omega[env_ids] = 0.0
