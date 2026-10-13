"""Stage 1 rebuilt as small steps (2026-10-13), each one held until it works before the next is added:

  1a  stand still on the mound in the set position for episode_length_s
  1b  (later) no ball: leg lift, stride, land and stop standing
  1c  (later) 1b + the throwing arm, ball scored by its progress toward the plate

Same robot, PD-residual actions and observations as PitchEnv; a short reward (alive, standing quality, fall,
action rate) and a short termination (fall = pelvis / head low or any body but the feet on the ground, time out).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply

from .pitch_env import PitchEnv


class StageEnv(PitchEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.stage_stats = {k: torch.zeros((), device=self.device)
                            for k in ("fall_rate", "stand_ok", "tilt_deg", "com_speed", "head_z", "both_feet")}

    # 1a: the reference (PD base, observations) stays at its first frame, the set position
    def _ref_time(self, extra: int = 0) -> torch.Tensor:
        if self.cfg.stage == "1a":
            return torch.zeros_like(self.t0)
        return super()._ref_time(extra)

    def _ref(self, t):
        r = super()._ref(t)
        if self.cfg.stage == "1a":  # a still pose: the athlete's first-frame velocities (the leg lift starting) would
            # be a constant PD velocity target - with zero actions it lifted the lead foot 16 cm and the robot fell
            for k in ("joint_vel", "root_lin_vel", "root_ang_vel"):
                r[k] = torch.zeros_like(r[k])
        return r

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clone()
        actions[:, 29:] = 0.0  # fingers keep the grip (no release in 1a)
        super()._pre_physics_step(actions)

    def _stand_state(self):
        c, d = self.cfg, self.robot.data
        forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values
        both = (forces > c.contact_force_n).all(-1)
        com, com_v = self._com()
        up = quat_apply(d.body_link_quat_w[:, self.torso_id],
                        torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
        head_z = self._head_z()
        speed = com_v.norm(dim=-1)
        ok = both & (speed < c.stand_max_com_v) & (tilt < c.stand_max_tilt_deg) & (head_z > c.stand_min_head_z)
        return both, speed, tilt, head_z, ok

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        both, speed, tilt, head_z, ok = self._stand_state()
        steps = max(int(round(c.episode_length_s / self.step_dt)), 1)
        up = ~self.fallen
        upright = torch.exp(-(((tilt - 0.8 * c.stand_max_tilt_deg).clamp_min(0.0) / 15.0) ** 2))
        still = torch.exp(-((speed / (2.0 * c.stand_max_com_v)) ** 2))
        tall = ((head_z - 1.0) / (c.stand_min_head_z + 0.05 - 1.0)).clamp(0.0, 1.0)
        quality = (upright + still + both.float() + tall) / 4.0
        rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        rew = (up.float() * (c.w_stage_alive + c.w_stage_quality * quality) / steps
               - c.w_stage_fall * self.fallen.float() - c.w_stage_rate * rate)
        done = self.reset_terminated | self.reset_time_outs
        a = 0.02
        for k, v, m in (("fall_rate", self.fallen.float(), done), ("stand_ok", ok.float(), self.reset_time_outs),
                        ("tilt_deg", tilt, up), ("com_speed", speed, up), ("head_z", head_z, up),
                        ("both_feet", both.float(), up)):
            n = m.float().sum()
            mean = (v * m.float()).sum() / n.clamp_min(1.0)
            self.stage_stats[k] = torch.where(n > 0, (1 - a) * self.stage_stats[k] + a * mean, self.stage_stats[k])
        self.extras["log"] = {f"stage/{k}": v.clone() for k, v in self.stage_stats.items()}
        return rew

    def _get_dones(self):
        c, d = self.cfg, self.robot.data
        self.fallen = (d.body_link_pos_w[:, self.pelvis_id, 2] < c.fallen_pelvis_z) | (self._head_z() < c.fallen_head_z)
        if c.fallen_touch_z > 0:
            p = d.body_link_pos_w[:, self.touch_ids]
            x = p[..., 0] - self.scene.env_origins[:, 0:1]
            ground = torch.where(x < c.mound_slope_start_x, torch.full_like(x, c.mound_top_z),
                                 (c.mound_top_z - (x - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
            self.fallen |= ((p[..., 2] - ground) < c.fallen_touch_z).any(-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.fallen.clone(), time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if self.cfg.stage == "1a":  # start at rest (the reset wrote the reference's first-frame velocities)
            jp = self.robot.data.joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
            self.qd_target[env_ids] = 0.0
            self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        if self.cfg.stage == "1a" and self.cfg.push_vel > 0:  # a small random shove at the start (robustness)
            n = len(env_ids)
            v = torch.zeros(n, 6, device=self.device)
            ang = torch.rand(n, device=self.device) * 2 * math.pi
            mag = torch.rand(n, device=self.device) * self.cfg.push_vel
            v[:, 0] = mag * torch.cos(ang)
            v[:, 1] = mag * torch.sin(ang)
            self.robot.write_root_velocity_to_sim(v, env_ids)
