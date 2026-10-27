"""Tee batting by imitating a retargeted OBP swing (DeepMimic-style tracking + ball-flight task reward).

* Reference state initialisation (RSI): episodes start at random phases of the reference swing.
* Residual control: PD targets = reference joint angles (at the end of the control step) + policy residual.
* Reward = tracking (pose, velocity, key bodies incl. bat, root) + the batting task reward of the base env.
* `speed` plays the reference faster than the athlete (same swing, stronger/faster robot): the curriculum
  that pushes bat speed from the human ~76 mph toward the ~96 mph needed for 150.3 m.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse

from aibaseball.mocap.motion import KEY_BODIES, MotionLib

from .mimic_env_cfg import TeeBattingMimicEnvCfg
from .tee_batting_env import TeeBattingEnv


class TeeBattingMimicEnv(TeeBattingEnv):
    cfg: TeeBattingMimicEnvCfg

    def __init__(self, cfg: TeeBattingMimicEnvCfg, render_mode: str | None = None, **kwargs):
        self.ref = None
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        # a library of retargeted swings; each environment uses one clip per episode
        self.ref = MotionLib(list(cfg.motion_files) or [cfg.motion_file], dev, settle_tail_s=cfg.settle_tail_s,
                             settle_tau=cfg.settle_tau)
        self.clip = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # reference joint order -> policy (body joint) order
        name_to_ref = {n: i for i, n in enumerate(self.ref.joint_names)}
        self.ref_cols = torch.tensor([name_to_ref[n] for n in self.body_joint_names], device=dev)
        self.key_body_ids = [self.robot.find_bodies(b)[0][0] for b in KEY_BODIES]
        self.t0 = torch.zeros(self.num_envs, device=dev)
        self.joint_vel_targets = torch.zeros_like(self.joint_targets)
        self.speed = float(cfg.speed)
        self.track_stats = {k: torch.zeros((), device=dev) for k in
                            ("track_reward", "key_err_m", "pose_err_rad", "t0", "min_ball_dist_m",
                             "ss_speed_win", "ss_ref_speed_win")}
        self.min_ball_dist = torch.full((self.num_envs,), 9.0, device=dev)
        self.ss_ref_contact_speed = None
        if cfg.post_clip_balance:
            self.track_stats.update({k: torch.zeros((), device=dev) for k in ("post_bal_score", "post_bal_reward")})
            self.hit_reward_ep = torch.zeros(self.num_envs, device=dev)  # this episode's hit reward (one-shot in rew)
            self.foot_ids = [self.robot.find_bodies(f"{s}_foot")[0][0] for s in "lr"]
            self.foot_zmin = torch.full((self.num_envs, 2), 9.0, device=dev)
            self.body_mass = self.robot.root_physx_view.get_masses().to(dev)

    # ------------------------------------------------------------------ reference time
    def _ref_time(self, extra_steps: int = 0) -> torch.Tensor:
        return self.t0 + self.speed * (self.episode_length_buf.float() + extra_steps) * self.step_dt

    def _ref(self, t: torch.Tensor) -> dict:
        r = self.ref.sample(t, self.clip)
        r["joint_pos"] = r["joint_pos"][:, self.ref_cols]
        r["joint_vel"] = r["joint_vel"][:, self.ref_cols] * self.speed
        r["root_lin_vel"] = r["root_lin_vel"] * self.speed
        r["root_ang_vel"] = r["root_ang_vel"] * self.speed
        origin = self.scene.env_origins
        r["root_pos"] = r["root_pos"] + origin
        r["key_pos"] = r["key_pos"] + origin.unsqueeze(1)
        return r

    # ------------------------------------------------------------------ actions (residual on the reference)
    def _pre_physics_step(self, actions: torch.Tensor):
        # playback-speed curriculum: same swing, executed progressively faster than the athlete
        c = self.cfg
        if c.speed_ramp_steps > 0:
            prog = min(1.0, self.common_step_counter / c.speed_ramp_steps)
            self.speed = c.speed + (c.speed_end - c.speed) * prog
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        nxt = self._ref(self._ref_time(1))
        tgt = nxt["joint_pos"] + self.actions * self.action_scale * self.cfg.residual_scale
        self.joint_targets[:, self.body_joint_ids] = torch.clamp(tgt, self.q_lo, self.q_hi)
        # velocity feed-forward: the PD damping tracks the reference joint velocity instead of
        # braking toward zero (removes most of the tracking lag of a 30+ m/s swing)
        ff = nxt["joint_vel"] * self.cfg.vel_feedforward
        if self.cfg.zero_ff_after_ref_end:  # the held final pose has no velocity (the sampler repeats the last frame's)
            ff = torch.where((self._ref_time(1) >= self.ref.duration[self.clip]).unsqueeze(-1), torch.zeros_like(ff), ff)
        self.joint_vel_targets[:, self.body_joint_ids] = ff
        self.new_hit[:] = False
        self.hit_reward[:] = 0.0

    def _apply_action(self):
        super()._apply_action()
        self.robot.set_joint_velocity_target(self.joint_vel_targets)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        if self.ref is None:
            return {"policy": obs}
        d = self.robot.data
        t = self._ref_time()
        nxt = self._ref(self._ref_time(1))
        q = d.joint_pos[:, self.body_joint_ids]
        knob, axis, *_ = self._bat_state()
        tip = knob + axis * self.bat_geom.length
        extra = torch.cat([
            (t / self.ref.duration[self.clip]).clamp(max=1.0 if self.cfg.clamp_ref_phase else float("inf")).unsqueeze(-1),
            nxt["joint_pos"] - q,
            quat_rotate_inverse(d.root_link_quat_w, nxt["root_pos"] - d.root_link_pos_w),
            quat_rotate_inverse(d.root_link_quat_w, nxt["key_pos"][:, -1] - tip),
        ], -1)
        return {"policy": torch.nan_to_num(torch.cat([obs, extra], -1))}

    # ------------------------------------------------------------------ rewards
    def _tracking(self):
        d = self.robot.data
        r = self._ref(self._ref_time())
        q = d.joint_pos[:, self.body_joint_ids]
        qd = d.joint_vel[:, self.body_joint_ids]
        pose_err = ((q - r["joint_pos"]) ** 2).mean(-1)
        vel_err = ((qd - r["joint_vel"]) ** 2).mean(-1)
        knob, axis, *_ = self._bat_state()
        key = torch.cat([d.body_link_pos_w[:, self.key_body_ids], (knob + axis * self.bat_geom.length).unsqueeze(1)], 1)
        key_err = ((key - r["key_pos"]) ** 2).sum(-1).mean(-1)
        root_err = ((d.root_link_pos_w - r["root_pos"]) ** 2).sum(-1)
        rot_err = quat_error_magnitude(d.root_link_quat_w, r["root_quat"]) ** 2
        c = self.cfg
        rew = (c.w_pose * torch.exp(-pose_err / c.sigma_pose ** 2)
               + c.w_vel * torch.exp(-vel_err / c.sigma_vel ** 2)
               + c.w_key * torch.exp(-key_err / c.sigma_key ** 2)
               + c.w_root * torch.exp(-root_err / c.sigma_root ** 2 - rot_err / c.sigma_rot ** 2))
        return rew, key_err.sqrt(), pose_err.sqrt()

    def _get_rewards(self) -> torch.Tensor:
        rew = super()._get_rewards()
        track, key_err, pose_err = self._tracking()
        # contact guidance: around the reference contact time, pull the sweet spot onto the ball
        knob, axis, cm, v_cm, w, _ = self._bat_state()
        ss_p, _ = self._sweet_spot(knob, axis, cm, v_cm, w)
        dist = (ss_p - self._true_ball_pos()).norm(dim=-1)
        dist = torch.where(self.hit, torch.zeros_like(dist), dist)
        self.min_ball_dist = torch.minimum(self.min_ball_dist, dist)
        window = torch.exp(-(((self._ref_time() - self.ref.contact_time[self.clip]) / self.cfg.contact_window_s) ** 2))
        contact_r = window * torch.exp(-((dist / self.cfg.sigma_contact) ** 2)) * (~self.hit | self.new_hit).float()
        # bat speed before contact: the sweet-spot velocity along the reference's, capped at the reference speed
        ss_r = torch.zeros_like(dist)
        if self.cfg.w_ss_speed > 0:
            t = self._ref_time()
            win = ((t >= self.ref.contact_time[self.clip] - self.cfg.ss_speed_window_s) & (t < self.ref.contact_time[self.clip])
                   & (~self.hit | self.new_hit))
            if win.any():
                h = 0.005
                ss_ref = [self._ref_sweet_spot(t + s * h) for s in (-1, 1)]
                v_ref = (ss_ref[1] - ss_ref[0]) / (2 * h) * self.speed
                v_ref_n = v_ref.norm(dim=-1).clamp_min(1.0)
                _, v_ss = self._sweet_spot(knob, axis, cm, v_cm, w)
                along = (v_ss * v_ref).sum(-1) / v_ref_n
                if self.ss_ref_contact_speed is None:  # per clip, the reference sweet-spot speed at contact (~31 m/s)
                    tc, cl = self.ref.contact_time, torch.arange(self.ref.num_clips, device=self.device)
                    r = [self.ref.sample(tc + s * h, cl)["key_pos"] for s in (-1, 1)]
                    self.ss_ref_contact_speed = ((self._sweet_spot_of(r[1]) - self._sweet_spot_of(r[0])) / (2 * h)).norm(dim=-1)
                v_c = self.ss_ref_contact_speed[self.clip] * self.speed
                ss_r = torch.where(win, along.clamp(0.0).minimum(v_ref_n) / v_c, ss_r)
                self.track_stats["ss_speed_win"] = 0.99 * self.track_stats["ss_speed_win"] + 0.01 * v_ss.norm(dim=-1)[win].mean()
                self.track_stats["ss_ref_speed_win"] = 0.99 * self.track_stats["ss_ref_speed_win"] + 0.01 * v_ref_n[win].mean()
        done = self.reset_terminated | self.reset_time_outs
        a = 0.01
        if done.any():
            self.track_stats["min_ball_dist_m"] = (1 - 0.02) * self.track_stats["min_ball_dist_m"] + 0.02 *                 self.min_ball_dist[done & (self.t0 < self.ref.contact_time[self.clip])].mean().nan_to_num(0.0)
        for k, v in (("track_reward", track.mean()), ("key_err_m", key_err.mean()), ("pose_err_rad", pose_err.mean()),
                     ("t0", self.t0.mean())):
            self.track_stats[k] = (1 - a) * self.track_stats[k] + a * v
        self.extras["log"].update({f"mimic/{k}": v.clone() for k, v in self.track_stats.items()})
        self.extras["log"]["mimic/speed"] = torch.tensor(self.speed, device=self.device)
        if self.cfg.post_clip_balance:
            rew, track = self._post_clip_balance(rew, track)
        return rew + self.cfg.w_track * track + self.cfg.w_contact * contact_r + self.cfg.w_ss_speed * ss_r

    def _post_clip_balance(self, rew, track):
        """Past the clip end: no mimic target, a standing reward capped by the episode's hit reward (cfg comment)."""
        c, d = self.cfg, self.robot.data
        self.hit_reward_ep = torch.where(self.new_hit, self.hit_reward.clamp_min(0.0), self.hit_reward_ep)
        post_s = (self._ref_time() - self.ref.duration[self.clip]) / self.speed
        past = post_s >= 0
        track = torch.where(past, torch.zeros_like(track), track)
        # the base reward charged a fall penalty: past the clip a fall only forfeits the balance reward (misses too)
        fall_pen = torch.where(self.hit, c.w_fall_after_hit, c.w_fall) * self.fallen.float()
        rew = rew + torch.where(past, fall_pen, torch.zeros_like(rew))
        tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        w = self.body_mass.unsqueeze(-1)
        com = ((d.body_com_pos_w * w).sum(1) / w.sum(1))[:, :2]
        feet = d.body_link_pos_w[:, self.foot_ids]
        a, b = feet[:, 0, :2], feet[:, 1, :2]
        ab = b - a
        s = (((com - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
        com_off = (com - (a + s.unsqueeze(-1) * ab)).norm(dim=-1)
        self.foot_zmin = torch.minimum(self.foot_zmin, feet[..., 2])
        lift = (feet[..., 2] - self.foot_zmin).clamp_min(0.0)
        score = (torch.exp(-(tilt / c.bal_sigma_tilt) ** 2)
                 + torch.exp(-(com_off / c.bal_sigma_com) ** 2)
                 + torch.exp(-(d.root_lin_vel_w.norm(dim=-1) / c.bal_sigma_lin) ** 2
                             - (d.root_ang_vel_w.norm(dim=-1) / c.bal_sigma_ang) ** 2)
                 + torch.exp(-(lift / c.bal_sigma_foot) ** 2).mean(-1)) / 4
        on = past & (post_s < c.bal_window_s) & self.hit
        bal = torch.where(on, c.bal_frac * self.hit_reward_ep * self.step_dt / c.bal_window_s * score, torch.zeros_like(rew))
        if on.any():
            a_ = 0.01
            self.track_stats["post_bal_score"] = (1 - a_) * self.track_stats["post_bal_score"] + a_ * score[on].mean()
            self.track_stats["post_bal_reward"] = (1 - a_) * self.track_stats["post_bal_reward"] + a_ * bal[on].mean()
            self.extras["log"].update({f"mimic/{k}": self.track_stats[k].clone() for k in ("post_bal_score", "post_bal_reward")})
        return rew + bal, track

    def _ref_sweet_spot(self, t: torch.Tensor) -> torch.Tensor:
        return self._sweet_spot_of(self._ref(t)["key_pos"])

    def _sweet_spot_of(self, key: torch.Tensor) -> torch.Tensor:
        """Reference sweet spot: on the bottom-hand -> bat-tip line, sweet_spot_x from the knob end."""
        tip, hand = key[:, -1], key[:, KEY_BODIES.index("l_hand")]
        u = tip - hand
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tip - u * (self.bat_geom.length - self.bat_geom.sweet_spot_x)

    # ------------------------------------------------------------------ dones
    def _get_dones(self):
        terminated, time_out = super()._get_dones()
        if self.ref is None:
            return terminated, time_out
        d = self.robot.data
        knob, axis, *_ = self._bat_state()
        r = self._ref(self._ref_time())
        key = torch.cat([d.body_link_pos_w[:, self.key_body_ids], (knob + axis * self.bat_geom.length).unsqueeze(1)], 1)
        key_err = ((key - r["key_pos"]) ** 2).sum(-1).mean(-1).sqrt()
        if self.cfg.play_mode:  # let the ball fly; the reference holds its final (follow-through) pose
            return terminated, time_out
        lost = key_err > self.cfg.max_key_err
        if self.cfg.post_clip_balance:  # no reference to stay on past the clip end
            lost = lost & (self._ref_time() < self.ref.duration[self.clip])
        self.lost = lost
        if self.cfg.lost_after_hit_is_fall:  # drifting off the follow-through is how a fall starts: penalise it as one
            self.fallen = self.fallen | (lost & self.hit)
        ref_end = self._ref_time(1) >= self.ref.duration[self.clip]
        if self.cfg.ref_hold_end:  # like play: the reference holds its last pose and the episode goes on
            return terminated | lost, time_out
        return terminated | lost, time_out | ref_end

    # ------------------------------------------------------------------ reset (reference state initialisation)
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self.ref is None:
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        n = len(env_ids)
        dev = self.device
        self.clip[env_ids] = torch.randint(0, self.ref.num_clips, (n,), device=dev)
        rsi = torch.rand(n, device=dev) < self.cfg.rsi_prob
        t_max = (self.ref.contact_time[self.clip[env_ids]] - self.cfg.rsi_margin_before_contact).clamp_min(0.0)
        self.min_ball_dist[env_ids] = 9.0
        if self.cfg.post_clip_balance:
            self.hit_reward_ep[env_ids] = 0.0
            self.foot_zmin[env_ids] = 9.0
        self.t0[env_ids] = torch.where(rsi, torch.rand(n, device=dev) * t_max, torch.zeros(n, device=dev))
        self.episode_length_buf[env_ids] = 0
        r = self.ref.sample(self.t0[env_ids], self.clip[env_ids])
        origins = self.scene.env_origins[env_ids]
        root = torch.cat([r["root_pos"] + origins, r["root_quat"], r["root_lin_vel"] * self.speed,
                          r["root_ang_vel"] * self.speed], -1)
        self.robot.write_root_state_to_sim(root, env_ids)
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(jp)
        jp[:, self.body_joint_ids] = r["joint_pos"][:, self.ref_cols]
        jv[:, self.body_joint_ids] = r["joint_vel"][:, self.ref_cols] * self.speed
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self.joint_targets[env_ids] = jp
        self.joint_vel_targets[env_ids] = jv
        # ball on the tee where the reference sweet spot meets it
        tee = origins + self.ref.tee_pos[self.clip[env_ids]]
        tee[:, 2] += self.cfg.tee_offset_z  # > 0: ball above the reference sweet-spot path (undercut -> lift + backspin)
        # tee-position randomisation (curriculum: grows from 0 to the configured range)
        c = self.cfg
        grow = min(1.0, self.common_step_counter / c.tee_noise_ramp_steps) if c.tee_noise_ramp_steps > 0 else 1.0
        tee[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * c.tee_xy_noise * grow
        tee[:, 2] += (torch.rand(n, device=dev) * 2 - 1) * c.tee_height_noise * grow
        self.tee_pos[env_ids] = tee
        self.ball_pos[env_ids] = tee
        unit_q = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
        self.ball.write_root_pose_to_sim(torch.cat([self._park(tee), unit_q], -1), env_ids)
        tee_body = tee.clone()
        tee_body[:, 2] = tee[:, 2] - self.phys.ball.radius - 0.5
        self.tee.write_root_pose_to_sim(torch.cat([tee_body, unit_q], -1), env_ids)

