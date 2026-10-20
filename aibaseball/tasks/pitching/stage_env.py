"""Stage 1 rebuilt as small steps (2026-10-13), each one held until it works before the next is added:

  1a  stand still on the mound in the set position for episode_length_s
  1b  no ball: leg lift, stride, land and stop standing (the legs follow the athlete until the robot's own lead
      foot lands >= lead_stride_min_m ahead of the back foot, then hold that stance; arms / trunk stay in the set
      position, the ball stays in the glove and is not scored)
  1b1 (2026-10-14, the pitch split into set / leg lift / stride / rotation / throw) leg lift only: the athlete's
      lift up to its peak, held there on the back leg, back down to the set position, stand still. The 1b policies
      (stride and stop) passed at 1750 by stepping the BACK foot 0.8 m away from the plate (pelvis -0.37 m): the
      balance point on the back leg was never learned on its own
  1c  (later) 1b + the throwing arm, ball scored by its progress toward the plate

Same robot, PD-residual actions and observations as PitchEnv; a short reward (alive, standing quality, fall,
action rate) and a short termination (fall = pelvis / head low or any body but the feet on the ground, time out).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from .pitch_env import PitchEnv


class StageEnv(PitchEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.stage_stats = {k: torch.zeros((), device=self.device)
                            for k in ("fall_rate", "stand_ok", "tilt_deg", "com_speed", "head_z", "both_feet",
                                      "landed", "stride_m", "lift_s", "back_slide_m",
                                      "fall_rate_set", "success_set", "fall_rate_rsi", "success_rsi", "drop_rate_set",
                                      "drop_rate_rsi")}
        n, dev = self.num_envs, self.device
        self.s_lifted = torch.zeros(n, dtype=torch.bool, device=dev)  # 1b: lead foot off the ground since 0.1 s
        self.s_landed = torch.zeros(n, dtype=torch.bool, device=dev)  # 1b: lead foot down after a real stride
        self.s_prev_planted = torch.ones(n, dtype=torch.bool, device=dev)
        self.s_air = torch.zeros(n, device=dev)  # s with both feet off the ground before the stride lands
        self.s_land_t = torch.zeros(n, device=dev)  # reference time at the robot's touchdown
        self.s_land_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.s_hold = torch.zeros(n, len(self.body_ids), device=dev)  # robot's joint angles at the touchdown
        self.s_root_land = torch.zeros(n, 3, device=dev)  # robot's root at the touchdown (world)
        self.s_lift_t = torch.zeros(n, device=dev)  # 1b1: s on the back leg alone with the lead foot lifted
        self.s_slid = torch.zeros(n, dtype=torch.bool, device=dev)  # 1b1: back foot left its spot
        self.s_feet0 = torch.zeros(n, 2, 3, device=dev)  # feet / pelvis at the episode's first step (world)
        self.s_pelvis0 = torch.zeros(n, 3, device=dev)
        self.s_need0 = torch.ones(n, dtype=torch.bool, device=dev)  # start positions still to be read
        # desync: each env's first episode is cut (time-out, bootstrapped) at a random length. The reference clock runs
        # on episode_length_buf, so rsl_rl's random start length cannot be used (df5ed19), and without it all envs
        # start, fall and time out in phase - every PPO batch saw one slice of the pitch (fc: mean episode length
        # waved 184 -> 1100 -> 690; fg / fh from fe 2950 only went down)
        self.s_cut = torch.full((n,), int(self.max_episode_length), device=dev)
        self.s_desync = True  # drawn on the first step (the first reset sets every cut back to the full length)
        self.s_crossed = torch.zeros(n, dtype=torch.bool, device=dev)  # ball crossed the plate (arm + ball step)
        self.s_cross_yz = torch.zeros(n, 2, device=dev)
        self.s_ball_maxx = torch.zeros(n, device=dev)
        self.s_ball_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.s_open = torch.zeros(n, dtype=torch.bool, device=dev)  # fingers opened (release_ahead_m trigger)
        self.s_rel_forced = torch.zeros(n, dtype=torch.bool, device=dev)  # release action: opened at the window end
        self.s_ball_end = torch.zeros(n, dtype=torch.bool, device=dev)
        self._jcol = {self.body_names[i]: i for i in range(len(self.body_ids))}  # body joint name -> column
        self._arm_cols = [i for i, nm in enumerate(self.body_names) if any(k in nm for k in ("shoulder", "elbow", "wrist"))]
        self.assist_steps0 = 0  # assist-force curriculum: env steps already done in earlier training cycles
        self._ls = None  # landed stances (scripts/capture_stage_states.py) for landed_start_prob
        if cfg.landed_start_prob > 0:
            d = torch.load(cfg.landed_states_path, map_location=dev)
            self._ls = (d["root"].to(dev), d["joint_pos"].to(dev))
            print(f"[StageEnv] {len(d['root'])} landed stances from {cfg.landed_states_path}", flush=True)

    # 1a: the reference (PD base, observations) stays at its first frame, the set position
    def _ref_time(self, extra: int = 0) -> torch.Tensor:
        if self.cfg.stage == "1a":
            return torch.zeros_like(self.t0)
        return super()._ref_time(extra)

    def _lift_clock(self, t):
        """1b1: reference time on the athlete's leg lift - up to lift_peak_s, held lift_hold_s, back down (time
        reversed), then the set position (0); and the velocity sign (+1 up, -1 down, 0 held / set)."""
        c = self.cfg
        tp, h = c.lift_peak_s, c.lift_hold_s
        t = t - c.lift_delay_s  # set position held lift_delay_s first (time to put the weight on the back leg)
        up, hold, down = (t >= 0) & (t < tp), (t >= tp) & (t < tp + h), (t >= tp + h) & (t < 2 * tp + h)
        tm = torch.where(up, t, torch.where(hold, torch.full_like(t, tp), torch.where(down, 2 * tp + h - t,
                                                                                         torch.zeros_like(t))))
        sgn = up.float() - down.float()
        return tm, sgn

    def _arm_blend(self, r, t):
        """SYW: the throwing arm's reference from the second motion around the release (arm_blend_times, smoothstep in
        and out), each joint on its own arm_ref_lead_s clock."""
        if not hasattr(self, "_blend_cols"):
            import re as _re
            self._blend_cols = [i for i, nm in enumerate(self.body_names) if _re.fullmatch(self.cfg.arm_blend_joints, nm)]
        t0, t1, t2, t3 = (self.release_ref + x for x in self.cfg.arm_blend_times)
        sm = lambda x: x * x * (3.0 - 2.0 * x)  # noqa: E731
        w = sm(((t - t0) / (t1 - t0)).clamp(0.0, 1.0)) * (1.0 - sm(((t - t2) / (t3 - t2)).clamp(0.0, 1.0)))
        w = w * self.cfg.arm_blend_weight
        main, leads = self.ref, self.cfg.arm_ref_lead_s or {}
        try:
            self.ref = self.ref_arm
            for col in self._blend_cols:
                rb = PitchEnv._ref(self, t + leads.get(self.body_names[col], 0.0))
                r["joint_pos"][:, col] += w * (rb["joint_pos"][:, col] - r["joint_pos"][:, col])
                r["joint_vel"][:, col] += w * (rb["joint_vel"][:, col] - r["joint_vel"][:, col])
        finally:
            self.ref = main

    def _ref(self, t):
        if self.cfg.stage == "1b1" and t.shape[0] == self.num_envs:
            # legs (and root / key targets) on the lift clock, scaled by leg_ref_scale; upper body at the set position
            z = torch.zeros_like(t)
            r0 = super()._ref(z)
            tm, sgn = self._lift_clock(t)
            r = super()._ref(tm)
            k, legs = self.cfg.leg_ref_scale, self.lead_leg_cols + self.back_leg_cols
            if not self.cfg.lift_back_leg_athlete:  # back leg stays at the set position (the athlete's back leg
                legs = self.lead_leg_cols  # already drifts toward the plate during the lift: with zero actions the
                # pelvis ran +0.2 m toward the plate by the peak)
            q = r0["joint_pos"].clone()
            q[:, legs] = q[:, legs] + k * (r["joint_pos"][:, legs] - q[:, legs])
            qd = torch.zeros_like(r["joint_vel"])
            qd[:, legs] = r["joint_vel"][:, legs] * k * sgn.unsqueeze(-1)
            if self.cfg.lift_shift_roll or self.cfg.lift_shift_pitch:
                # weight onto the back leg, ramped in over the set-position wait: both hip rolls / pitches + x, ankle
                # rolls / pitches - x (a parallel shift of the pelvis over the feet). Zero-action scan of 121
                # (roll, pitch) pairs: 0.45 / 0.20 keeps the COM 4 cm from the back foot through the lift, stays up 3 s
                c = self.cfg
                w = (t / max(c.lift_delay_s, 1e-3)).clamp(0.0, 1.0)
                # ... and out again while the lead foot comes down (held, the lead foot never reached the ground)
                t_dn = c.lift_delay_s + c.lift_peak_s + c.lift_hold_s
                out = ((t - t_dn) / (c.lift_peak_s + c.lift_shift_out_extra_s)).clamp(0.0, 1.0)
                w = w * (1.0 - out)
                # ... to a small end offset instead of the bare set position (zero actions settled to 0.12 m/s at
                # 2.3 s, then tipped back: 0.64 m/s at 3 s; scan of 40 end offsets: roll -0.05 / pitch 0 at rest)
                roll = c.lift_shift_roll * w + c.lift_end_roll * out
                pitch = c.lift_shift_pitch * w + c.lift_end_pitch * out
                for side in ("l", "r"):
                    for j, v in (("hip_roll", 1.0), ("ankle_roll", -1.0)):
                        q[:, self._jcol[f"{side}_{j}"]] += v * roll
                    for j, v in (("hip_pitch", 1.0), ("ankle_pitch", -1.0)):
                        q[:, self._jcol[f"{side}_{j}"]] += v * pitch
            r["joint_pos"], r["joint_vel"] = q, qd
            for key in ("root_pos", "key_pos"):
                r[key] = r0[key] + k * (r[key] - r0[key])
            r["root_quat"] = r0["root_quat"]
            r["root_lin_vel"] = r["root_lin_vel"] * k * sgn.unsqueeze(-1)
            r["root_ang_vel"] = torch.zeros_like(r["root_ang_vel"])
            return r
        if self.cfg.stage == "1c" and t.shape[0] == self.num_envs:
            return self._ref_stride(t)
        r = super()._ref(t)
        if self.cfg.arm_ref_lead_s and t.shape[0] == self.num_envs:  # per-joint time shift of the throw reference
            for nm, lead in self.cfg.arm_ref_lead_s.items():
                rr, col = super()._ref(t + lead), self._jcol[nm]
                r["joint_pos"][:, col], r["joint_vel"][:, col] = rr["joint_pos"][:, col], rr["joint_vel"][:, col]
        if self.ref_arm is not None and t.shape[0] == self.num_envs:
            self._arm_blend(r, t)
        if self.cfg.stage == "1b" and t.shape[0] == self.num_envs:
            # PD base / targets: upper body at the set position; legs on the athlete's stride (frozen at the release
            # time if the robot has not landed by then), after the robot's touchdown blending over stand_pd_blend_s
            # to its own leg angles at the touchdown (hold the stance, stop)
            legs = self.lead_leg_cols + self.back_leg_cols
            if self.cfg.stage_finish == "athlete" and self.cfg.finish_trunk_athlete:
                legs = legs + self.trunk_cols  # trunk rotation / flexion with the athlete (zero actions from 0.7 s:
                # falls at 1.15 s vs 1.40 s without - not used yet)
            z = torch.zeros_like(t)
            q = super()._ref(z)["joint_pos"].clone()
            if self.cfg.stage_finish == "athlete":  # the legs keep the athlete's clock through the follow-through
                # (back leg comes through to the finishing stance at ~1.6 s) instead of freezing the touchdown lunge
                t_leg = t.clamp(max=self.ref.duration)
            else:
                t_leg = torch.where(self.s_landed, self.s_land_t, t.clamp(max=self.release_ref))
            rl = super()._ref(t_leg)
            # root / key point targets (observations) on the legs' clock: frozen at the release time before a
            # touchdown, the robot's own root after it. They ran on with the athlete's follow-through (pelvis 0.5 m
            # toward -y and forward) and the 1b3 policy chased them after landing: pelvis slid 1 m sideways, fell
            r["root_pos"], r["key_pos"] = rl["root_pos"], rl["key_pos"]
            r["root_lin_vel"] = rl["root_lin_vel"] * (~self.s_landed).float().unsqueeze(-1)
            r["root_ang_vel"] = rl["root_ang_vel"] * (~self.s_landed).float().unsqueeze(-1)
            k = self.cfg.leg_ref_scale
            if k != 1.0:  # a smaller version of the athlete's leg lift and stride: q0 + k (q - q0), root likewise
                rl["joint_pos"] = q + k * (rl["joint_pos"] - q)
                rl["joint_vel"] = rl["joint_vel"] * k
                r0 = super()._ref(z)["root_pos"]
                r["root_pos"] = r0 + k * (r["root_pos"] - r0)
            if self.cfg.stage_finish != "athlete":
                r["root_pos"] = torch.where(self.s_landed.unsqueeze(-1), self.s_root_land, r["root_pos"])
            since = (self.episode_length_buf - self.s_land_step).float() * self.step_dt
            w = (self.s_landed.float() * (since / self.cfg.stand_pd_blend_s).clamp(0.0, 1.0)).unsqueeze(-1)
            # finish "hold": the robot's stance at the touchdown; "set": the set position's standing legs (the back
            # leg comes through, as the athlete's does - nobody holds a 1.3 m lunge still)
            fin = self.s_hold[:, legs] if self.cfg.stage_finish == "hold" else q[:, legs]
            if self.cfg.stage_finish == "athlete":
                w = torch.zeros_like(w)
            q[:, legs] = rl["joint_pos"][:, legs] + w * (fin - rl["joint_pos"][:, legs])
            qd = torch.zeros_like(r["joint_vel"])
            moving = (~self.s_landed & (t < self.release_ref)).float().unsqueeze(-1)
            if self.cfg.stage_finish == "athlete":
                moving = (t < self.ref.duration).float().unsqueeze(-1)
            qd[:, legs] = rl["joint_vel"][:, legs] * moving
            if self.cfg.throw_upper:  # arm + ball step: trunk and arms on the athlete's clock (the throw)
                keep = set(legs) | (set() if self.cfg.throw_upper_trunk else set(self.trunk_cols))
                up_cols = [i for i in range(q.shape[1]) if i not in keep]
                k = self.cfg.throw_upper_scale  # fall-fix curriculum: 0 = the stride task's upper body, 1 = the throw
                if self.cfg.throw_upper_scale_joints:  # per-joint scale overrides (regex on the joint name -> scale)
                    import re as _re
                    k = torch.tensor([next((sc for pat, sc in self.cfg.throw_upper_scale_joints.items()
                                            if _re.fullmatch(pat, self.body_names[i])), self.cfg.throw_upper_scale)
                                      for i in up_cols], device=q.device).unsqueeze(0)
                q[:, up_cols] = q[:, up_cols] + k * (r["joint_pos"][:, up_cols] - q[:, up_cols])
                qd[:, up_cols] = k * r["joint_vel"][:, up_cols] * (t < self.ref.duration).float().unsqueeze(-1)
            c = self.cfg
            rs = ((self.episode_length_buf - self.release_step).float() * self.step_dt).unsqueeze(-1)
            relf = self.released.float().unsqueeze(-1)
            if c.follow_trunk_after_release and c.throw_upper and not c.throw_upper_trunk:
                # (b2) the waist joins the athlete's follow-through rotation after the release (blended in over
                # follow_trunk_blend_s): the reference held it at the set pose while the athlete's waist turned -34 -> +80 deg
                tc = self.trunk_cols
                wt = relf * (rs / c.follow_trunk_blend_s).clamp(0.0, 1.0)
                tgt = q[:, tc] + c.throw_upper_scale * (r["joint_pos"][:, tc] - q[:, tc])
                q[:, tc] = q[:, tc] + wt * (tgt - q[:, tc])
                qd[:, tc] = wt * c.throw_upper_scale * r["joint_vel"][:, tc] * (t < self.ref.duration).float().unsqueeze(-1)
            if c.trunk_window_times is not None and c.throw_upper and not c.throw_upper_trunk:
                import re as _re
                tc = [i for i in self.trunk_cols if _re.fullmatch(c.trunk_window_joints, self.body_names[i])]
                t0, t1, t2, t3 = (self.release_ref + x for x in c.trunk_window_times)
                sm = lambda x: x * x * (3.0 - 2.0 * x)  # noqa: E731
                ww = sm(((t - t0) / (t1 - t0)).clamp(0.0, 1.0)) * (1.0 - sm(((t - t2) / (t3 - t2)).clamp(0.0, 1.0)))
                ww = (ww * c.trunk_window_weight).unsqueeze(-1)
                tgt = q[:, tc] + c.throw_upper_scale * (r["joint_pos"][:, tc] - q[:, tc])
                q[:, tc] = q[:, tc] + ww * (tgt - q[:, tc])
                qd[:, tc] = qd[:, tc] + ww * (c.throw_upper_scale * r["joint_vel"][:, tc] - qd[:, tc])
            if c.back_catch_target is not None:
                self._back_catch_cartesian(q, qd, rs, relf)
            if c.back_catch_q is not None:
                # (b1) back-leg catch step: after the release the back leg's reference blends from the athlete's late
                # forward step (touchdown +0.72 s, 0.41 m ahead) to the robot standers' early wide stance (their back-leg
                # angles at release + 0.45 s), over [back_catch_t0, back_catch_t0 + back_catch_blend_s], then holds it
                bc = self.back_leg_cols
                wb = relf * ((rs - c.back_catch_t0) / c.back_catch_blend_s).clamp(0.0, 1.0)
                tgt = torch.tensor(c.back_catch_q, device=q.device).unsqueeze(0)
                q[:, bc] = q[:, bc] + wb * (tgt - q[:, bc])
                qd[:, bc] = qd[:, bc] * (1.0 - wb)
            r["joint_pos"], r["joint_vel"] = q, qd
        if self.cfg.stage == "1a":  # a still pose: the athlete's first-frame velocities (the leg lift starting) would
            # be a constant PD velocity target - with zero actions it lifted the lead foot 16 cm and the robot fell
            for k in ("joint_vel", "root_lin_vel", "root_ang_vel"):
                r[k] = torch.zeros_like(r[k])
        return r

    def _back_catch_cartesian(self, q, qd, rs, relf):
        """(b1') back-foot catch step, Cartesian: from release + back_catch_t0 the back foot's reference moves from where
        it is to back_catch_target (x toward the plate, y toward first base, from the lead foot, ground frame; foot flat,
        yaw = lead foot yaw + back_catch_yaw_deg) by release + back_catch_t1 on a smoothstep with a lift arc, then holds
        it; per-step leg IK on the current pelvis pose gives the back-leg PD targets, blended in over back_catch_wblend_s
        from the athlete path. Writes q / qd (back-leg columns) in place."""
        from isaaclab.utils.math import matrix_from_quat
        from aibaseball.robot.leg_ik import LegIK, rotvec_of

        c, d, dev = self.cfg, self.robot.data, self.device
        if not hasattr(self, "_cat_ik"):
            self._cat_ik = LegIK("r", device=dev)
            n = self.num_envs
            self._cat_init = torch.zeros(n, dtype=torch.bool, device=dev)
            self._cat_p0 = torch.zeros(n, 3, device=dev)
            self._cat_R0 = torch.eye(3, device=dev).repeat(n, 1, 1)
            self._cat_q = torch.zeros(n, 6, device=dev)
        bc = self.back_leg_cols
        rel = self.released
        new = rel & ~self._cat_init & (rs[:, 0] >= c.back_catch_t0)
        if new.any():
            fid = self.foot_ids[1]
            self._cat_p0[new] = d.body_link_pos_w[new, fid]
            self._cat_R0[new] = matrix_from_quat(d.body_link_quat_w[new, fid])
            self._cat_q[new] = d.joint_pos[new][:, self.body_ids[bc]]
            self._cat_init |= new
        act = rel & self._cat_init
        if not act.any():
            return
        ids = act.nonzero(as_tuple=False).squeeze(-1)
        lead = d.body_link_pos_w[ids, self.foot_ids[0]]
        lq = d.body_link_quat_w[ids, self.foot_ids[0]]
        lyaw = torch.atan2(2 * (lq[:, 0] * lq[:, 3] + lq[:, 1] * lq[:, 2]), 1 - 2 * (lq[:, 2] ** 2 + lq[:, 3] ** 2))
        tgt = lead.clone()
        tgt[:, 0] += c.back_catch_target[0]
        tgt[:, 1] += c.back_catch_target[1]
        x = tgt[:, 0] - self.scene.env_origins[ids, 0]
        ground = torch.where(x < c.mound_slope_start_x, torch.full_like(x, c.mound_top_z),
                             (c.mound_top_z - (x - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
        tgt[:, 2] = ground + (self.s_feet0[ids, 1, 2] - c.mound_top_z)  # ankle height over the sole when flat
        yaw = lyaw + math.radians(c.back_catch_yaw_deg)
        cy, sy, z0, o1 = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        R_t = torch.stack([torch.stack([cy, -sy, z0], -1), torch.stack([sy, cy, z0], -1), torch.stack([z0, z0, o1], -1)], -2)
        s = ((rs[ids, 0] - c.back_catch_t0) / (c.back_catch_t1 - c.back_catch_t0)).clamp(0.0, 1.0)
        sig = (s * s * (3.0 - 2.0 * s)).unsqueeze(-1)
        p0, R0 = self._cat_p0[ids], self._cat_R0[ids]
        p = p0 + sig * (tgt - p0)
        p[:, 2] = p[:, 2] + 4.0 * c.back_catch_apex_m * s * (1.0 - s)
        v = rotvec_of(R0.transpose(-1, -2) @ R_t) * sig  # slerp R0 -> R_t
        ang = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        k = v / ang
        K = torch.zeros(len(ids), 3, 3, device=dev)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
        K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
        a_ = ang.unsqueeze(-1)
        R = R0 @ (torch.eye(3, device=dev) + torch.sin(a_) * K + (1 - torch.cos(a_)) * (K @ K))
        qi, _, _ = self._cat_ik.solve(matrix_from_quat(d.root_link_quat_w[ids]), d.root_link_pos_w[ids], R, p,
                                      self._cat_q[ids], iters=4)
        self._cat_q[ids] = qi
        wb = ((rs[ids] - c.back_catch_t0) / c.back_catch_wblend_s).clamp(0.0, 1.0)
        qb = q[ids][:, bc]
        q[ids.unsqueeze(-1), torch.tensor(bc, device=dev)] = qb + wb * (qi - qb)
        qd[ids.unsqueeze(-1), torch.tensor(bc, device=dev)] = qd[ids][:, bc] * (1.0 - wb)

    def _ref_stride(self, t):
        """1c (stride + rotation, one-month plan W1): the athlete's whole lower body and trunk - lift, stride toward
        the plate, hip / trunk rotation, follow-through to the finishing stance - on a clock that starts after
        lift_delay_s in the set position, scaled by leg_ref_scale around the set position, played at stride_speed,
        held at its last frame. A weight shift onto the back leg (hip rolls / pitches +, ankle rolls / pitches -)
        ramps in over the wait and out between stride_shift_out_t0 and _t1 (athlete time). Arms stay at the set
        position (freeze_arms); the ball comes in the next step."""
        c = self.cfg
        z = torch.zeros_like(t)
        r0 = super()._ref(z)
        te = ((t - c.lift_delay_s) * c.stride_speed).clamp(0.0, self.ref.duration)
        r = super()._ref(te)
        k = c.leg_ref_scale
        cols = self.lead_leg_cols + self.back_leg_cols + (self.trunk_cols if c.stride_trunk else [])
        q = r0["joint_pos"].clone()
        q[:, cols] = q[:, cols] + k * (r["joint_pos"][:, cols] - q[:, cols])
        moving = ((t > c.lift_delay_s) & (te < self.ref.duration)).float().unsqueeze(-1)
        qd = torch.zeros_like(r["joint_vel"])
        qd[:, cols] = r["joint_vel"][:, cols] * k * c.stride_speed * moving
        w = (t / max(c.lift_delay_s, 1e-3)).clamp(0.0, 1.0)
        w = w * (1.0 - ((te - c.stride_shift_out_t0) / max(c.stride_shift_out_t1 - c.stride_shift_out_t0, 1e-3))
                 .clamp(0.0, 1.0))
        for side in ("l", "r"):
            for j, v in (("hip_roll", 1.0), ("ankle_roll", -1.0)):
                q[:, self._jcol[f"{side}_{j}"]] += v * c.lift_shift_roll * w
            for j, v in (("hip_pitch", 1.0), ("ankle_pitch", -1.0)):
                q[:, self._jcol[f"{side}_{j}"]] += v * c.lift_shift_pitch * w
        r["joint_pos"], r["joint_vel"] = q, qd
        for key in ("root_pos", "key_pos"):
            r[key] = r0[key] + k * (r[key] - r0[key])
        r["root_lin_vel"] = r["root_lin_vel"] * k * c.stride_speed * moving
        r["root_ang_vel"] = r["root_ang_vel"] * c.stride_speed * moving
        return r

    def stride_ok(self):
        """1c: lead foot moved >= lead_advance_min_m toward the plate (+x) from its start, back foot not more than
        back_retreat_max_m behind its start (the 1b4 1750 'stride' stepped the back foot away from the plate)."""
        c, feet = self.cfg, self.robot.data.body_link_pos_w[:, self.foot_ids]
        return (((feet[:, 0, 0] - self.s_feet0[:, 0, 0]) >= c.lead_advance_min_m)
                & ((feet[:, 1, 0] - self.s_feet0[:, 1, 0]) >= -c.back_retreat_max_m))

    def _apply_action(self):
        if self.cfg.assist_force_n > 0:
            # assist-force curriculum (1b-2 forward stride): a horizontal push on the pelvis toward the plate during the
            # athlete's drive phase, before the robot's own touchdown, decayed to 0 over assist_decay_steps env steps
            # - PPO gets forward strides that land to learn from, and the final policy runs on full physics alone
            c = self.cfg
            # (assist_steps0: env steps of earlier cycles - train.py sets it from ASSIST_START, each 150-iteration
            # cycle is a new process whose step counter starts at 0)
            decay = max(0.0, 1.0 - (self.assist_steps0 + self.common_step_counter) / max(c.assist_decay_steps, 1))
            tr = self._ref_time()
            on = ((tr >= c.assist_t0) & (tr < c.assist_t1) & ~self.s_landed & ~self.fallen).float()
            f_w = torch.zeros(self.num_envs, 3, device=self.device)
            f_w[:, 0] = c.assist_force_n * decay * on
            f_b = quat_rotate_inverse(self.robot.data.root_link_quat_w, f_w)  # the API takes the body frame
            self.robot.set_external_force_and_torque(f_b.unsqueeze(1), torch.zeros_like(f_b).unsqueeze(1),
                                                     body_ids=[self.pelvis_id])
        super()._apply_action()

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clone()
        rel_a = actions[:, 29 + self.cfg.release_action_col].clamp(-1.0, 1.0)  # release action (a finger column)
        actions[:, 29:] = 0.0  # fingers keep the grip (no release in 1a)
        if self.cfg.freeze_arms:  # arms on their PD reference (the set position): no policy action, no action noise
            actions[:, self._arm_cols] = 0.0
        if self.cfg.release_neutral:  # PitchEnv's release-keyed PD (stand blend, frozen base pose, gains, feed-forward)
            # was built for the old whole-pitch task; here the legs follow the same reference before and after
            rel, self.released = self.released, torch.zeros_like(self.released)
            super()._pre_physics_step(actions)
            self.released = rel
        else:
            super()._pre_physics_step(actions)
        c = self.cfg
        if c.release_action_window_s is not None:
            # the policy picks the release time: fingers open (latched) once the reference time passes
            # release_ref + window x a, a = one finger column in [-1, 1] - before release_ref - window it cannot fire,
            # at release_ref + window it is forced; a = 0 opens exactly at release_ref (scripted_release_s 0.0's step)
            tr, w = self._ref_time(), c.release_action_window_s
            fire = tr >= self.release_ref + w * rel_a
            forced = tr >= self.release_ref + w
            self.s_rel_forced |= forced & ~fire & ~self.s_open
            self.s_open |= fire | forced
            opened = (self.hold_q - c.finger_action_scale).expand(self.num_envs, -1)
            self.finger_next = torch.where(self.s_open.unsqueeze(-1),
                                           torch.clamp(opened, self.q_lo[self.finger_ids], self.q_hi[self.finger_ids]),
                                           self.finger_next)
        if c.release_ahead_m is not None:
            # release when the throwing hand passes release_ahead_m in front of the pelvis (toward the plate), not at
            # the athlete's release time: the robot's arm lags (s1d2 2350: at 0.98 s the hand was ~1 m behind the
            # pelvis and the ball left sideways at (-0.5, -11.5, +4.6) m/s; the athlete's hand is 0.37 m ahead)
            d, tr = self.robot.data, self._ref_time()
            ahead = (d.body_link_pos_w[:, self.hand_id, 0] - d.root_link_pos_w[:, 0]) >= c.release_ahead_m
            if c.release_vy_frac is not None:
                # ... and when the hand's path points at the plate: its sideways velocity turns from the first-base
                # side to + release_vy_frac x speed (the athlete's hand moves straight at the plate just before the
                # release and swings 10 m/s toward third base 0.02 s later; s1d3 / s1d4 balls crossed 5-11 m wide)
                hv = d.body_link_lin_vel_w[:, self.hand_id]
                ahead = ahead & (hv[:, 0] > 5.0) & (hv[:, 1] >= c.release_vy_frac * hv[:, :2].norm(dim=-1))
            if c.release_heading_deg is not None:
                # ... or the hand's horizontal velocity heading within +-release_heading_deg of the plate line (+x):
                # the athlete's hand heading sweeps -60 -> +45 deg over 0.95-1.0 s (~3700 deg/s, ~9 deg per 2.5 ms
                # step) and crosses 0 at ~0.975 s (release 0.98 s); gk 4525 released at -37 deg on the athlete's time
                hv = d.body_link_lin_vel_w[:, self.hand_id]
                hd = torch.rad2deg(torch.atan2(hv[:, 1], hv[:, 0]))
                ahead = ahead & (hv[:, 0] > 5.0) & (hd.abs() <= c.release_heading_deg)
            if c.release_elev_deg is not None:
                # ... and the hand's velocity elevation (deg above horizontal) inside (lo, hi): gm 4825 released on the
                # plate line but ~48 deg downward from z 1.03 m (the ball landed short at 15 m/s)
                hv = d.body_link_lin_vel_w[:, self.hand_id]
                el = torch.rad2deg(torch.atan2(hv[:, 2], hv[:, :2].norm(dim=-1)))
                ahead = ahead & (el >= c.release_elev_deg[0]) & (el <= c.release_elev_deg[1])
            self.s_open |= (ahead & (tr >= c.release_window[0])) | (tr >= c.release_window[1])
            opened = (self.hold_q - c.finger_action_scale).expand(self.num_envs, -1)
            self.finger_next = torch.where(self.s_open.unsqueeze(-1),
                                           torch.clamp(opened, self.q_lo[self.finger_ids], self.q_hi[self.finger_ids]),
                                           self.finger_next)

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

    def _update_stride(self):
        """1b: a lead-foot touchdown (the step it comes down, after it left the ground) >= lead_stride_min_m ahead of
        the back foot toward the plate. Checked only at the touchdown: a short step that stays planted while the back
        foot slides back (0.29 m -> 0.54 m in the zero-action test) is no stride; the lead foot may lift and try again."""
        c, d = self.cfg, self.robot.data
        forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values
        planted = forces > c.contact_force_n
        feet = d.body_link_pos_w[:, self.foot_ids]
        stride = feet[:, 0, 0] - feet[:, 1, 0]
        tr = self._ref_time()
        self.s_lifted |= ~planted[:, 0] & (tr >= c.lead_lift_after_s)
        down = planted[:, 0] & ~self.s_prev_planted
        self.s_prev_planted = planted[:, 0].clone()
        # toward the plate: the lead foot itself moved >= lead_advance_min_m forward of its start, and the back foot
        # did not step back (1b4 1750 "strode" 0.73 m by putting the BACK foot 0.8 m away from the plate)
        ahead = (feet[:, 0, 0] - self.s_feet0[:, 0, 0]) >= c.lead_advance_min_m
        kept = (feet[:, 1, 0] - self.s_feet0[:, 1, 0]) >= -c.back_retreat_max_m
        new = (down & self.s_lifted & ~self.s_landed & (stride >= c.lead_stride_min_m) & ahead & kept
               & ~self.fallen)
        self.s_landed |= new
        self.s_land_t = torch.where(new, tr, self.s_land_t)
        self.s_land_step = torch.where(new, self.episode_length_buf, self.s_land_step)
        self.s_hold = torch.where(new.unsqueeze(-1), d.joint_pos[:, self.body_ids], self.s_hold)
        self.s_root_land = torch.where(new.unsqueeze(-1), d.root_link_pos_w, self.s_root_land)
        return new, stride

    def _update_lift(self):
        """1b1: time on the back leg alone with the lead foot >= lift_min_m above its start (back foot planted), and
        whether the back foot ever left its spot (> back_slide_max_m horizontally). Returns (lift done now, slide)."""
        c, d = self.cfg, self.robot.data
        forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values
        planted = forces > c.contact_force_n
        feet = d.body_link_pos_w[:, self.foot_ids]
        high = (feet[:, 0, 2] - self.s_feet0[:, 0, 2]) >= c.lift_min_m
        on_back = high & ~planted[:, 0] & planted[:, 1] & ~self.fallen
        before = self.s_lift_t >= c.lift_hold_min_s
        self.s_lift_t += on_back.float() * self.step_dt
        new = ~before & (self.s_lift_t >= c.lift_hold_min_s)
        slide = (feet[:, 1, :2] - self.s_feet0[:, 1, :2]).norm(dim=-1)
        self.s_slid |= slide > c.back_slide_max_m
        return new, slide

    def lift_done(self):
        """1b1: held the lift on the back leg long enough, back foot never slid, lead foot back near its start."""
        c = self.cfg
        home = (self.robot.data.body_link_pos_w[:, self.foot_ids[0], :2] - self.s_feet0[:, 0, :2]).norm(dim=-1)
        return (self.s_lift_t >= c.lift_hold_min_s) & ~self.s_slid & (home < c.lead_home_max_m)

    def stage_success(self):
        """At rest standing now (and, 1b, after a real stride; 1b1, after the lift, feet back in place)."""
        ok = self._stand_state()[-1]
        if self.cfg.stage == "1b1":
            return ok & self.lift_done()
        if self.cfg.stage == "1c":
            return ok & self.stride_ok()
        return ok & self.s_landed if self.cfg.stage == "1b" else ok

    def release_score(self) -> torch.Tensor:
        """0..1 pitch score from the release state (gravity only): 0.4 A + 0.6 B, A = exp(-(heading / sigma)^2) x
        min(v / 20, 1) x min(x reached in the air / plate distance, 1), B = crossed the plate in the air x
        exp(-(y / 0.5)^2) x exp(-((z - 0.8) / 0.6)^2) (y from the zone centre, z above the plate-level ground)."""
        c = self.cfg
        v = self.rel_vel
        p = self.rel_pos - self.scene.env_origins
        vx = v[:, 0].clamp_min(1e-3)
        head = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0]))
        g = 9.81
        h = (p[:, 2] - self.field_z - 0.037).clamp_min(0.0)  # ball centre over the ground at contact (radius)
        t_g = (v[:, 2] + torch.sqrt(v[:, 2] ** 2 + 2 * g * h)) / g
        t_p = ((c.plate_distance - p[:, 0]) / vx).clamp_min(0.0)
        air = (t_g >= t_p) & (v[:, 0] > 0)
        x_air = p[:, 0] + vx * torch.minimum(t_g, t_p)
        fh = torch.exp(-((head / c.ball_rest_head_sigma_deg) ** 2))
        fs = (v.norm(dim=-1) / 20.0).clamp(max=1.0)
        fd = (x_air / c.plate_distance).clamp(0.0, 1.0)
        if c.ball_rest_no_dist:
            a = ((fh + fs) / 2.0 if c.ball_rest_additive else fh * fs) * (v[:, 0] > 0).float()
        else:
            a = ((fh + fs + fd) / 3.0 if c.ball_rest_additive else fh * fs * fd) * (v[:, 0] > 0).float()
        y = p[:, 1] + v[:, 1] * t_p - c.zone_center_y
        z = p[:, 2] + v[:, 2] * t_p - 0.5 * g * t_p**2 - self.field_z
        b = air.float() * torch.exp(-((y / 0.5) ** 2)) * torch.exp(-(((z - 0.8) / 0.6) ** 2))
        return 0.4 * a + 0.6 * b

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        if not hasattr(self, "lead_knee_jid"):
            self.lead_knee_jid = self.robot.find_joints("l_knee")[0][0]
        both, speed, tilt, head_z, ok = self._stand_state()
        new = torch.zeros_like(self.fallen)
        q_on = torch.ones_like(speed)
        if c.stage == "1b1":  # standing quality counts once the lift is done (feet back home); + a one-off lift bonus
            new, slide = self._update_lift()
            done_lift = self.lift_done()
            q_on = done_lift.float()
            ok = ok & done_lift
        elif c.stage == "1b":  # standing quality counts once the stride has landed; + a one-off stride bonus
            new, stride = self._update_stride()
            q_on = self.s_landed.float()
            ok = ok & self.s_landed
        steps = max(int(round(c.episode_length_s / self.step_dt)), 1)
        up = ~self.fallen
        upright = torch.exp(-(((tilt - c.upright_start_frac * c.stand_max_tilt_deg).clamp_min(0.0) / c.upright_width_deg) ** 2))
        still = torch.exp(-((speed / (2.0 * c.stand_max_com_v)) ** 2))
        tall = ((head_z - 1.0) / (c.stand_min_head_z + 0.05 - 1.0)).clamp(0.0, 1.0)
        support = both.float()
        if c.stage == "1b1" and c.lift_balance_shaping:
            # from lift_balance_lead_s before the lift until the lead foot is due down again: COM over the back foot
            # (1b1 850-1000: never put the weight on the back leg, the COM 0.18 m off it ran away at 1.6 m/s)
            tr = self._ref_time() - c.lift_delay_s
            win = (tr >= -c.lift_balance_lead_s) & (tr < 2 * c.lift_peak_s + c.lift_hold_s)
            com, _ = self._com()
            back = self.robot.data.body_link_pos_w[:, self.foot_ids[1], :2]
            over = torch.exp(-(((com[:, :2] - back).norm(dim=-1) / c.lift_balance_sigma_m) ** 2))
            support = torch.where(win, over, support)
            q_on = torch.ones_like(speed)  # standing quality all episode (balance before the lift is the point)
        if c.throw_upper and c.brake_lead_sigma_m > 0:
            # braking after the release: "support" also needs the COM not ahead of the lead foot (fg D75 3100: after the
            # release the COM ran 1+ m past the lead foot at 0.8-1.9 m/s, walked down the mound's front slope, fell)
            com, _ = self._com()
            feet = self.robot.data.body_link_pos_w[:, self.foot_ids]
            if c.brake_mode == "support":
                # COM over the feet on the ground (a human finish: the back foot steps forward and the weight settles
                # over the feet; the "lead" mode penalised exactly that, the COM ahead of the lead foot): horizontal
                # distance from the COM to the segment between both feet when both are down, else to the one planted
                # foot; in the air (mid-step) no penalty
                forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values
                planted = forces > c.contact_force_n
                p = com[:, :2]
                a, b = feet[:, 0, :2], feet[:, 1, :2]
                ab = b - a
                s = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
                d_seg = (p - (a + s.unsqueeze(-1) * ab)).norm(dim=-1)
                d_a, d_b = (p - a).norm(dim=-1), (p - b).norm(dim=-1)
                dist = torch.where(planted[:, 0] & planted[:, 1], d_seg,
                                   torch.where(planted[:, 0], d_a, torch.where(planted[:, 1], d_b, torch.zeros_like(d_a))))
                over = torch.exp(-(((dist - c.brake_support_tol_m).clamp_min(0.0) / c.brake_lead_sigma_m) ** 2))
            else:
                over = torch.exp(-(((com[:, 0] - feet[:, 0, 0]).clamp_min(0.0) / c.brake_lead_sigma_m) ** 2))
            support = torch.where(self.released, over * (0.5 + 0.5 * support), support)
        if c.plant_brake_sigma > 0:
            # braking at the lead-foot plant (diag_fall ft 3825, start noise 0.02): the episodes that fell at ~1.8 s
            # carried 1.44 m/s forward COM speed over 1.0-1.4 s vs 0.78 for the ones standing (athlete ~0.9 after
            # killing 2.3 -> 0.8 m/s in 0.1 s); their lead foot then slid and lifted. In that window "support" also
            # needs the forward COM speed under plant_brake_v: x exp(-((v_x - v)+ / sigma)^2)
            tr = self._ref_time()
            _, com_v = self._com()
            win = (tr >= c.plant_brake_t0) & (tr < c.plant_brake_t1)
            slow = torch.exp(-(((com_v[:, 0] - c.plant_brake_v).clamp_min(0.0) / c.plant_brake_sigma) ** 2))
            support = torch.where(win, support * slow, support)
        if c.lead_knee_sigma > 0:
            # a firm front leg (diag_fall fu 3850, start noise 0.02): the robot planted with the lead knee at 1.16 rad
            # and kept it at 0.6-0.8 rad, lead-foot horizontal force only 50-65 N (fallers and standers alike); the
            # athlete plants at ~0.9 rad and straightens it by 1.10 s (5-7 rad/s), blocking the stride's momentum
            # through the leg. Over reference time [lead_knee_t0, _t1) "support" also needs the lead knee under
            # lead_knee_max: x exp(-((q - max)+ / sigma)^2)
            tr = self._ref_time()
            win = (tr >= c.lead_knee_t0) & (tr < c.lead_knee_t1)
            qk = self.robot.data.joint_pos[:, self.lead_knee_jid]
            firm = torch.exp(-(((qk - c.lead_knee_max).clamp_min(0.0) / c.lead_knee_sigma) ** 2))
            support = torch.where(win, support * firm, support)
        if c.com_inboard_sigma > 0:
            # COM inboard of the lead foot in the finish (diag_fall fz 4100): at 1.8 s, on the lead foot alone, the
            # late fallers' COM was +0.23 m toward first base of the lead foot and they tipped over it (1B side 22/25);
            # the standers' -0.35 m (their back foot had landed 0.50 m to the 3B side at ~1.39 s vs 0.23 m). Athlete:
            # pelvis -0.03 m of the lead foot. Over reference time [com_inboard_t0, _t1) "support" also needs
            # COM y - lead foot y under com_inboard_max: x exp(-((dy - max)+ / sigma)^2)
            tr = self._ref_time()
            win = (tr >= c.com_inboard_t0) & (tr < c.com_inboard_t1)
            com, _ = self._com()
            dy = com[:, 1] - self.robot.data.body_link_pos_w[:, self.foot_ids[0], 1]
            inb = torch.exp(-(((dy - c.com_inboard_max).clamp_min(0.0) / c.com_inboard_sigma) ** 2))
            support = torch.where(win, support * inb, support)
        quality = (upright + still + support + tall) / 4.0
        rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        if c.throw_upper and c.w_stage_ball > 0:  # ball outcome, once when its flight ends: progress toward the plate
            # (farthest x / plate distance) + a zone bonus - the 4th term (the stride bonus is off in this step)
            prog = (self.s_ball_maxx / c.plate_distance).clamp(0.0, 1.0)
            if c.ball_aim_sigma_m > 0:
                # graded aim: x progress counts only toward the zone - progress x exp(-(distance from the zone centre
                # in the plate plane (or, short of the plate, sideways from the plate line) / sigma)^2); a ball that
                # crossed 10 m wide scored full progress before (s1d3 2650: plate y +9.9 m, 0 in the zone)
                b = self.ball.data.root_pos_w - self.scene.env_origins
                yz = torch.where(self.s_crossed.unsqueeze(-1), self.s_cross_yz,
                                 torch.stack([b[:, 1], b[:, 2] - self.field_z], -1))
                zc = c.zone_bottom + 0.5 * c.zone_height
                dist = torch.sqrt((yz[:, 0] - c.zone_center_y) ** 2 + (self.s_crossed.float() * (yz[:, 1] - zc)) ** 2)
                prog = prog * torch.exp(-((dist / c.ball_aim_sigma_m) ** 2))
            rew_ball = self.s_ball_end.float() * (c.w_stage_ball * prog + c.w_stage_zone * self.in_zone().float())
        else:
            rew_ball = torch.zeros_like(speed)
        if c.throw_upper and c.w_stage_ball_rest > 0:
            # pitch score from the release state, paid once at the time-out and only if the episode ends at rest (the
            # W1 gate counts "at rest and T3+ on the same pitch"); gravity-only flight like eval_stage's tiers. The
            # old progress term paid a rolling ball in full (s_ball_maxx) - the gm failure mode (plate z 0.04 m)
            score = self.release_score()
            pay = self.reset_time_outs & ok & ~self.fallen & self.released & ~self.recovery_ep
            if c.ball_rest_set_only:  # only where the drop rule applies (set starts): RSI starts threw with no drop
                pay = pay & (self.t0 <= 0.0)
            rew_ball = rew_ball + pay.float() * c.w_stage_ball_rest * score
            if c.w_stage_rel_speed > 0:  # release speed on its own, same pay condition (gw: speed was ~0.5 per m/s)
                rew_ball = rew_ball + pay.float() * c.w_stage_rel_speed * (
                    self.rel_vel.norm(dim=-1) / c.rel_speed_full_mps).clamp(max=1.0)
        if c.w_finish_com > 0:
            # finish shaping (ha, peer decision 2026-10-20): over [release + t0, release + t1] keep the COM behind and
            # inboard (3B side) of the lead foot - diag_fall_release: at release + 0.3 s the standers had it ~8 cm behind
            # / ~10 cm inboard, the fallers at or past it (forward, toward 1B), and faster throws pushed it further.
            # Per step w x mean(f_x, f_y) x dt / window: the whole window pays w at most
            since = (self.episode_length_buf - self.release_step).float() * self.step_dt
            on = self.released & ~self.fallen & (since >= c.finish_com_t0) & (since < c.finish_com_t1)
            com, _ = self._com()
            lead = self.robot.data.body_link_pos_w[:, self.foot_ids[0]]
            dx, dy = com[:, 0] - lead[:, 0], com[:, 1] - lead[:, 1]  # + toward the plate / toward first base
            fx = torch.exp(-(((dx + c.finish_com_margin).clamp_min(0.0) / c.finish_com_sigma) ** 2))
            fy = torch.exp(-(((dy + c.finish_com_margin).clamp_min(0.0) / c.finish_com_sigma) ** 2))
            rew_ball = rew_ball + on.float() * c.w_finish_com * 0.5 * (fx + fy) * self.step_dt / (c.finish_com_t1 - c.finish_com_t0)
        rew = (rew_ball + up.float() * (c.w_stage_alive + c.w_stage_quality * quality * q_on) / steps + c.w_stage_stride * new.float()
               - c.w_stage_fall * (self.fallen | self.s_drop).float() - c.w_stage_rate * rate)
        done = self.reset_terminated | self.reset_time_outs
        a = 0.02
        # by start type (the eval runs full pitches from the set position only; RSI starts mid-pitch at t0 > 0):
        # success = the episode timed out standing at rest, over all ended episodes of that start type
        set_start = self.t0 <= 0.0
        win = (ok & self.reset_time_outs & ~self.fallen).float()
        for k, v, m in (("fall_rate_set", self.fallen.float(), done & set_start), ("success_set", win, done & set_start),
                        ("fall_rate_rsi", self.fallen.float(), done & ~set_start), ("success_rsi", win, done & ~set_start),
                        ("drop_rate_set", self.s_drop.float(), done & set_start),
                        ("drop_rate_rsi", self.s_drop.float(), done & ~set_start),
                        ("fall_rate", self.fallen.float(), done), ("stand_ok", ok.float(), self.reset_time_outs),
                        ("tilt_deg", tilt, up), ("com_speed", speed, up), ("head_z", head_z, up),
                        ("both_feet", both.float(), up), ("landed", self.s_landed.float(), self.reset_time_outs),
                        ("stride_m", stride if c.stage == "1b" else speed, new),
                        ("lift_s", self.s_lift_t, self.reset_time_outs),
                        ("back_slide_m", slide if c.stage == "1b1" else speed, self.reset_time_outs)):
            n = m.float().sum()
            mean = (v * m.float()).sum() / n.clamp_min(1.0)
            self.stage_stats[k] = torch.where(n > 0, (1 - a) * self.stage_stats[k] + a * mean, self.stage_stats[k])
        self.extras["log"] = {f"stage/{k}": v.clone() for k, v in self.stage_stats.items()}
        return rew

    def _track_ball(self):
        """Arm + ball step: release detection (PitchEnv._check_release) and the ball at the plate - first crossing of
        x = plate_distance (y, z above the plate-level ground), else the farthest x it reached."""
        self._check_release()
        b = self.ball.data.root_pos_w - self.scene.env_origins
        self.s_ball_maxx = torch.maximum(self.s_ball_maxx, torch.where(self.released, b[:, 0], self.s_ball_maxx))
        new = self.released & ~self.s_crossed & (b[:, 0] >= self.cfg.plate_distance)
        self.s_cross_yz = torch.where(new.unsqueeze(-1), torch.stack([b[:, 1], b[:, 2] - self.field_z], -1), self.s_cross_yz)
        self.s_crossed |= new
        # the flight ends at the plate or on the ground (mound / field): one-off ball outcome reward then
        c = self.cfg
        ground = torch.where(b[:, 0] < c.mound_slope_start_x, torch.full_like(b[:, 0], c.mound_top_z),
                             (c.mound_top_z - (b[:, 0] - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
        since = (self.episode_length_buf - self.release_step).float() * self.step_dt
        down = self.released & (since > 0.05) & (b[:, 2] - ground < 0.06)
        self.s_ball_end = self.released & ~self.s_ball_done & (self.s_crossed | down)
        self.s_ball_done |= self.s_ball_end

    def in_zone(self):
        c, yz = self.cfg, self.s_cross_yz
        return (self.s_crossed & ((yz[:, 0] - c.zone_center_y).abs() <= c.zone_width / 2)
                & (yz[:, 1] >= c.zone_bottom) & (yz[:, 1] <= c.zone_bottom + c.zone_height))

    def _get_dones(self):
        c, d = self.cfg, self.robot.data
        if c.throw_upper:
            self._track_ball()
        # start positions, read on the first step after a reset (the reset's writes show up then). Not "episode
        # step 1": rsl_rl starts training at random episode lengths, and the first episodes kept zeros (world
        # origin) - lifted, back foot "slid" ~100 m
        first = self.s_need0
        if first.any():
            f = first.unsqueeze(-1)
            self.s_feet0 = torch.where(f.unsqueeze(-1), d.body_link_pos_w[:, self.foot_ids], self.s_feet0)
            self.s_pelvis0 = torch.where(f, d.root_link_pos_w, self.s_pelvis0)
            self.s_need0[:] = False
        self.fallen = (d.body_link_pos_w[:, self.pelvis_id, 2] < c.fallen_pelvis_z) | (self._head_z() < c.fallen_head_z)
        if c.fallen_touch_z > 0:
            p = d.body_link_pos_w[:, self.touch_ids]
            x = p[..., 0] - self.scene.env_origins[:, 0:1]
            ground = torch.where(x < c.mound_slope_start_x, torch.full_like(x, c.mound_top_z),
                                 (c.mound_top_z - (x - c.mound_slope_start_x) * c.mound_slope).clamp_min(self.field_z))
            self.fallen |= ((p[..., 2] - ground) < c.fallen_touch_z).any(-1)
        if c.stage == "1b" and c.max_flight_s > 0:  # no flight phase before the touchdown (1b at 1300 leapt 0.3 s)
            forces = self.feet.data.net_forces_w_history[:, :, self.foot_sensor_ids].norm(dim=-1).max(1).values
            air = (forces <= c.contact_force_n).all(-1) & ~self.s_landed
            self.s_air = torch.where(air, self.s_air + self.step_dt, torch.zeros_like(self.s_air))
            self.fallen |= self.s_air > c.max_flight_s
        if self.s_desync:
            self.s_cut = torch.randint(1, int(self.max_episode_length), self.s_cut.shape, device=self.device)
            self.s_desync = False
        time_out = (self.episode_length_buf >= self.max_episode_length - 1) | (self.episode_length_buf >= self.s_cut)
        self.s_drop = torch.zeros_like(self.fallen)
        if c.throw_upper and c.drop_ends:  # a dropped ball is a failed pitch, like a fall (reward: same penalty)
            # only full pitches from the set position (what the eval scores): an RSI start after the athlete's release
            # (t0 up to 1.9 s) has no throw to make (fg / fi ended every such episode as a "drop": RSI 0% success), and
            # mid-windup starts lose the ball by themselves (t0 < 0.9 s rule, fj: 35% of RSI episodes "dropped")
            self.s_drop = self.released & (self.t0 <= 0.0) & ((self.rel_vel.norm(dim=-1) < c.drop_min_speed)
                                           | (self.rel_ref_t < c.drop_min_release_t))
        return self.fallen | self.s_drop, time_out

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if self.cfg.release_neutral:  # the policy keeps seeing the ball in the hand: once it flies off, ball_rel x 10
            # and ball_vrel x 0.1 reach the hundreds (a stander trained without a release fell 64/64 at the release)
            pol = obs["policy"]
            nb, nf = len(self.body_ids), len(self.finger_ids)
            i0 = 2 * nb + 2 * nf + 10  # after joints, fingers, gravity, root velocities, height: ball rel / vrel / ang, released
            if not hasattr(self, "_ball_obs"):
                self._ball_obs = pol[:, i0:i0 + 10].clone()
            held = ~self.released
            self._ball_obs[held] = pol[held, i0:i0 + 10]
            pol[:, i0:i0 + 10] = self._ball_obs
        return obs

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.s_lifted[env_ids] = False
        self.s_landed[env_ids] = False
        self.s_prev_planted[env_ids] = True
        self.s_air[env_ids] = 0.0
        self.s_land_step[env_ids] = 0
        self.s_lift_t[env_ids] = 0.0
        self.s_need0[env_ids] = True
        self.s_cut[env_ids] = int(self.max_episode_length)
        if hasattr(self, "_cat_init"):
            self._cat_init[env_ids] = False
        self.s_slid[env_ids] = False
        self.s_crossed[env_ids] = False
        self.s_ball_maxx[env_ids] = 0.0
        self.s_ball_done[env_ids] = False
        self.s_open[env_ids] = False
        self.s_rel_forced[env_ids] = False
        self.s_ball_end[env_ids] = False
        if self.cfg.stage == "1b" and self.cfg.rsi_prob > 0:
            # reference-state starts mid-stride (PitchEnv wrote the athlete's pose and velocities at t0): the lead foot
            # has already lifted, and the stride is measured from the athlete's set-position feet, not the start
            mid = env_ids[self.t0[env_ids] > 0]
            if len(mid):
                self.s_lifted[mid] = True
                kp = self.ref.sample(torch.zeros(len(mid), device=self.device))["key_pos"]  # l/r hand, l/r foot, ...
                self.s_feet0[mid] = kp[:, 2:4] + self.scene.env_origins[mid].unsqueeze(1)
                self.s_pelvis0[mid] = self.ref.root_pos[0] + self.scene.env_origins[mid]
                # starts after the athlete's lead foot strike (follow-through): the stride has already landed
                late = mid[self.t0[mid] >= self.cfg.rsi_landed_after_s]
                if len(late):
                    self.s_landed[late] = True
                    self.s_land_t[late] = self.t0[late]
                    self.s_land_step[late] = -100000
                    self.s_root_land[late] = self.robot.data.root_link_pos_w[late]
                self.s_need0[mid] = False
                if self.cfg.rsi_upper_ref:  # the upper body starts on this task's reference (the scaled throw), not
                    # the athlete's full swing: fx F25 trained from mid-throw starts whose arms snapped from the
                    # athlete's pose to the 0.25 blend - training falls 0.996 vs 21/64 from the set position
                    r = self._ref(self._ref_time())
                    legs = set(self.lead_leg_cols + self.back_leg_cols)
                    up = [i for i in range(len(self.body_ids)) if i not in legs]
                    jcols = self.body_ids[up] if torch.is_tensor(self.body_ids) else torch.tensor(
                        [self.body_ids[i] for i in up], device=self.device)
                    jp = self.robot.data.joint_pos[mid].clone()
                    jv = self.robot.data.joint_vel[mid].clone()
                    jp[:, jcols] = r["joint_pos"][mid][:, up]
                    jv[:, jcols] = r["joint_vel"][mid][:, up]
                    self.robot.write_joint_state_to_sim(jp, jv, env_ids=mid)
        if self.cfg.stage in ("1a", "1b1"):  # start at rest (the reset wrote the reference's first-frame velocities)
            jp = self.robot.data.joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
            self.qd_target[env_ids] = 0.0
            self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        if self.cfg.start_joint_noise > 0:
            # independent starts: every episode began on the identical reference pose (no state or physics
            # randomisation), so the 64 noise-free eval pitches differed only by solver noise and flipped together
            # (fo S70 45 -> 11). Uniform +-start_joint_noise rad on the body joints, velocities kept
            jp = self.robot.data.joint_pos[env_ids].clone()
            jv = self.robot.data.joint_vel[env_ids].clone()
            b = self.body_ids if torch.is_tensor(self.body_ids) else torch.tensor(self.body_ids, device=self.device)
            amp = self.cfg.start_joint_noise
            if self.cfg.start_noise_ramp_steps > 0:  # ramped in from 0 (fp: full noise at once, set success 3-9%)
                amp *= min(1.0, self.common_step_counter / self.cfg.start_noise_ramp_steps)
            jp[:, b] += (2 * torch.rand(len(env_ids), len(b), device=self.device) - 1) * amp
            self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        if self._ls is not None and self.cfg.landed_start_prob > 0:
            self._landed_start(env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.landed_start_prob])
        if self.cfg.stage == "1a" and self.cfg.push_vel > 0:  # a small random shove at the start (robustness)
            n = len(env_ids)
            v = torch.zeros(n, 6, device=self.device)
            ang = torch.rand(n, device=self.device) * 2 * math.pi
            mag = torch.rand(n, device=self.device) * self.cfg.push_vel
            v[:, 0] = mag * torch.cos(ang)
            v[:, 1] = mag * torch.sin(ang)
            self.robot.write_root_velocity_to_sim(v, env_ids)

    def _landed_start(self, ids):
        """Start these envs at rest in a recorded landed stance: stride done, legs holding that stance (reverse
        curriculum - the stance after the touchdown was reached every episode but never practised)."""
        if len(ids) == 0:
            return
        roots, jps = self._ls
        k = torch.randint(0, len(roots), (len(ids),), device=self.device)
        root = roots[k].clone()
        root[:, :3] += self.scene.env_origins[ids]
        root[:, 2] += self.cfg.reset_lift_m
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, ids)
        jp = jps[k].clone()
        # fingers already at their closed target: with no ball in the hand they slammed shut from the grip pose and
        # the finger velocities (still all 1a ever saw) came out 30-70 normalized units -> leg actions up to 4
        jp[:, self.finger_ids] = self.hold_q
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        for a in self._seas():
            a.motor_pos[ids] = jp[:, a.joint_indices]
            a.motor_vel[ids] = 0.0
            a.needs_sync[ids] = False
        self.q_target[ids] = jp
        self.q_target[ids.unsqueeze(-1), self.finger_ids] = self.hold_q
        self.qd_target[ids] = 0.0
        ball = torch.zeros(len(ids), 13, device=self.device)  # the ball out of the way (not scored in 1b)
        ball[:, :3] = self.scene.env_origins[ids] + torch.tensor([-1.5, 1.5, 0.05], device=self.device)
        ball[:, 3] = 1.0
        self.ball.write_root_state_to_sim(ball, ids)
        self.t0[ids] = self.release_ref
        self.s_lifted[ids] = True
        self.s_landed[ids] = True
        self.s_land_t[ids] = self.release_ref
        self.s_land_step[ids] = -100000  # PD base already fully on the held stance
        self.s_hold[ids] = jp[:, self.body_ids]
        self.s_root_land[ids] = root[:, :3]
