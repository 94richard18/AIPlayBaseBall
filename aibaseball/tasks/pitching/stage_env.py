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
                                      "landed", "stride_m", "lift_s", "back_slide_m")}
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
        if self.cfg.stage == "1b" and t.shape[0] == self.num_envs:
            # PD base / targets: upper body at the set position; legs on the athlete's stride (frozen at the release
            # time if the robot has not landed by then), after the robot's touchdown blending over stand_pd_blend_s
            # to its own leg angles at the touchdown (hold the stance, stop)
            legs = self.lead_leg_cols + self.back_leg_cols
            z = torch.zeros_like(t)
            q = super()._ref(z)["joint_pos"].clone()
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
            r["root_pos"] = torch.where(self.s_landed.unsqueeze(-1), self.s_root_land, r["root_pos"])
            since = (self.episode_length_buf - self.s_land_step).float() * self.step_dt
            w = (self.s_landed.float() * (since / self.cfg.stand_pd_blend_s).clamp(0.0, 1.0)).unsqueeze(-1)
            # finish "hold": the robot's stance at the touchdown; "set": the set position's standing legs (the back
            # leg comes through, as the athlete's does - nobody holds a 1.3 m lunge still)
            fin = self.s_hold[:, legs] if self.cfg.stage_finish == "hold" else q[:, legs]
            q[:, legs] = rl["joint_pos"][:, legs] + w * (fin - rl["joint_pos"][:, legs])
            qd = torch.zeros_like(r["joint_vel"])
            moving = (~self.s_landed & (t < self.release_ref)).float().unsqueeze(-1)
            qd[:, legs] = rl["joint_vel"][:, legs] * moving
            r["joint_pos"], r["joint_vel"] = q, qd
        if self.cfg.stage == "1a":  # a still pose: the athlete's first-frame velocities (the leg lift starting) would
            # be a constant PD velocity target - with zero actions it lifted the lead foot 16 cm and the robot fell
            for k in ("joint_vel", "root_lin_vel", "root_ang_vel"):
                r[k] = torch.zeros_like(r[k])
        return r

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
        actions[:, 29:] = 0.0  # fingers keep the grip (no release in 1a)
        if self.cfg.freeze_arms:  # arms on their PD reference (the set position): no policy action, no action noise
            actions[:, self._arm_cols] = 0.0
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

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
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
        upright = torch.exp(-(((tilt - 0.8 * c.stand_max_tilt_deg).clamp_min(0.0) / 15.0) ** 2))
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
        quality = (upright + still + support + tall) / 4.0
        rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        rew = (up.float() * (c.w_stage_alive + c.w_stage_quality * quality * q_on) / steps + c.w_stage_stride * new.float()
               - c.w_stage_fall * self.fallen.float() - c.w_stage_rate * rate)
        done = self.reset_terminated | self.reset_time_outs
        a = 0.02
        for k, v, m in (("fall_rate", self.fallen.float(), done), ("stand_ok", ok.float(), self.reset_time_outs),
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

    def _get_dones(self):
        c, d = self.cfg, self.robot.data
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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.fallen.clone(), time_out

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
        self.s_slid[env_ids] = False
        if self.cfg.stage in ("1a", "1b1"):  # start at rest (the reset wrote the reference's first-frame velocities)
            jp = self.robot.data.joint_pos[env_ids].clone()
            self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
            self.qd_target[env_ids] = 0.0
            self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
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
