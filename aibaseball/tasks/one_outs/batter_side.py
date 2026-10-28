"""The RL batter inside the joint ONE OUTS scene (batter session, design approved by the planning session 2026-10-28).

BatterSide runs the trained batter's own env code unchanged (TeeBattingEnv -> TeeBattingMimicEnv -> PitchedBattingEnv:
perception with bias / noise / delay, mimic obs, ref clock + tempo, residual PD targets, the swept bat-ball impact,
the batted-ball flight and play_log) on the joint env's "batter" articulation. Only the framework is replaced:
_JointFramework sits between TeeBattingEnv and DirectRLEnv in the MRO and takes the joint env's sim / robot / device
instead of building its own, and the pitch-library part of the reset is replaced by the live hand-off.

Frames: the pitcher's world is fixed (mound along +x, pitch line at plate_yaw -18 deg), so the batter math runs in its
own frame B through RotatedArticulation: W = Rz(yaw) B + pivot, yaw = 162 deg (B: +x plate -> mound, +y third-base side).
Every robot.data field the batter env reads in world axes is rotated on read (positions, quaternions, linear / angular
velocities); body-frame fields (projected gravity, root velocities in the body frame), joint states and the bat
impulse (applied in the body frame) are invariant. Root-state writes are rotated back. The obs terms are all
root-frame differences, so with every world quantity (robot, ref root / key positions, ball, pitch, tee, target)
in B the obs equal the native env's up to float error (M1.5 exactness test). Per env, scene.env_origins is set so that the pitch-crossing plane
of the batter's clip (x = tee_x) is the front of the plate and the field is z = 0 (real heights: cfg.real_height; the
joint env uses only the clips whose contact height needs no re-centring, |tee_z + tee_offset_z - 0.80| <= 0.03).

Clocks (joint env hooks): control() at 100 Hz before the pitcher's action, ball_world(dt) every physics step (batter
PD targets, impact test, ball flight), post() after the step (dones, rewards -> play_log). Before its episode starts
(release_t_pred - the clip's lead), a batter holds the clip's first frame under PD; its episode then starts exactly as
in training (stance, ref clock at t0, the pitch released release_t later).
"""

from __future__ import annotations

import os

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product (w, x, y, z), written out directly: exact when either factor is the identity (isaaclab's
    quat_mul uses the 8-multiply form, which rounds even then - enough to change the contact-rich rollout)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

from aibaseball.physics.pitch_gen import PitchLibrary, fly_to_x
from aibaseball.tasks.tee_batting.pitched_env import PitchedBattingEnv

JOINT_CLIPS = ("obp_167-7", "obp_282-1", "obp_42-1", "obp_42-2")  # |dz| <= 0.03 m (one scene, real heights)


BATTER_YAW_DEG = 162.0
BLEND_S = 0.5  # after_done "blend": seconds from the done pose to the stance  # Rz(180 - plate_yaw 18): the batter frame B into the pitcher's world
TAKE_BLEND_S = 0.5  # take modes 5 / 6: stance -> default pose from the scheduled start
TAKE_BLEND_FAST_S = 0.25  # take mode 8
N_LEG_WAIST = 15  # body joints 0..14: legs (12) + waist (3); 15.. arms


class _RotatedData:
    """robot.data seen in frame B: x_B = R^T (x_W - pivot), q_B = q_R^-1 q_W, v_B = R^T v_W (per env)."""

    _POS = ("root_link_pos_w", "root_com_pos_w", "root_pos_w", "body_link_pos_w", "body_com_pos_w", "body_pos_w")
    _QUAT = ("root_link_quat_w", "root_com_quat_w", "root_quat_w", "body_link_quat_w", "body_com_quat_w", "body_quat_w")
    _VEC = ("root_lin_vel_w", "root_ang_vel_w", "root_com_lin_vel_w", "root_com_ang_vel_w", "root_link_lin_vel_w",
            "root_link_ang_vel_w", "body_lin_vel_w", "body_ang_vel_w", "body_com_lin_vel_w", "body_com_ang_vel_w",
            "body_link_lin_vel_w", "body_link_ang_vel_w")
    _STATE = ("root_state_w", "root_link_state_w", "root_com_state_w")

    def __init__(self, art):
        self._art = art

    def __getattr__(self, name):
        a, x = self._art, getattr(self._art.inner.data, name)
        if name in self._POS:
            return a.vec_to_b(x - (a.pivot if x.dim() == 2 else a.pivot.unsqueeze(1)))
        if name in self._QUAT:
            return a.quat_to_b(x)
        if name in self._VEC:
            return a.vec_to_b(x)
        if name in self._STATE:
            return torch.cat([a.vec_to_b(x[:, :3] - a.pivot), a.quat_to_b(x[:, 3:7]), a.vec_to_b(x[:, 7:10]),
                              a.vec_to_b(x[:, 10:13])], -1)
        if name.endswith("_w") and name not in ("projected_gravity_w",):
            raise AttributeError(f"RotatedArticulation: world-frame field {name} has no B-frame mapping yet")
        return x  # joint states, body-frame fields, limits, default_root_state (env-local, set in B): invariant


class RotatedArticulation:
    """The batter articulation in frame B (yaw about z + per-env pivot); everything not listed passes through."""

    def __init__(self, inner, pivot: torch.Tensor, yaw_deg: float = BATTER_YAW_DEG):
        self.inner, self.pivot = inner, pivot
        z = torch.zeros(pivot.shape[0], device=pivot.device)
        self.q_r = quat_from_euler_xyz(z, z, torch.full_like(z, yaw_deg * torch.pi / 180.0))
        self.data = _RotatedData(self)

    def _qr(self, x):  # one yaw for every env: also fits env subsets (x[ids])
        return self.q_r[0].expand(*x.shape[:-1], 4)

    def vec_to_b(self, v):
        return quat_rotate_inverse(self._qr(v).reshape(-1, 4), v.reshape(-1, 3)).reshape(v.shape)

    def vec_to_w(self, v):
        return quat_apply(self._qr(v).reshape(-1, 4), v.reshape(-1, 3)).reshape(v.shape)

    def quat_to_b(self, q):
        qr = self._qr(q).reshape(-1, 4)
        qr_inv = qr * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qr.device)
        return quat_mul(qr_inv, q.reshape(-1, 4)).reshape(q.shape)

    def pos_to_w(self, p):
        return self.vec_to_w(p) + self.pivot

    def write_root_state_to_sim(self, root, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        qr = self.q_r[ids]
        w = torch.cat([quat_apply(qr, root[:, :3]) + self.pivot[ids], quat_mul(qr, root[:, 3:7]),
                       quat_apply(qr, root[:, 7:10]), quat_apply(qr, root[:, 10:13])], -1)
        self.inner.write_root_state_to_sim(w, env_ids)

    def write_root_pose_to_sim(self, pose, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        qr = self.q_r[ids]
        self.inner.write_root_pose_to_sim(torch.cat([quat_apply(qr, pose[:, :3]) + self.pivot[ids],
                                                     quat_mul(qr, pose[:, 3:7])], -1), env_ids)

    def __getattr__(self, name):  # joint writes / targets, find_*, body-frame external forces, num_*: invariant
        return getattr(self.inner, name)


class _Null:
    """Stands in for the batter env's own ball / tee / markers (the joint env owns the visible ball)."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Scene:
    def __init__(self, env_origins):
        self.env_origins = env_origins
        self.num_envs = env_origins.shape[0]


class _JointFramework(DirectRLEnv):
    """The DirectRLEnv fields the batter env code reads, taken from the joint env (no sim of its own)."""

    def __init__(self, cfg, render_mode=None, **kwargs):  # noqa: D401 - replaces DirectRLEnv.__init__
        j = self._joint
        self.cfg, self.render_mode = cfg, None
        self.sim = j.sim  # device, num_envs, physics_dt, step_dt, max_episode_length: DirectRLEnv properties of these
        self.scene = _Scene(torch.zeros(j.num_envs, 3, device=j.device))  # set per clip in _place()
        self.robot = RotatedArticulation(j.batter, j.plate_front_w())
        self.ball, self.tee, self.ball_marker, self.landing_markers = _Null(), _Null(), _Null(), _Null()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros_like(self.reset_terminated)
        self.common_step_counter = 0
        self.extras = {}
        self._is_closed = True  # DirectRLEnv.close() / __del__: the sim belongs to the joint env, never close it here

    def _reset_idx(self, env_ids):
        self.robot.reset(env_ids)
        self.episode_length_buf[env_ids] = 0


class BatterSide(PitchedBattingEnv, _JointFramework):
    def __init__(self, joint, cfg, checkpoint: str, clip_of_env=None, after_done: str = "policy", own_timing=None,
                 timing_bias=None, shift_hist=True, no_handoff=False, swing=None):
        """joint: the JointPitchEnv; cfg: the batter's play cfg (PitchedBattingHumanPlayEnvCfg with pitch_release_csv =
        the P3c release table, effort via AIB_EFFORT_SCALE); clip_of_env: clip index per env (into JOINT_CLIPS order),
        default round-robin. after_done: what the batter does once its training episode would have ended (pitch
        passed, batted ball landed, fallen, time out): "policy" keeps the policy running (never trained there),
        "hold" freezes the PD targets at the joint pose of that moment, "blend" moves them from that pose to the
        clip's stance (frame 0) over BLEND_S (smoothstep, zero velocity targets: the PD damps the motion).
        own_timing: (n, 2) per env (hand-off time, contact-plane time) of the same seed's live pitch from a first pass
        (the pitch does not depend on the batter): two-pass oracle timing, see _own_clock. None = schedule every env on
        the table's median-flight row. timing_bias: cfg.timing_bias_s override (default the game value 0.055).
        shift_hist: at the hand-off, move the perceived-ball history (the static pre-release ball) by the live release
        minus the predicted one, as if the ball had waited at its own release point (native); without it the first
        delayed percepts after the release see a jump of a few cm (a one-step velocity / crossing spike). Default on
        (planning 10-28: joint seed 1 in-zone hits 0.24 -> 0.33, dz scatter 0.084 -> 0.061 m).
        no_handoff: control run - the live pitch never replaces the batter's own (the predicted median-flight row,
        released at its own release_t exactly as in the native env); the joint env's ball mirrors the batter's.
        swing: (n,) per env, 1 / True = swing, else a take (ONE OUTS PA, the engine's take decision made offline on the
        env's seed-invariant crossing): its episode runs as usual (clock, hand-off, contact check) but
        0 = the PD targets stay on the clip's stance (frame 0, no feed-forward) the whole time (20's M3 PA film 10-28:
          falls on its face ~1.6 s after the start - the stance is not balanced under PD alone);
        2 = the policy keeps running with its reference clock pinned to the stance (residual on, tempo ignored, no
          feed-forward): the policy's own balance feedback around the stance;
        3 = the PD targets go to the articulation's default pose (M1's standing batter);
        4 = diagnostic: the batter's episode never starts (the pre-start stance hold for the whole joint episode);
        5 = from the scheduled start, blend (smoothstep, TAKE_BLEND_S) the stance to the default pose;
        6 = as 5 for the legs + waist only, the arms (bat) stay in the stance;
        7 = as 6 from the joint episode's start (t 0) instead of the scheduled batter start;
        8 = as 6 with a TAKE_BLEND_FAST_S blend.
        Take diag 10-28 (64 envs, 16 per mode): the stance does not stand under PD alone - mode 4 falls 16/16 at 1.4-2.4 s,
        mode 0 16/16, mode 2 12/16 (and moves the bat ~8 m/s); the default pose (3) stands 16/16 but jumps there at the start.
        None = every env swings. take_dev_t / take_dev_joint: per take env the first time a body joint is 0.3 rad off
        its PD target, and which (fall diagnosis)."""
        assert after_done in ("policy", "hold", "blend"), after_done
        self.after_done = after_done
        assert os.environ.get("AIB_EFFORT_SCALE") == "1.5" or os.environ.get("AIB_EFFORT_PROBE") == "1",             "the delivery batter runs at AIB_EFFORT_SCALE 1.5 (AIB_EFFORT_PROBE=1: a zero-shot strength probe, never a delivery)"
        assert cfg.pitch_release_csv, "the predicted pitch comes from the P3c release table"
        self._joint = joint
        cfg.motion_files = [f for f in cfg.motion_files if os.path.splitext(os.path.basename(f))[0] in JOINT_CLIPS]
        assert len(cfg.motion_files) == len(JOINT_CLIPS), cfg.motion_files
        cfg.real_height = True
        cfg.timing_noise_s = 0.0  # the engine's true arrival (batter-oracle-timing); bias stays the game value
        cfg.timing_bias_s = 0.055 if timing_bias is None else timing_bias
        cfg.aim_offset_z = 0.0
        ctrl = cfg.sim.dt * cfg.decimation  # 100 Hz control on the joint env's physics step
        cfg.sim.dt = joint.physics_dt
        cfg.decimation = int(round(ctrl / cfg.sim.dt))
        assert abs(joint.step_dt * joint.cfg.batter_every - ctrl) < 1e-9, "control() must run at the batter's 100 Hz"
        n = joint.num_envs
        PitchedBattingEnv.__init__(self, cfg)  # TeeBattingEnv's super().__init__ lands in _JointFramework
        names = [os.path.splitext(os.path.basename(f))[0] for f in cfg.motion_files]
        order = torch.tensor([names.index(c) for c in JOINT_CLIPS], device=self.device)  # JOINT_CLIPS index -> MotionLib
        sel = torch.arange(n, device=self.device) % len(JOINT_CLIPS) if clip_of_env is None else \
            torch.as_tensor(clip_of_env, device=self.device)
        self.fixed_clip = order[sel]  # mimic_env's reset keeps these (fixed_clip)
        # the predicted pitch (before the hand-off): the P3c table's median-flight row, as one training pitch of it
        k = int(self.lib.flight_time.argsort()[len(self.lib.flight_time) // 2])
        self.lib = PitchLibrary(*(getattr(self.lib, f)[k:k + 1] for f in PitchLibrary.__dataclass_fields__))
        self.active = torch.zeros(n, dtype=torch.bool, device=self.device)  # its episode has started
        self.start_t = torch.zeros(n, device=self.device)  # joint episode time of the batter's episode start
        self.handed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)  # its training episode would have ended
        self.done_t = torch.full((n,), float("nan"), device=self.device)  # joint episode time of that
        self._load_policy(checkpoint)
        self.hold_q = torch.zeros(n, len(self.body_joint_ids), device=self.device)
        self.rec = None  # per-env swing timing / geometry (record_swing), off by default
        self._perc = None
        self.fallen = torch.zeros(n, dtype=torch.bool, device=self.device)  # _get_dones (post) sets it; drivers read it every step
        self.trace = None  # list: per control step, the policy's input / output and body joints (trace diff vs native)
        self.shift_hist = shift_hist
        self.no_handoff = no_handoff
        assert not (no_handoff and own_timing is not None), "no_handoff flies the predicted row on its own schedule"
        self.handoff_jump = torch.full((n,), float("nan"), device=self.device)
        self.own = None if own_timing is None else torch.as_tensor(own_timing, dtype=torch.float32, device=self.device)
        assert self.own is None or self.own.shape == (n, 2), None if self.own is None else self.own.shape
        self._obs = None
        mode = torch.ones(n, dtype=torch.long, device=self.device) if swing is None else             torch.as_tensor(swing, device=self.device).long()
        assert mode.shape == (n,) and bool(((mode >= 0) & (mode <= 8)).all()), mode
        self.take_dev_t = torch.full((n,), float("nan"), device=self.device)
        self.take_dev_joint = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.take_bat_vmax = torch.zeros(n, device=self.device)  # per env, max bat CM speed over the whole run (a take must not swing)
        self.take_mode = torch.where(mode == 1, torch.full_like(mode, -1), mode)  # -1 = swings
        self.swing = mode == 1

    # ------------------------------------------------------------------ frames
    def _place(self):
        """Per env, the batter-frame origin (scene.env_origins, in B) for its clip. The pivot (B = 0) is the joint env's
        judged plate plane at field level, plate frame (plate_distance, zone_center_y, field_z): the plane the pitcher's
        tiers and j_cross use and the one build_library_from_release crosses (library frame x = 0 there, y = -y_plate).
        So library point (0, y_L, z) = B (0, y_L + zone_center_y, z), and the clip's tee is at the library's (0, 0) plus
        the pitch offset: env_origins = (-tee_x, zone_center_y - tee_y, 0)."""
        tee = self.ref.tee_pos[self.fixed_clip]
        zc = float(self._joint.cfg.zone_center_y)
        self.scene.env_origins = torch.stack([-tee[:, 0], zc - tee[:, 1], torch.zeros_like(tee[:, 0])], -1)

    # ------------------------------------------------------------------ policy
    def _load_policy(self, path: str):
        from rsl_rl.modules import ActorCritic, EmpiricalNormalization

        ck = torch.load(path, map_location=self.device, weights_only=False)
        sd = ck["model_state_dict"]
        n_obs, n_act = sd["actor.0.weight"].shape[1], sd["actor.6.weight"].shape[0]
        n_critic = sd["critic.0.weight"].shape[1]
        ac = ActorCritic(n_obs, n_critic, n_act, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                         activation="elu", init_noise_std=0.3).to(self.device)
        ac.load_state_dict(sd)
        ac.eval()
        norm = EmpiricalNormalization(shape=[n_obs], until=1.0e8).to(self.device)
        norm.load_state_dict(ck["obs_norm_state_dict"])
        norm.eval()
        self._actor, self._norm = ac, norm

    # ------------------------------------------------------------------ joint env hooks
    def _joint_time(self) -> torch.Tensor:
        j = self._joint
        return j.episode_length_buf.float() * j.step_dt

    def reset(self, env_ids):
        """Joint env reset (first after load): the clip's stance (the batter env's own reset, through the proxy) and the
        start time of the batter's episode, so that its pitch release (release_t of the predicted pitch) falls on the
        pitcher's predicted release. Until then the batter holds the stance under PD."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._place()
        PitchedBattingEnv._reset_idx(self, ids)
        self.active[ids] = False
        self.handed[ids] = False
        self.done[ids] = False
        self.done_t[ids] = float("nan")
        self.start_t[ids] = self._joint.release_t_pred[ids] - self.release_t[ids]
        if self.own is not None:  # the native release of this env's own pitch: start = hand-off - release
            h, tp = self.own[ids, 0], self.own[ids, 1]
            rel = (self.ref.contact_time[self.clip[ids]] - self.t0[ids]) / self.speed - (tp - h) + self.cfg.timing_bias_s
            self.start_t[ids] = h - torch.maximum(rel, torch.full_like(rel, self.cfg.min_lead_s))
        self._obs = None

    def _own_clock(self, ids):
        """Two-pass oracle timing (planning session 10-28, the ONE OUTS metric's assumption): at the episode start the
        batter env's pitch clock is the native one of this env's own pitch (pitched_env._reset_idx with its flight
        time tp - h, jitter 0, the game bias): released at the hand-off h, the reference contact timing_bias_s before
        the ball reaches the contact plane at tp. The start falls on a 100 Hz control step up to 10 ms after start_t;
        the reference clock absorbs that (ts - start_t) * speed, so the ball-vs-reference offset stays exact."""
        h, tp = self.own[ids, 0], self.own[ids, 1]
        ts = self._joint_time()[ids]
        self.release_t[ids] = h - ts
        self.flight_time[ids] = tp - h  # until the hand-off replaces it with the live fly_to_x time
        self.ref_t[ids] = self.ref.contact_time[self.clip[ids]] - (tp - self.cfg.timing_bias_s - ts) * self.speed

    def control(self):
        start = ~self.active & (self._joint_time() >= self.start_t - 1e-6) & (self.take_mode != 4)
        if start.any():  # the training episode starts now: the stance again (it was held there), ref clock, phys_t 0
            ids = start.nonzero(as_tuple=False).squeeze(-1)
            PitchedBattingEnv._reset_idx(self, ids)
            if self.own is not None:
                self._own_clock(ids)
            self.active[ids] = True
            self._obs = None
        if self._obs is None:
            self._obs = self._get_observations()["policy"]
        with torch.inference_mode():
            act = self._actor.act_inference(self._norm(self._obs))
        if self.trace is not None:
            self.trace.append(dict(active=self.active.clone(), k=self.episode_length_buf.clone(), obs=self._obs.clone(),
                                   act=act.clone(), q=self.robot.data.joint_pos[:, self.body_joint_ids].clone()))
        self._pre_physics_step(act)
        if self.rec is not None:  # mean tempo action (the clipped one the clock used): before the hand-off / from it to done (diag_bat_contact's window)
            r, tempo = self.rec, self.actions[:, -1].clamp(-1.0, 1.0)
            live = self.active & ~self.done
            for key, m in (("pre", live & ~self.handed), ("fly", live & self.handed)):
                r["tempo_" + key] += torch.where(m, tempo, torch.zeros_like(tempo))
                r["n_" + key] += m.float()
        hold = ~self.active
        if hold.any():  # waiting: frame 0 of the clip, no feed-forward
            pose = self._ref(torch.zeros(self.num_envs, device=self.device))["joint_pos"]
            tgt = self.joint_targets[:, self.body_joint_ids]
            self.joint_targets[:, self.body_joint_ids] = torch.where(hold.unsqueeze(-1), pose, tgt)
            self.joint_vel_targets[hold] = 0.0
        frz = self.done if self.after_done != "policy" else torch.zeros_like(self.done)
        if frz.any():  # past the training episode: hold the pose it ended in (or blend it to the stance)
            q = self.hold_q
            if self.after_done == "blend":
                s = ((self._joint_time() - self.done_t) / BLEND_S).clamp(0.0, 1.0).nan_to_num(0.0)
                s = (s * s * (3.0 - 2.0 * s)).unsqueeze(-1)
                stance = self._ref(torch.zeros(self.num_envs, device=self.device))["joint_pos"]
                q = self.hold_q + s * (stance - self.hold_q)
            tgt = self.joint_targets[:, self.body_joint_ids]
            self.joint_targets[:, self.body_joint_ids] = torch.where(frz.unsqueeze(-1), q, tgt)
            self.joint_vel_targets[frz] = 0.0
        take = ~self.swing
        if take.any():  # a take (see __init__ swing): stance / policy around the stance / default pose
            z = torch.zeros(self.num_envs, device=self.device)
            pose = self._ref(z)["joint_pos"]
            pin = self.take_mode == 2
            if pin.any():  # the clock stays on frame 0 (as cfg.take_hold), the policy's residual stays on
                self.ref_t = torch.where(pin, torch.minimum(self.ref_t, z), self.ref_t)
                res = self.actions[:, :-1] * self.action_scale * self.cfg.residual_scale
                pose = torch.where(pin.unsqueeze(-1), torch.clamp(pose + res, self.q_lo, self.q_hi), pose)
            dflt = self.take_mode == 3
            dpose = self.robot.data.default_joint_pos[:, self.body_joint_ids]
            if dflt.any():
                pose = torch.where(dflt.unsqueeze(-1), dpose, pose)
            bl = self.take_mode >= 5
            if bl.any():  # stance -> default pose from the scheduled start (legs + waist only for 6..8)
                t0 = torch.where(self.take_mode == 7, torch.zeros_like(self.start_t), self.start_t)
                dur = torch.where(self.take_mode == 8, torch.full_like(t0, TAKE_BLEND_FAST_S), torch.full_like(t0, TAKE_BLEND_S))
                s = ((self._joint_time() - t0) / dur).clamp(0.0, 1.0)
                s = (s * s * (3.0 - 2.0 * s)).unsqueeze(-1).expand_as(pose).clone()
                s[:, N_LEG_WAIST:] = torch.where((self.take_mode >= 6).unsqueeze(-1), 0.0, s[:, N_LEG_WAIST:])
                pose = torch.where(bl.unsqueeze(-1), pose + s * (dpose - pose), pose)
            tgt = self.joint_targets[:, self.body_joint_ids]
            self.joint_targets[:, self.body_joint_ids] = torch.where(take.unsqueeze(-1), pose, tgt)
            self.joint_vel_targets[take] = 0.0

    def ball_world(self, dt: float):
        if self.no_handoff:  # the batter's own pitch is "handed" once it leaves the hand (the swing record needs it)
            self.handed |= self.active & (self.phys_t >= self.release_t)
        else:
            new = self._joint.j_fly & ~self.handed
            if new.any():
                self._hand_off(new.nonzero(as_tuple=False).squeeze(-1))
        prev_ball = self.pitch_pos.clone() if self.rec is not None else None
        self._apply_action()
        if self.rec is not None:
            self._record(prev_ball, dt)
        h = self.hit.unsqueeze(-1)
        p = torch.where(h, self.ball_pos, self.pitch_pos)
        v = torch.where(h, self.ball_vel, self.pitch_vel)
        w = torch.where(h, self.ball_omega, self.pitch_omega)
        return self.robot.pos_to_w(p), self.robot.vec_to_w(v), self.robot.vec_to_w(w)

    def post(self):
        self.episode_length_buf[self.active] += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self._get_rewards()  # hit records -> play_log, fall flags
        new = self.active & ~self.done & (self.reset_terminated | self.reset_time_outs)
        if new.any():
            self.hold_q[new] = self.robot.data.joint_pos[new][:, self.body_joint_ids]
            self.done_t[new] = self._joint_time()[new]
            self.done |= new
        if self.trace:  # the targets this control step wrote (all dofs), as diag_bat_contact --trace
            self.trace[-1]["tp"] = self.robot.data.joint_pos_target.clone()
            self.trace[-1]["tv"] = self.robot.data.joint_vel_target.clone()
        self._obs = self._get_observations()["policy"]
        win = ~self.done & ~self.fallen & (self.take_mode != 4)  # while the pitch is live, before any fall (a falling bat whips)
        win |= (self.take_mode == 4) & ~self.fallen & (self._joint_time() < self._joint.release_t_pred + 1.0)
        v = self.robot.data.body_com_lin_vel_w[:, self.bat_id].norm(dim=-1)
        self.take_bat_vmax = torch.where(win, torch.maximum(self.take_bat_vmax, v), self.take_bat_vmax)
        new = ~self.swing & torch.isnan(self.take_dev_t)
        if new.any():  # take fall diagnosis: the first body joint 0.3 rad off its PD target
            err = (self.robot.data.joint_pos[:, self.body_joint_ids] - self.joint_targets[:, self.body_joint_ids]).abs()
            m, j = err.max(-1)
            hit = new & (m > 0.3)
            self.take_dev_t = torch.where(hit, self._joint_time(), self.take_dev_t)
            self.take_dev_joint = torch.where(hit, j, self.take_dev_joint)
        if self.rec is not None and self._perc is not None:  # the ball the policy was given (B, z above the field)
            self.rec["perc"].append((self._joint_time().clone(), self._perc[:, 2].clone()))

    def _perceived_ball(self) -> torch.Tensor:
        self._perc = super()._perceived_ball()
        return self._perc

    # ------------------------------------------------------------------ swing timing / geometry (diag_bat_contact's)
    def record_swing(self):
        """Per env, the native contact diag's quantities at physics-step resolution (joint time): t_plane = the pitch
        crosses the clip's contact plane (x = contact_x_ref, interpolated), ref_err = ref clock minus the clip's contact
        time then (+ = the swing is early), t_square = the bat most square to the pitch line while the sweet spot is
        fast, gap / t_gap / dz / dperp = the closest ball-to-bat-surface approach before contact."""
        n, dev = self.num_envs, self.device
        nan = lambda: torch.full((n,), float("nan"), device=dev)  # noqa: E731
        self.rec = dict(t_plane=nan(), ref_err=nan(), rate_plane=nan(), t_square=nan(), sq_ax=torch.ones(n, device=dev), gap=torch.full((n,), 9.0, device=dev),
                        t_gap=nan(), dz=nan(), dperp=nan(), ss_speed_max=torch.zeros(n, device=dev), k=0,
                        tempo_pre=torch.zeros(n, device=dev), n_pre=torch.zeros(n, device=dev),
                        tempo_fly=torch.zeros(n, device=dev), n_fly=torch.zeros(n, device=dev),
                        t_commit=nan(), perc=[])  # t_commit: sweet spot first > 10 m/s

    def _record(self, prev_ball, dt):
        r = self.rec
        r["k"] += 1
        m = int(round(self._joint.step_dt / dt))  # physics steps per joint step; its clock advances after them
        t = self._joint_time() + ((r["k"] - 1) % m + 1) * dt  # end of this physics step
        live = self.active & self.handed & ~self.done
        knob, axis, cm, v_cm, w, _ = self._bat_state()
        b = self.pitch_pos
        x = ((b - knob) * axis).sum(-1).clamp(0.0, self.bat_geom.length)
        p = knob + axis * x.unsqueeze(-1)
        off = b - p
        gap = off.norm(dim=-1) - self.bat_geom.radius_at(x) - self.phys.ball.radius
        better = live & ~self.hit & (gap < r["gap"])
        _, v_ss = self._sweet_spot(knob, axis, cm, v_cm, w)
        dp = v_ss - (v_ss * axis).sum(-1, keepdim=True) * axis
        dp[:, 2] = 0.0
        dp = dp / dp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        r["gap"] = torch.where(better, gap, r["gap"])
        r["t_gap"] = torch.where(better, t, r["t_gap"])
        r["dz"] = torch.where(better, off[:, 2], r["dz"])
        r["dperp"] = torch.where(better, (off * dp).sum(-1), r["dperp"])
        xr = self.contact_x_ref
        cross = live & (prev_ball[:, 0] >= xr) & (b[:, 0] < xr) & torch.isnan(r["t_plane"])
        frac = (prev_ball[:, 0] - xr) / (prev_ball[:, 0] - b[:, 0]).clamp_min(1e-9)
        r["t_plane"] = torch.where(cross, t - dt + dt * frac, r["t_plane"])
        r["ref_err"] = torch.where(cross, self.ref_t - self.ref.contact_time[self.clip], r["ref_err"])
        r["rate_plane"] = torch.where(cross, self.rate, r["rate_plane"])  # the clock rate there (delta_s), not the run's last
        sp = v_ss.norm(dim=-1)
        r["ss_speed_max"] = torch.where(live, torch.maximum(r["ss_speed_max"], sp), r["ss_speed_max"])
        r["t_commit"] = torch.where(live & (sp > 10.0) & torch.isnan(r["t_commit"]), t, r["t_commit"])
        sq = live & (sp > 10.0) & (axis[:, 0].abs() < r["sq_ax"])
        r["sq_ax"] = torch.where(sq, axis[:, 0].abs(), r["sq_ax"])
        r["t_square"] = torch.where(sq, t, r["t_square"])

    # ------------------------------------------------------------------ the live pitch replaces the predicted one
    def _hand_off(self, ids):
        """At the joint env's hand-off (this physics step, before the flight step): the batter's pitch becomes the live
        release state in B, released now. Target, crossing and flight time follow from it (drag + Magnus, fly_to_x)."""
        j, a = self._joint, self.robot
        if (~self.active[ids]).any():
            print(f"[batter_side] {int((~self.active[ids]).sum())} envs released before their episode start", flush=True)
        pos = a.vec_to_b(j.j_pos[ids] - a.pivot[ids])
        vel, omg = a.vec_to_b(j.j_vel[ids]), a.vec_to_b(j.j_omega[ids])
        x_c = self.contact_x_ref[ids]
        p_hit, t_hit = torch.empty_like(pos), torch.empty_like(x_c)
        for x in x_c.unique():  # fly_to_x takes one plane (the clips here all have contact_x_ref 0)
            m = x_c == x
            p_hit[m], _, t_hit[m] = fly_to_x(pos[m], vel[m], omg[m], self.phys, float(x))
        jump = pos - self.pitch_pos[ids]  # live release minus the predicted (static) pre-release ball
        self.handoff_jump[ids] = jump.norm(dim=-1)
        if self.shift_hist:
            self.ball_obs_hist[ids] += jump.unsqueeze(1)
        self.pitch_pos[ids], self.pitch_prev[ids] = pos, pos
        self.pitch_vel[ids], self.pitch_omega[ids] = vel, omg
        self.release_t[ids] = self.phys_t[ids]  # released now: _advance_pitch flies it from this step
        self.flight_time[ids] = t_hit
        self.tee_pos[ids] = p_hit
        self.pitch_plate[ids] = torch.stack([p_hit[:, 1] - float(j.cfg.zone_center_y), p_hit[:, 2]], -1)
        self.pitch_kmh[ids] = vel.norm(dim=-1) * 3.6
        self.handed[ids] = True
