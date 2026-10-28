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


BATTER_YAW_DEG = 162.0  # Rz(180 - plate_yaw 18): the batter frame B into the pitcher's world


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
    def __init__(self, joint, cfg, checkpoint: str, clip_of_env=None):
        """joint: the JointPitchEnv; cfg: the batter's play cfg (PitchedBattingHumanPlayEnvCfg with pitch_release_csv =
        the P3c release table, effort via AIB_EFFORT_SCALE); clip_of_env: clip index per env (into JOINT_CLIPS order),
        default round-robin."""
        assert os.environ.get("AIB_EFFORT_SCALE") == "1.5", "the delivery batter runs at AIB_EFFORT_SCALE 1.5"
        assert cfg.pitch_release_csv, "the predicted pitch comes from the P3c release table"
        self._joint = joint
        cfg.motion_files = [f for f in cfg.motion_files if os.path.splitext(os.path.basename(f))[0] in JOINT_CLIPS]
        assert len(cfg.motion_files) == len(JOINT_CLIPS), cfg.motion_files
        cfg.real_height = True
        cfg.timing_noise_s = 0.0  # the engine's true arrival (batter-oracle-timing); bias stays the game value
        cfg.timing_bias_s = 0.055
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
        self._load_policy(checkpoint)
        self._obs = None

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
        self.start_t[ids] = self._joint.release_t_pred[ids] - self.release_t[ids]
        self._obs = None

    def control(self):
        start = ~self.active & (self._joint_time() >= self.start_t - 1e-6)
        if start.any():  # the training episode starts now: the stance again (it was held there), ref clock, phys_t 0
            ids = start.nonzero(as_tuple=False).squeeze(-1)
            PitchedBattingEnv._reset_idx(self, ids)
            self.active[ids] = True
            self._obs = None
        if self._obs is None:
            self._obs = self._get_observations()["policy"]
        with torch.inference_mode():
            act = self._actor.act_inference(self._norm(self._obs))
        self._pre_physics_step(act)
        hold = ~self.active
        if hold.any():  # waiting: frame 0 of the clip, no feed-forward
            pose = self._ref(torch.zeros(self.num_envs, device=self.device))["joint_pos"]
            tgt = self.joint_targets[:, self.body_joint_ids]
            self.joint_targets[:, self.body_joint_ids] = torch.where(hold.unsqueeze(-1), pose, tgt)
            self.joint_vel_targets[hold] = 0.0

    def ball_world(self, dt: float):
        new = self._joint.j_fly & ~self.handed
        if new.any():
            self._hand_off(new.nonzero(as_tuple=False).squeeze(-1))
        self._apply_action()
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
        self._obs = self._get_observations()["policy"]

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
        self.pitch_pos[ids], self.pitch_prev[ids] = pos, pos
        self.pitch_vel[ids], self.pitch_omega[ids] = vel, omg
        self.release_t[ids] = self.phys_t[ids]  # released now: _advance_pitch flies it from this step
        self.flight_time[ids] = t_hit
        self.tee_pos[ids] = p_hit
        self.pitch_plate[ids] = torch.stack([p_hit[:, 1] - float(j.cfg.zone_center_y), p_hit[:, 2]], -1)
        self.pitch_kmh[ids] = vel.norm(dim=-1) * 3.6
        self.handed[ids] = True
