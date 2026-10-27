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
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, quat_from_euler_xyz, quat_mul

from aibaseball.physics.pitch_gen import fly_to_x
from aibaseball.tasks.tee_batting.pitched_env import PitchedBattingEnv
from aibaseball.tasks.tee_batting.mimic_env import TeeBattingMimicEnv

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

    def _qr(self, x):
        return self.q_r if x.dim() == 2 else self.q_r.unsqueeze(1).expand(*x.shape[:-1], 4)

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


class _JointFramework(DirectRLEnv):
    """The DirectRLEnv fields the batter env code reads, taken from the joint env (no sim of its own)."""

    def __init__(self, cfg, render_mode=None, **kwargs):  # noqa: D401 - replaces DirectRLEnv.__init__
        j = self._joint
        self.cfg, self.render_mode = cfg, None
        self.sim = j.sim
        self.device, self.num_envs = j.device, j.num_envs
        self.physics_dt = cfg.sim.dt
        self.step_dt = cfg.sim.dt * cfg.decimation
        self.max_episode_length_s = cfg.episode_length_s
        self.max_episode_length = int(round(cfg.episode_length_s / self.step_dt))
        self.robot = RotatedArticulation(j.batter, j.plate_front_w())
        self.ball, self.tee, self.ball_marker, self.landing_markers = _Null(), _Null(), _Null(), _Null()
        self.scene = _Scene(torch.zeros(self.num_envs, 3, device=self.device))  # set per clip in _place()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros_like(self.reset_terminated)
        self.common_step_counter = 0
        self.extras = {}

    def _reset_idx(self, env_ids):
        self.robot.reset(env_ids)
        self.episode_length_buf[env_ids] = 0


class BatterSide(PitchedBattingEnv, _JointFramework):
    def __init__(self, joint, cfg, checkpoint: str, clip_of_env=None):
        """joint: the JointPitchEnv; cfg: the batter's play cfg (PitchedBattingHumanPlayEnvCfg, effort via
        AIB_EFFORT_SCALE); clip_of_env: clip index per env (into cfg.motion_files), default round-robin."""
        assert os.environ.get("AIB_EFFORT_SCALE") == "1.5", "the delivery batter runs at AIB_EFFORT_SCALE 1.5"
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
        n = joint.num_envs
        self._clip_of_env = (torch.arange(n) % len(JOINT_CLIPS)) if clip_of_env is None else torch.as_tensor(clip_of_env)
        PitchedBattingEnv.__init__(self, cfg)  # TeeBattingEnv's super().__init__ lands in _JointFramework
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
        tee = self.ref.tee_pos[self.clip]
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
    def reset(self, env_ids):
        """Joint env reset (once per process): clips, placement, the hold pose; the batter episode starts later."""
        ids = torch.as_tensor(env_ids, device=self.device)
        self.clip[ids] = self._clip_of_env.to(self.device)[ids]
        self._place()
        self.active[ids] = False
        self.handed[ids] = False
        lead = (self.ref.contact_time[self.clip[ids]] - self.t0[ids]) / self.speed - self._flight_pred() + self.cfg.timing_bias_s
        self.start_t[ids] = (self._joint.release_t_pred[ids] - lead.clamp_min(self.cfg.min_lead_s))

    def control(self):
        j = self._joint
        start = ~self.active & (j.episode_time() >= self.start_t)
        if start.any():
            ids = start.nonzero(as_tuple=False).squeeze(-1)
            self._reset_idx(ids)  # the training reset (stance, ref clock) + the predicted pitch (below)
            self.active[ids] = True
            self._obs = None
        if self._obs is None:
            self._obs = self._get_observations()["policy"]
        with torch.inference_mode():
            act = self._actor.act_inference(self._norm(self._obs))
        self._pre_physics_step(act)

    def ball_world(self, dt: float):
        j = self._joint
        new = j.j_fly & ~self.handed & self.active
        if new.any():
            self._hand_off(new.nonzero(as_tuple=False).squeeze(-1))
        hold = ~self.active
        if hold.any():  # not started: the clip's first frame under PD (what its reset will write)
            self.joint_targets[hold] = self._hold_pose()[hold]
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

    # ------------------------------------------------------------------ pitch (replaces the library part of the reset)
    def _flight_pred(self):
        return torch.full((self.num_envs,), self._joint.cfg.flight_t_pred_s, device=self.device)[: 1].expand(1)

    def _hold_pose(self):
        return self.robot.data.default_joint_pos

    def _hand_off(self, ids):
        raise NotImplementedError("filled in once the joint env's frame helpers are fixed (batter session)")

    def _reset_idx(self, env_ids):
        TeeBattingMimicEnv._reset_idx(self, env_ids)  # robot to the clip's frame 0, ref clock, perception buffers
        raise NotImplementedError("the predicted-pitch part follows pitched_env._reset_idx lines 238-272")
