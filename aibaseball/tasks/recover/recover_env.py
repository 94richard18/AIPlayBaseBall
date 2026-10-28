"""Human25-R: the post-contact recovery policy's env (see RecoverEnvCfg). Episodes reset to a bank state (the frozen
swing at BatterSide's hand-off trigger) and run 2 s at 100 Hz; the ball and tee stay parked underground.

obs (proprioception only, all body-frame or joint-space, so frame B of the joint scene gives the same numbers):
joint pos - default, joint vel x 0.05, projected gravity, root lin vel x 0.5, root ang vel x 0.25 (body frame), pelvis
height above the field - 0.94, anchor - joint pos, previous action, time since the hand-off / 2 s.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from aibaseball.tasks.tee_batting.tee_batting_env import TeeBattingEnv

from .recover_env_cfg import RecoverEnvCfg


def recover_obs(env, anchor: torch.Tensor, prev_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The recovery policy's input; env = a TeeBattingEnv-like object (RecoverEnv, or BatterSide in the joint scene)."""
    d = env.robot.data
    ids = env.body_joint_ids
    q = d.joint_pos[:, ids]
    pelvis_z = d.body_link_pos_w[:, env.pelvis_id, 2] - env.scene.env_origins[:, 2]
    obs = torch.cat([
        q - d.default_joint_pos[:, ids],
        d.joint_vel[:, ids] * 0.05,
        d.projected_gravity_b,
        d.root_lin_vel_b * 0.5,
        d.root_ang_vel_b * 0.25,
        (pelvis_z - 0.94).unsqueeze(-1),
        anchor - q,
        prev_action,
        (t / 2.0).unsqueeze(-1),
    ], -1)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def recover_anchor(q0: torch.Tensor, stance: torch.Tensor, t: torch.Tensor, blend_s: float) -> torch.Tensor:
    """PD anchor: smoothstep from the hand-off targets q0 to the stance over blend_s (t = time since the hand-off)."""
    s = (t / blend_s).clamp(0.0, 1.0)
    s = (s * s * (3.0 - 2.0 * s)).unsqueeze(-1)
    return q0 + s * (stance - q0)


class RecoverEnv(TeeBattingEnv):
    cfg: RecoverEnvCfg

    def __init__(self, cfg: RecoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        b = np.load(cfg.bank_path)
        assert list(b["joint_names"]) == list(self.robot.joint_names), "bank joint order differs from the robot's"
        self.bank = {k: torch.as_tensor(b[k], device=dev).float() for k in ("root", "jp", "jv", "tgt")}
        self.bank["act"] = torch.as_tensor(b["act"][:, :len(self.body_joint_ids)], device=dev).float()
        self.bank_hit = torch.as_tensor(b["hit"], device=dev)
        self.n_bank = self.bank["root"].shape[0]
        self.idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.q0 = torch.zeros(n, len(self.body_joint_ids), device=dev)
        self.anchor = torch.zeros_like(self.q0)
        self.t = torch.zeros(n, device=dev)
        self.on_ground = torch.zeros(n, dtype=torch.bool, device=dev)
        self.min_pelvis_z = torch.full((n,), 9.0, device=dev)
        self.foot_ids = [self.robot.find_bodies(f"{s}_foot")[0][0] for s in "lr"]
        self.foot_zmin = torch.full((n, 2), 9.0, device=dev)
        self.body_mass = self.robot.root_physx_view.get_masses().to(dev)
        self.stand_z = float(self.robot.data.default_root_state[0, 2])
        self.next_eval = 0  # eval: bank states in order, one per env
        self.rstats = {k: torch.zeros((), device=dev) for k in
                       ("ground_rate", "ground_rate_hit", "ground_rate_miss", "ep_len_s", "height", "tilt_deg", "still")}

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        stance = self.robot.data.default_joint_pos[:, self.body_joint_ids]
        self.anchor = recover_anchor(self.q0, stance, self.t + self.step_dt, self.cfg.anchor_blend_s)
        tgt = self.anchor + self.actions * self.action_scale * self.cfg.residual_scale
        self.joint_targets[:, self.body_joint_ids] = torch.clamp(tgt, self.q_lo, self.q_hi)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_targets)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.joint_targets))

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        return {"policy": recover_obs(self, self.anchor, self.actions, self.t)}

    # ------------------------------------------------------------------ dones
    def _get_dones(self):
        self.t = self.episode_length_buf.float() * self.step_dt
        d = self.robot.data
        pz = d.body_link_pos_w[:, self.pelvis_id, 2] - self.scene.env_origins[:, 2]
        self.min_pelvis_z = torch.minimum(self.min_pelvis_z, pz)
        tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
        self.on_ground = (pz < self.cfg.ground_pelvis_z) | (tilt > self.cfg.max_tilt_deg)
        self.pz, self.tilt = pz, tilt
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        term = self.on_ground if self.cfg.terminate else torch.zeros_like(self.on_ground)
        return term, time_out

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        c, d = self.cfg, self.robot.data
        tilt = torch.deg2rad(self.tilt)
        height = torch.exp(-((self.pz - self.stand_z) / c.sigma_height) ** 2)
        upright = torch.exp(-(tilt / c.sigma_tilt) ** 2)
        jv_rms = d.joint_vel[:, self.body_joint_ids].pow(2).mean(-1).sqrt()
        still = torch.exp(-(d.root_lin_vel_w.norm(dim=-1) / c.sigma_lin) ** 2 - (d.root_ang_vel_w.norm(dim=-1) / c.sigma_ang) ** 2
                          - (jv_rms / c.sigma_jv) ** 2) * (self.t >= c.still_after_s).float()
        feet = d.body_link_pos_w[:, self.foot_ids]
        self.foot_zmin = torch.minimum(self.foot_zmin, feet[..., 2])
        lift = (feet[..., 2] - self.foot_zmin).clamp_min(0.0)
        feet_r = torch.exp(-(lift / c.sigma_foot) ** 2).prod(-1)
        w = self.body_mass.unsqueeze(-1)
        com = ((d.body_com_pos_w * w).sum(1) / w.sum(1))[:, :2]
        a, b = feet[:, 0, :2], feet[:, 1, :2]
        ab = b - a
        s = (((com - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-6)).clamp(0.0, 1.0)
        com_r = torch.exp(-((com - (a + s.unsqueeze(-1) * ab)).norm(dim=-1) / c.sigma_com) ** 2)
        rate = ((self.actions - self.prev_actions) ** 2).sum(-1)
        ground = self.on_ground & bool(c.terminate)
        rew = (c.w_alive + c.w_height * height + c.w_upright * upright + c.w_still * still + c.w_feet * feet_r
               + c.w_com * com_r - c.w_action_rate * rate - c.w_ground * ground.float())
        done = self.reset_terminated | self.reset_time_outs
        if done.any():
            g = self.on_ground[done].float()
            h = self.bank_hit[self.idx[done]]
            a_ = 0.02
            self.rstats["ground_rate"] = (1 - a_) * self.rstats["ground_rate"] + a_ * g.mean()
            if h.any():
                self.rstats["ground_rate_hit"] = (1 - a_) * self.rstats["ground_rate_hit"] + a_ * g[h].mean()
            if (~h).any():
                self.rstats["ground_rate_miss"] = (1 - a_) * self.rstats["ground_rate_miss"] + a_ * g[~h].mean()
            self.rstats["ep_len_s"] = (1 - a_) * self.rstats["ep_len_s"] + a_ * self.t[done].mean()
        a_ = 0.01
        for k, v in (("height", self.pz.mean()), ("tilt_deg", self.tilt.mean()), ("still", still.mean())):
            self.rstats[k] = (1 - a_) * self.rstats[k] + a_ * v
        self.extras["log"] = {f"recover/{k}": v.clone() for k, v in self.rstats.items()}
        return rew

    # ------------------------------------------------------------------ reset: a bank state
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)  # DirectRLEnv bookkeeping + the base reset (overwritten below)
        dev, c = self.device, self.cfg
        ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        n = len(ids)
        if c.terminate:
            k = torch.randint(0, self.n_bank, (n,), device=dev)
        else:  # eval: the bank in order (env i <- state i, wrapping)
            k = (torch.arange(n, device=dev) + self.next_eval) % self.n_bank
            self.next_eval += n
        self.idx[ids] = k
        root = self.bank["root"][k].clone()
        root[:, :3] += self.scene.env_origins[ids]
        root[:, 7:10] += (torch.rand(n, 3, device=dev) * 2 - 1) * c.bank_root_vel_noise
        self.robot.write_root_state_to_sim(root, ids)
        jv = self.bank["jv"][k] * (1.0 + (torch.rand_like(self.bank["jv"][k]) * 2 - 1) * c.bank_vel_noise)
        self.robot.write_joint_state_to_sim(self.bank["jp"][k], jv, env_ids=ids)
        self.joint_targets[ids] = self.bank["tgt"][k]
        self.q0[ids] = self.bank["tgt"][k][:, self.body_joint_ids]
        self.anchor[ids] = self.q0[ids]
        self.actions[ids] = self.bank["act"][k]  # the swing policy's last body action seeds "previous action"
        self.prev_actions[ids] = self.actions[ids]
        self.t[ids] = 0.0
        self.min_pelvis_z[ids] = 9.0
        self.foot_zmin[ids] = 9.0
        park = self.scene.env_origins[ids].clone()
        park[:, 2] = -5.0
        unit_q = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
        self.ball.write_root_pose_to_sim(torch.cat([park, unit_q], -1), ids)
        park[:, 2] = -8.0
        self.tee.write_root_pose_to_sim(torch.cat([park, unit_q], -1), ids)
