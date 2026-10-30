"""Human30 (planning 14, 10-30 backup): the swing policy trained WITH the scripted arm aim in the loop.

PitchedBattingEnv + the row-9 scripted aim of ArmAimEnv (scripted mode, aim_hold off): after the swing policy's targets are
written, offset = bound x clamp(gain(side) x |dy| / bound, +-1) on the aim joints, dy = the percept's predicted crossing y -
the nominal target y, from saim_window_s (reference clock) before the clip's contact until the hit / the ball past the
plane, then decaying with tau. The swing policy observes q - offset (and, with saim_comp_full, the bat moved back by the
Jacobian-mapped offset), as in the game (BatterSide saim) - so fine-tuning lets the swing adapt to the layer instead of
the layer being bolted onto a swing that never saw it (h29s: hit 0.908 -> 0.700, launch 19.6 -> 5.9 deg).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch

from .pitched_env import PitchedBattingEnv


class SaimSwingEnv(PitchedBattingEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        g = json.load(open(cfg.saim_gain))
        assert g["joints"] == list(cfg.aim_joints), (g["joints"], cfg.aim_joints)
        names = list(self.body_joint_names)
        self.aim_cols = torch.tensor([names.index(j) for j in cfg.aim_joints], device=dev)
        self.aim_ids = self.body_joint_ids[self.aim_cols]
        self._gp = torch.tensor(g["gain_plus"], device=dev)
        self._gm = torch.tensor(g["gain_minus"], device=dev)
        self.res_off = torch.zeros(n, len(cfg.aim_joints), device=dev)  # rad, applied offset
        self.res_closed = torch.zeros(n, dtype=torch.bool, device=dev)  # window over (hit / ball past the plane)
        self._ball_seen = None
        self._bat_comp = None
        self.track_stats.update({k: torch.zeros((), device=dev) for k in ("saim_win_frac", "saim_abs_off", "saim_sat")})
        print(f"[saim_swing] scripted aim {cfg.saim_gain}: bound {cfg.saim_bound} rad, window {cfg.saim_window_s} s, "
              f"comp {'full' if cfg.saim_comp_full else 'q'}", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        c = self.cfg
        self.res_closed |= self.hit | (self.pitch_pos[:, 0] < self.contact_x_ref)
        win = (self.ref_t >= self.ref.contact_time[self.clip] - c.saim_window_s) & ~self.res_closed
        if self._ball_seen is None:
            dy = torch.zeros(self.num_envs, device=self.device)
        else:
            dy = self._ball_seen[:, 1] - self.nominal_target[:, 1]
        raw = torch.where((dy >= 0).unsqueeze(-1), self._gp, self._gm) * dy.abs().unsqueeze(-1) / c.saim_bound
        res_a = raw.clamp(-1.0, 1.0) * win.unsqueeze(-1)
        decay = torch.exp(torch.tensor(-self.step_dt / c.saim_tau_s, device=self.device))
        self.res_off = torch.where(win.unsqueeze(-1), c.saim_bound * res_a,
                                   torch.where(self.res_closed.unsqueeze(-1), self.res_off * decay, torch.zeros_like(self.res_off)))
        tgt = self.joint_targets[:, self.aim_ids] + self.res_off
        self.joint_targets[:, self.aim_ids] = torch.clamp(tgt, self.q_lo[self.aim_cols], self.q_hi[self.aim_cols])
        a = 0.01
        self.track_stats["saim_win_frac"] = (1 - a) * self.track_stats["saim_win_frac"] + a * win.float().mean()
        if win.any():
            self.track_stats["saim_abs_off"] = (1 - a) * self.track_stats["saim_abs_off"] + a * self.res_off[win].abs().mean()
            self.track_stats["saim_sat"] = (1 - a) * self.track_stats["saim_sat"] + a * (res_a[win].abs() > 0.95).float().mean()

    def _bat_state(self):
        out = super()._bat_state()
        if self._bat_comp is None:
            return out
        knob, axis, cm, v_cm, w, quat = out
        dp, dw = self._bat_comp  # the knob's linear and the bat's angular displacement from the offset (first order)
        return (knob - dp, axis - torch.cross(dw, axis, dim=-1), cm - dp - torch.cross(dw, cm - knob, dim=-1), v_cm, w, quat)

    def _perceived_ball(self) -> torch.Tensor:
        self._ball_seen = super()._perceived_ball()
        return self._ball_seen

    def _get_observations(self) -> dict:
        # the swing policy sees q - offset on the aim joints (+ the bat moved back by J x offset): the shift is not a
        # disturbance for it to correct
        jp = self.robot.data.joint_pos
        jp[:, self.aim_ids] -= self.res_off
        if self.cfg.saim_comp_full:
            J = self.robot.root_physx_view.get_jacobians()[:, self.bat_id]  # world axes (n, 6, dofs [+6 floating base])
            cols = self.aim_ids + (J.shape[-1] - self.robot.num_joints)
            dx = torch.einsum("nij,nj->ni", J[:, :, cols], self.res_off)
            self._bat_comp = (dx[:, :3], dx[:, 3:])
        try:
            return super()._get_observations()
        finally:
            jp[:, self.aim_ids] += self.res_off
            self._bat_comp = None

    def _get_rewards(self) -> torch.Tensor:
        rew = super()._get_rewards()
        self.extras["log"].update({f"saim/{k}": self.track_stats[k].clone() for k in ("saim_win_frac", "saim_abs_off", "saim_sat")})
        return rew

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None or len(env_ids) == self.num_envs else env_ids
        self.res_off[ids] = 0.0
        self.res_closed[ids] = False
