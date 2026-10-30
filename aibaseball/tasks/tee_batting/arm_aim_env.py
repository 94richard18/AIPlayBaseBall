"""Human28 "ArmAim" (planning 14, user 10-30 "動打者架構"): a separate residual policy on the arms + waist_yaw that moves
the sweet spot onto the ball, on top of the FROZEN swing policy (delivery 12500 @ 1.5x).

* Each control step the env runs the frozen swing actor on the swing env's own observation (computed once per step in
  _get_observations, so the perception history advances exactly as without ArmAim) and applies its action through
  PitchedBattingEnv._pre_physics_step unchanged. The ArmAim action then adds aim_bound * tanh(a) to the PD targets of
  cfg.aim_joints, only in the window [ref clock >= contact - aim_window_s .. first contact / the ball past the contact
  plane]; zero before it, decaying with aim_decay_tau_s after it (no snap at contact). The swing policy never sees the
  offset (its previous-action input stays its own action).
* Obs (proprio + task): the aim joints' q / qd, swing target - q for them, sweet spot - perceived ball and the
  perceived ball velocity in the root frame, bat axis (root frame), t_arrive + released, window phase, previous offset.
* Reward (cfg weights, see ArmAimEnvCfg): aim term once per swing at closest approach + w_hit_q * q at contact + the
  miss / fall penalties of the swing env - w_res * sum(tanh(a)^2) in the window. No tracking, speed or distance terms.
"""

from __future__ import annotations

import atexit
import json
from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_rotate_inverse

from .pitched_env import PitchedBattingEnv


def load_frozen_actor(path: str, dev):
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = ck["model_state_dict"]
    n_obs, n_act, n_cr = sd["actor.0.weight"].shape[1], sd["actor.6.weight"].shape[0], sd["critic.0.weight"].shape[1]
    ac = ActorCritic(n_obs, n_cr, n_act, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                     activation="elu", init_noise_std=0.3).to(dev)
    ac.load_state_dict(sd)
    ac.eval()
    norm = EmpiricalNormalization(shape=[n_obs], until=1.0e8).to(dev)
    norm.load_state_dict(ck["obs_norm_state_dict"])
    norm.eval()
    return ac, norm, n_obs


class ArmAimEnv(PitchedBattingEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        names = list(self.body_joint_names)
        self.aim_cols = torch.tensor([names.index(j) for j in cfg.aim_joints], device=dev)
        self.aim_ids = self.body_joint_ids[self.aim_cols]
        self._swing_actor, self._swing_norm, n_sw = load_frozen_actor(cfg.swing_checkpoint, dev)
        self._swing_obs = None
        self.res_a = torch.zeros(n, len(cfg.aim_joints), device=dev)  # tanh(action) of this step
        self.res_off = torch.zeros_like(self.res_a)  # rad, applied offset
        self.res_win = torch.zeros(n, dtype=torch.bool, device=dev)
        self.res_closed = torch.zeros(n, dtype=torch.bool, device=dev)  # window over (contact / ball past the plane)
        self.res_opened = torch.zeros(n, dtype=torch.bool, device=dev)  # aim_hold: the window has opened (action latched)
        self.res_hold = torch.zeros_like(self.res_a)  # aim_hold: tanh(action) latched at the window's first step
        self.res_pen = torch.zeros(n, device=dev)  # the w_res term: sum of the policy's own share squared
        self.swing_gap = torch.zeros_like(self.res_a)  # swing target - q on the aim joints
        self._scripted = None
        if cfg.aim_scripted:  # falsifier: offset = gain(side) x |dy|, dy = predicted crossing y - the nominal target y
            g = json.load(open(cfg.aim_scripted))
            assert g["joints"] == list(cfg.aim_joints)
            self._scripted = (torch.tensor(g["gain_plus"], device=dev), torch.tensor(g["gain_minus"], device=dev))
            self._scripted_z = None
            if cfg.aim_scripted_z:  # + a height offset: raise the sweet spot by D_fit(true plate y, z) - D target (12500 fit)
                self._scripted_z = (torch.tensor(g["gain_zplus"], device=dev), torch.tensor(g["gain_zminus"], device=dev))
            self._prev_off_z = torch.zeros_like(self.res_a)
            self._rl_num_z = torch.zeros(n, device=dev)
            self._rl_den_z = torch.zeros(n, device=dev)
            self._rl_frac_z = []
            self._raw_z = torch.zeros_like(self.res_a)
            self.res_sat_ep = torch.zeros(n, dtype=torch.bool, device=dev)  # this pitch's offset reached the bound
            self.res_dy_ep = torch.zeros(n, device=dev)  # dy at the window's last step
            self.res_dyt_ep = torch.zeros(n, device=dev)  # the true plate y (pitch_plate) at the same step
            self._sat_n = [0, 0, 0.0, 0.0, 0.0]  # episodes, saturated, sum |dy|, sum |true dy|, sum |dy - true dy|
            # realised share of the commanded offset: (q - swing target) projected on the previous step's offset, per group
            js = list(cfg.aim_joints)
            grp = {"l_arm": [i for i, j in enumerate(js) if j.startswith("l_") and "wrist" not in j],
                   "r_arm": [i for i, j in enumerate(js) if j.startswith("r_") and "wrist" not in j],
                   "wrists": [i for i, j in enumerate(js) if "wrist" in j], "waist": [js.index("waist_yaw")]}
            self._grp = {k: torch.tensor(v, device=dev) for k, v in grp.items()}
            self._prev_st = torch.zeros_like(self.res_a)
            self._prev_off = torch.zeros_like(self.res_a)
            self._prev_win = torch.zeros(n, dtype=torch.bool, device=dev)
            self._rl_num = {k: torch.zeros(n, device=dev) for k in grp}
            self._rl_den = {k: torch.zeros(n, device=dev) for k in grp}
            self._rl_cmd = torch.zeros(n, device=dev)  # sum over window steps of mean |offset| (rad)
            self._rl_steps = torch.zeros(n, device=dev)
            self._rl_frac = {k: [] for k in grp}
            self._rl_cmd_ep = []
            atexit.register(self._print_scripted)
        self.track_stats.update({k: torch.zeros((), device=dev) for k in ("aim_win_frac", "aim_abs_off", "aim_sat")})
        # the ArmAim policy's spaces (the swing env's 30 / 152 stay what the swing actor uses)
        self.cfg.action_space = len(cfg.aim_joints)
        self.cfg.observation_space = 4 * len(cfg.aim_joints) + 3 + 3 + 3 + 3
        swing_actions = self.actions
        self._configure_gym_env_spaces()  # (it also re-allocates self.actions at the new size)
        self.actions = swing_actions  # the swing env's buffer (30): the swing actor's previous-action input
        print(f"[armaim] frozen swing actor {cfg.swing_checkpoint} (obs {n_sw}); aim joints {len(cfg.aim_joints)}, "
              f"bound {cfg.aim_bound} rad, window {cfg.aim_window_s} s; obs {self.cfg.observation_space}", flush=True)

    def _print_scripted(self):
        n, k, d, dt, de = self._sat_n
        import numpy as np
        fr = " ".join(f"{g} {np.median(v):.2f}" if v else f"{g} nan" for g, v in self._rl_frac.items())
        cm = np.median(self._rl_cmd_ep) if self._rl_cmd_ep else float('nan')
        if self._rl_frac_z:
            fr += f"; z part {np.median(self._rl_frac_z):.2f}"
        print(f"[armaim scripted] realised fraction of the commanded offset (median over pitches): {fr}; "
              f"commanded mean |offset| {cm:.3f} rad, window {self.cfg.aim_window_s} s", flush=True)
        print(f"[armaim scripted] sign {self.cfg.aim_scripted_sign:+.0f}: episodes {n}, offset at the +-{self.cfg.aim_bound} "
              f"bound {k / max(n, 1):.3f}, mean |dy| {d / max(n, 1):.3f} m, mean |true plate y| {dt / max(n, 1):.3f}, "
              f"mean |dy - true| {de / max(n, 1):.3f} (dy from {'the true plate y' if self.cfg.aim_scripted_true else 'the percept'})", flush=True)

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        if self._swing_obs is None:
            self._swing_obs = PitchedBattingEnv._get_observations(self)["policy"]
        with torch.no_grad():
            sw = self._swing_actor.act_inference(self._swing_norm(self._swing_obs))
        super()._pre_physics_step(sw.clone())
        # the window: from aim_window_s (reference clock) before the clip's contact until contact / the ball past the plane
        self.res_closed |= self.hit | (self.pitch_pos[:, 0] < self.contact_x_ref)
        self.res_win = (self.ref_t >= self.ref.contact_time[self.clip] - c.aim_window_s) & ~self.res_closed
        if self._scripted is not None:
            dyt = self.pitch_plate[:, 0]  # the true plate y (world target y = nominal y + this)
            dy = dyt if c.aim_scripted_true else self._ball_seen[:, 1] - self.nominal_target[:, 1]
            self.res_dyt_ep = torch.where(self.res_win, dyt, self.res_dyt_ep)
            dy = dy * c.aim_scripted_sign
            gp, gm = self._scripted
            raw = torch.where((dy >= 0).unsqueeze(-1), gp, gm) * dy.abs().unsqueeze(-1) / c.aim_bound
            if self._scripted_z is not None:
                a0, ay, az = c.aim_scripted_D_fit
                dz = a0 + ay * self.pitch_plate[:, 0] + az * self.pitch_plate[:, 1] - c.aim_scripted_D_target
                zp, zm = self._scripted_z
                self._raw_z = torch.where((dz >= 0).unsqueeze(-1), zp, zm) * dz.abs().unsqueeze(-1) / c.aim_bound
                raw = raw + self._raw_z
            w = self.res_win
            self.res_sat_ep |= w & (raw.abs() > 1.0).any(-1)
            self.res_dy_ep = torch.where(w, dy, self.res_dy_ep)
            base = raw.clamp(-1.0, 1.0)
        else:
            base = torch.zeros_like(self.res_a)
        pol = torch.tanh(actions.clamp(-5.0, 5.0))
        if c.aim_hold:  # one decision per swing, held through the window
            opening = self.res_win & ~self.res_opened
            self.res_hold = torch.where(opening.unsqueeze(-1), pol, self.res_hold)
            self.res_opened |= self.res_win
            share = c.aim_delta * self.res_hold if self._scripted is not None else self.res_hold
        else:
            share = pol if self._scripted is None else torch.zeros_like(pol)
        self.res_a = (base + share).clamp(-1.0, 1.0) * self.res_win.unsqueeze(-1)
        self.res_pen = (share ** 2).sum(-1) * self.res_win.float()
        decay = torch.exp(torch.tensor(-self.step_dt / c.aim_decay_tau_s, device=self.device))
        self.res_off = torch.where(self.res_win.unsqueeze(-1), c.aim_bound * self.res_a,
                                   torch.where(self.res_closed.unsqueeze(-1), self.res_off * decay, torch.zeros_like(self.res_off)))
        if self._scripted is not None:
            q = self.robot.data.joint_pos[:, self.aim_ids]
            pw = self._prev_win
            for g, ix in self._grp.items():
                o = self._prev_off[:, ix]
                self._rl_num[g] += ((q[:, ix] - self._prev_st[:, ix]) * o).sum(-1) * pw
                self._rl_den[g] += (o ** 2).sum(-1) * pw
            self._rl_cmd += self._prev_off.abs().mean(-1) * pw
            if self._scripted_z is not None:
                oz = self._prev_off_z
                self._rl_num_z += ((q - self._prev_st) * oz).sum(-1) * pw
                self._rl_den_z += (oz ** 2).sum(-1) * pw
            self._rl_steps += pw.float()
            self._prev_st = self.joint_targets[:, self.aim_ids].clone()
            self._prev_off = self.res_off.clone()
            self._prev_off_z = c.aim_bound * self._raw_z * self.res_win.unsqueeze(-1)  # the z part (before the bound clamp)
            self._prev_win = self.res_win.clone()
        tgt = self.joint_targets[:, self.aim_ids] + self.res_off
        self.joint_targets[:, self.aim_ids] = torch.clamp(tgt, self.q_lo[self.aim_cols], self.q_hi[self.aim_cols])
        a = 0.01
        self.track_stats["aim_win_frac"] = (1 - a) * self.track_stats["aim_win_frac"] + a * self.res_win.float().mean()
        if self.res_win.any():
            w = self.res_win
            self.track_stats["aim_abs_off"] = (1 - a) * self.track_stats["aim_abs_off"] + a * self.res_off[w].abs().mean()
            self.track_stats["aim_sat"] = (1 - a) * self.track_stats["aim_sat"] + a * (self.res_a[w].abs() > 0.95).float().mean()

    # ------------------------------------------------------------------ observations
    def _perceived_ball(self) -> torch.Tensor:
        self._ball_seen = super()._perceived_ball()
        return self._ball_seen

    def _get_observations(self) -> dict:
        self._swing_obs = super()._get_observations()["policy"]  # once per step: advances the percept history
        c = self.cfg
        d = self.robot.data
        rq, rp = d.root_link_quat_w, d.root_link_pos_w
        knob, axis, cm, v_cm, w, _ = self._bat_state()
        ss_p, _ = self._sweet_spot(knob, axis, cm, v_cm, w)
        q = d.joint_pos[:, self.aim_ids]
        qd = d.joint_vel[:, self.aim_ids]
        clip = c.ball_obs_clip
        phase = ((self.ref_t - (self.ref.contact_time[self.clip] - c.aim_window_s)) / c.aim_window_s).clamp(-1.0, 3.0)
        obs = torch.cat([
            q - d.default_joint_pos[:, self.aim_ids],
            qd * 0.05,
            self.joint_targets[:, self.aim_ids] - q,
            quat_rotate_inverse(rq, ss_p - self._ball_seen).clamp(-clip, clip),
            quat_rotate_inverse(rq, self.v_seen) * 0.05,
            quat_rotate_inverse(rq, axis),
            (self.t_arrive * 2.0).unsqueeze(-1),
            self.released_seen.float().unsqueeze(-1),
            phase.unsqueeze(-1),
            self.res_off / c.aim_bound,
        ], -1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        # the aim term, read before the swing env's _get_rewards pays (and latches) it
        pay = (self.new_hit | ((self.reset_terminated | self.reset_time_outs) & ~self.hit)) & ~self.aim_paid
        gap = self.aim_gap.clamp_min(0.0).nan_to_num(1.0, posinf=1.0)
        r_aim = c.w_aim * torch.exp(-((self.aim_ax / c.sigma_aim_ax) ** 2) - (gap / c.sigma_aim_gap) ** 2) * pay.float()
        super()._get_rewards()  # side effects only: _hit_rewards (hit_reward = w_hit_q * q), stats, logs, plate_crossed
        miss = (self.reset_time_outs & ~self.hit) | self.passed
        rew = (self.hit_reward + r_aim - c.w_miss * miss.float()
               - torch.where(self.hit, c.w_fall_after_hit, c.w_fall) * self.fallen.float()
               - c.w_res * self.res_pen)
        self.extras["log"].update({f"armaim/{k}": self.track_stats[k].clone() for k in ("aim_win_frac", "aim_abs_off", "aim_sat")})
        return rew

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if self._scripted is not None and self.common_step_counter > 0:
            done = self.res_closed[env_ids]  # pitches whose window ran
            k0 = self._sat_n[0] // 256
            self._sat_n[0] += int(done.sum())
            self._sat_n[1] += int((self.res_sat_ep[env_ids] & done).sum())
            self._sat_n[2] += float(self.res_dy_ep[env_ids][done].abs().sum())
            self._sat_n[3] += float(self.res_dyt_ep[env_ids][done].abs().sum())
            self._sat_n[4] += float((self.res_dy_ep[env_ids] * self.cfg.aim_scripted_sign - self.res_dyt_ep[env_ids])[done].abs().sum())
            ok = done & (self._rl_steps[env_ids] > 0)
            for g in self._grp:
                den = self._rl_den[g][env_ids]
                m = ok & (den > 1e-8)
                self._rl_frac[g] += (self._rl_num[g][env_ids][m] / den[m]).tolist()
                self._rl_num[g][env_ids] = 0.0
                self._rl_den[g][env_ids] = 0.0
            if self._scripted_z is not None:
                m = ok & (self._rl_den_z[env_ids] > 1e-8)
                self._rl_frac_z += (self._rl_num_z[env_ids][m] / self._rl_den_z[env_ids][m]).tolist()
                self._rl_num_z[env_ids] = 0.0
                self._rl_den_z[env_ids] = 0.0
            self._rl_cmd_ep += (self._rl_cmd[env_ids][ok] / self._rl_steps[env_ids][ok]).tolist()
            self._rl_cmd[env_ids] = 0.0
            self._rl_steps[env_ids] = 0.0
            self._prev_win[env_ids] = False
            self._prev_off[env_ids] = 0.0
            self.res_sat_ep[env_ids] = False
            self.res_dy_ep[env_ids] = 0.0
            if self._sat_n[0] // 256 > k0:
                self._print_scripted()
        self.res_a[env_ids] = 0.0
        self.res_off[env_ids] = 0.0
        self.res_win[env_ids] = False
        self.res_closed[env_ids] = False
        self.res_opened[env_ids] = False
        self.res_hold[env_ids] = 0.0
        self.res_pen[env_ids] = 0.0
        self._swing_obs = None  # recomputed for the new episode before the next action
