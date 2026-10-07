"""Step D: hitting pitched fastballs (analytic pitch library, same drag + Magnus physics).

* Each episode picks a retargeted swing (clip) and a library pitch. The pitch is shifted so that it
  crosses that clip's contact point (+ location offset), and released so that it arrives there when the
  reference reaches its contact time at nominal tempo (+ a timing offset the batter must absorb).
* The batter perceives the ball with bias, noise and latency (step B), now including its velocity.
* Extra action: tempo in [-1, 1] -> reference playback rate x (1 + tempo_scale * tempo), so the
  batter can start / speed up / slow down its swing to meet the ball.
* The ball is integrated every physics step (RK4) and the swept contact test interpolates both the
  bat and the ball (a 150 km/h pitch travels ~10 cm per physics step).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from aibaseball.physics.ball_flight import _deriv
from aibaseball.physics.pitch_gen import build_library
from isaaclab.utils.math import quat_rotate_inverse

from .mimic_env import TeeBattingMimicEnv
from .pitched_env_cfg import PitchedBattingEnvCfg


class PitchedBattingEnv(TeeBattingMimicEnv):
    cfg: PitchedBattingEnvCfg

    def __init__(self, cfg: PitchedBattingEnvCfg, render_mode: str | None = None, **kwargs):
        self.lib = None
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        lo = min(cfg.pitch_speed_range_start[0], cfg.pitch_speed_range_final[0])
        hi = max(cfg.pitch_speed_range_start[1], cfg.pitch_speed_range_final[1])
        self.lib = build_library(cfg.pitch_library_size, self.phys, speed_kmh=(lo, hi)).to(dev)  # whole curriculum
        self.ref_t = torch.zeros(n, device=dev)
        self.rate = torch.full((n,), self.speed, device=dev)
        self.phys_t = torch.zeros(n, device=dev)  # physical time since episode start
        self.release_t = torch.zeros(n, device=dev)
        self.pitch_pos = torch.zeros(n, 3, device=dev)
        self.pitch_vel = torch.zeros(n, 3, device=dev)
        self.pitch_omega = torch.zeros(n, 3, device=dev)
        self.pitch_prev = torch.zeros(n, 3, device=dev)
        self.pitch_kmh = torch.zeros(n, device=dev)
        self.contact_x_ref = torch.zeros(n, device=dev)
        self.passed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.nominal_target = torch.zeros(n, 3, device=dev)
        self.flight_time = torch.zeros(n, device=dev)
        self.t_arrive = torch.zeros(n, device=dev)
        self.v_seen = torch.zeros(n, 3, device=dev)
        self.released_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.track_stats.update({k: torch.zeros((), device=dev) for k in ("pitch_kmh", "timing_err_s", "tempo")})

    # ------------------------------------------------------------------ curriculum
    def _progress(self) -> float:
        c = self.cfg
        return min(1.0, self.common_step_counter / c.pitch_ramp_steps) if c.pitch_ramp_steps > 0 else 1.0

    # ------------------------------------------------------------------ reference clock with tempo
    def _ref_time(self, extra_steps: int = 0) -> torch.Tensor:
        if self.lib is None:  # during construction
            return super()._ref_time(extra_steps)
        return self.ref_t + extra_steps * self.rate * self.step_dt

    def _pre_physics_step(self, actions: torch.Tensor):
        c = self.cfg
        a = actions.clamp(-c.action_clip, c.action_clip)
        tempo = a[:, -1].clamp(-1.0, 1.0)
        self.rate = self.speed * (1.0 + c.tempo_scale * tempo)
        self.ref_t = self.ref_t + self.rate * self.step_dt
        self.prev_actions[:] = self.actions
        self.actions = a
        nxt = self._ref(self.ref_t)
        tgt = nxt["joint_pos"] + a[:, :-1] * self.action_scale * c.residual_scale
        self.joint_targets[:, self.body_joint_ids] = torch.clamp(tgt, self.q_lo, self.q_hi)
        self.joint_vel_targets[:, self.body_joint_ids] = nxt["joint_vel"] / self.speed * self.rate.unsqueeze(-1) * c.vel_feedforward
        self.new_hit[:] = False
        self.hit_reward[:] = 0.0
        self.track_stats["tempo"] = 0.99 * self.track_stats["tempo"] + 0.01 * tempo.abs().mean()

    # ------------------------------------------------------------------ ball
    def _true_ball_pos(self) -> torch.Tensor:
        return self.pitch_pos if self.lib is not None else self.tee_pos

    def _incoming_ball(self):
        return self.pitch_pos, self.pitch_prev, self.pitch_vel, self.pitch_omega

    def _apply_action(self):
        super()._apply_action()  # impact check uses (pitch_prev -> pitch_pos) for this physics step
        self._advance_pitch(self.physics_dt)

    def _advance_pitch(self, dt: float):
        self.pitch_prev = self.pitch_pos.clone()
        self.phys_t += dt
        flying = (self.phys_t >= self.release_t) & ~self.hit
        c = self.aero_c
        p, v, w = self.pitch_pos, self.pitch_vel, self.pitch_omega
        a1, w1 = _deriv(v, w, self.phys, c)
        a2, w2 = _deriv(v + 0.5 * dt * a1, w + 0.5 * dt * w1, self.phys, c)
        a3, w3 = _deriv(v + 0.5 * dt * a2, w + 0.5 * dt * w2, self.phys, c)
        a4, w4 = _deriv(v + dt * a3, w + dt * w3, self.phys, c)
        new_p = p + dt / 6 * (v + 2 * (v + 0.5 * dt * a1) + 2 * (v + 0.5 * dt * a2) + (v + dt * a3))
        new_v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        new_w = w + dt / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        m = flying.unsqueeze(-1)
        self.pitch_pos = torch.where(m, new_p, p)
        self.pitch_vel = torch.where(m, new_v, v)
        self.pitch_omega = torch.where(m, new_w, w)

    # ------------------------------------------------------------------ observations
    def _perceived_ball(self) -> torch.Tensor:
        """What the batter uses as "the ball": its predicted crossing of the contact plane, extrapolated
        (gravity-only ballistic) from the delayed, noisy percept - the pitched-ball analogue of the tee.
        Before the release (ball in the pitcher's hand) it is the nominal target; the time to arrival
        then comes from the pitcher's delivery timing."""
        seen = super()._perceived_ball()  # updates the delayed/noisy history
        if self.lib is None:
            return seen
        hist = self.ball_obs_hist
        v = (hist[:, -2] - hist[:, -1]) / self.step_dt if hist.shape[1] > 1 else self.pitch_vel
        delay = self.obs_delay_steps * self.step_dt
        released = (self.phys_t - delay) >= self.release_t + self.step_dt  # the batter has seen it leave the hand
        vx = v[:, 0].clamp(max=-1.0)
        t = ((self.contact_x_ref - seen[:, 0]) / vx).clamp(0.0, 1.5)
        g = torch.tensor([0.0, 0.0, -self.phys.gravity], device=self.device)
        pred = seen + v * t.unsqueeze(-1) + 0.5 * g * (t**2).unsqueeze(-1)
        pred[:, 0] = self.contact_x_ref
        nominal = self.nominal_target
        self.t_arrive = torch.where(released, (t - delay).clamp_min(0.0),
                                    (self.release_t - self.phys_t) + self.flight_time)
        self.v_seen = v
        self.released_seen = released
        return torch.where(released.unsqueeze(-1), pred, nominal)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        if self.lib is None:
            return {"policy": obs}
        extra = torch.stack([self.t_arrive * 2.0, self.v_seen.norm(dim=-1) / 40.0, self.released_seen.float(),
                             self.rate / self.speed - 1.0, (self.pitch_kmh - 120.0) / 30.0 * 0.0], -1)
        return {"policy": torch.nan_to_num(torch.cat([obs, extra], -1))}

    # ------------------------------------------------------------------ dones / rewards
    def _get_dones(self):
        terminated, time_out = super()._get_dones()
        if self.lib is None:
            return terminated, time_out
        # the pitch got past the batter (or hit the ground) without contact -> miss
        self.passed = (~self.hit) & ((self.pitch_pos[:, 0] < self.contact_x_ref - 1.0) | (self.pitch_pos[:, 2] < 0.0))
        return terminated | self.passed, time_out

    def _get_rewards(self) -> torch.Tensor:
        rew = super()._get_rewards()
        if self.lib is None:
            return rew
        c = self.cfg
        rew = rew - c.w_miss * self.passed.float() - c.w_tempo * self.actions[:, -1] ** 2
        if self.new_hit.any():
            # timing error: physical time of contact vs. when the ball reached the reference contact point
            self.track_stats["pitch_kmh"] = 0.95 * self.track_stats["pitch_kmh"] + 0.05 * self.pitch_kmh[self.new_hit].mean()
        self.extras["log"].update({f"mimic/{k}": self.track_stats[k].clone() for k in ("pitch_kmh", "tempo")})
        return rew

    # ------------------------------------------------------------------ reset: new pitch
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self.lib is None:
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        n, dev, c = len(env_ids), self.device, self.cfg
        prog = self._progress()
        lo = c.pitch_speed_range_start[0] + (c.pitch_speed_range_final[0] - c.pitch_speed_range_start[0]) * prog
        hi = c.pitch_speed_range_start[1] + (c.pitch_speed_range_final[1] - c.pitch_speed_range_start[1]) * prog
        i_lo = torch.searchsorted(self.lib.speed_kmh, torch.tensor([lo], device=dev)).item()
        i_hi = max(i_lo + 1, torch.searchsorted(self.lib.speed_kmh, torch.tensor([hi], device=dev)).item())
        k = torch.randint(i_lo, i_hi, (n,), device=dev)
        # target: the clip's contact point (+ location offset, already applied to tee_pos by the parent reset)
        target = self.tee_pos[env_ids].clone()
        shift = target - self.lib.target[k]
        self.pitch_pos[env_ids] = self.lib.release[k] + shift
        self.pitch_prev[env_ids] = self.pitch_pos[env_ids]
        self.pitch_vel[env_ids] = self.lib.v0[k]
        self.pitch_omega[env_ids] = self.lib.omega[k]
        self.pitch_kmh[env_ids] = self.lib.speed_kmh[k]
        self.contact_x_ref[env_ids] = target[:, 0]
        nominal = self.scene.env_origins[env_ids] + self.ref.tee_pos[self.clip[env_ids]]
        nominal[:, 2] += c.tee_offset_z
        self.nominal_target[env_ids] = nominal
        self.flight_time[env_ids] = self.lib.flight_time[k]
        # timing: arrive at the contact point when the reference reaches contact at nominal tempo (+ offset).
        # Episodes start in the stance (t0 = 0); if the pitch needs longer than the swing, the batter first
        # waits in the stance (the reference clock starts negative and holds frame 0).
        clip = self.clip[env_ids]
        jitter = (torch.rand(n, device=dev) * 2 - 1) * c.timing_noise_s * prog
        release = (self.ref.contact_time[clip] - self.t0[env_ids]) / self.speed - self.lib.flight_time[k] + jitter
        wait = (c.min_lead_s - release).clamp_min(0.0)
        self.release_t[env_ids] = release + wait
        self.phys_t[env_ids] = 0.0
        self.ref_t[env_ids] = self.t0[env_ids] - wait * self.speed
        self.rate[env_ids] = self.speed
        self.passed[env_ids] = False
        # hide the tee (the ball is pitched)
        hidden = self.tee_pos[env_ids].clone()
        hidden[:, 2] = -3.0
        unit_q = torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)
        self.tee.write_root_pose_to_sim(torch.cat([hidden, unit_q], -1), env_ids)
        hidden_ball = self.pitch_pos[env_ids].clone()
        hidden_ball[:, 2] = -5.0  # the physics ball waits underground; the incoming pitch is a visual marker
        self.ball.write_root_pose_to_sim(torch.cat([hidden_ball, unit_q], -1), env_ids)
