"""Time-windowed action noise for rsl_rl's ActorCritic (peer design 2026-10-19, not used in a run yet).

Chosen action columns get their noise std multiplied by k only while the phase observation is inside a reference-time
window (e.g. the last 0.3 s before the release). The scaling lives inside ActorCritic.update_distribution, so the
rollout's sampled action, its stored log-prob and the PPO update's recomputed log-prob all use the same std - unlike
env-side noise injection, which would make PPO's ratios wrong.

The phase is read back from the *normalized* observation with a snapshot of the normalizer's mean / std. The snapshot is
taken at install and refreshed only right after each PPO update, so the rollout (which collects with that snapshot) and
the update that consumes it (which recomputes with the same snapshot) decode identical masks for identical stored
observations. The normalizer itself keeps updating during the rollout, so the decoded phase can differ from the true one
by that drift - harmless for a window mask; the PPO consistency is exact.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal


class WindowedStd:
    def __init__(self, actor_critic, normalizer, phase_col: int, phase_lo: float, phase_hi: float, cols, k: float):
        self.ac, self.norm = actor_critic, normalizer
        self.phase_col, self.lo, self.hi, self.k = phase_col, phase_lo, phase_hi, float(k)
        n = (actor_critic.std if hasattr(actor_critic, "std") else actor_critic.log_std).shape[0]
        self.colmask = torch.zeros(n, dtype=torch.bool)
        self.colmask[list(cols)] = True
        self.snapshot()
        self._orig = actor_critic.update_distribution
        actor_critic.update_distribution = self.update_distribution

    def snapshot(self):
        if hasattr(self.norm, "_mean"):
            self.m = self.norm._mean.squeeze(0)[self.phase_col].clone()
            self.s = self.norm._std.squeeze(0)[self.phase_col].clone() + self.norm.eps
        else:  # no normalization (Identity)
            self.m, self.s = torch.tensor(0.0), torch.tensor(1.0)

    def window(self, obs: torch.Tensor) -> torch.Tensor:
        raw = obs[:, self.phase_col] * self.s.to(obs.device) + self.m.to(obs.device)
        return (raw >= self.lo) & (raw <= self.hi)

    def update_distribution(self, observations):
        self._orig(observations)
        d = self.ac.distribution
        on = self.window(observations).unsqueeze(-1) & self.colmask.to(observations.device).unsqueeze(0)
        scale = torch.where(on, torch.full_like(d.stddev, self.k), torch.ones_like(d.stddev))
        self.ac.distribution = Normal(d.mean, d.stddev * scale)


def install(runner, env, cols, k: float, window_s: float = 0.3) -> WindowedStd:
    """Wire it into an OnPolicyRunner: phase column and window from the env, snapshot refreshed after every update."""
    base = env.unwrapped
    nb, nf = len(base.body_ids), len(base.finger_ids)
    phase_col = 2 * nb + 2 * nf + 10 + 10  # PitchEnv._get_observations: ... ball (9) + released (1), then t / phase_duration
    pd = base.cfg.phase_duration_s
    rel = float(base.release_ref)
    ws = WindowedStd(runner.alg.actor_critic, runner.obs_normalizer, phase_col, (rel - window_s) / pd, rel / pd, cols, k)
    update0 = runner.alg.update

    def update(*a, **kw):
        out = update0(*a, **kw)
        ws.snapshot()  # the next rollout and its update use this one
        return out

    runner.alg.update = update
    print(f"[INFO]: windowed action std: cols {list(cols)} x {k} for phase obs (col {phase_col}) in "
          f"[{(rel - window_s):.3f}, {rel:.3f}] s reference time", flush=True)
    return ws
