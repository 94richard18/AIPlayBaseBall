"""Analytic pitch generator: aimed fastballs with the same drag + Magnus model as everything else.

The batting world has home plate at the origin and the pitcher toward +X, so a pitch travels toward -X.
A library of pitches is aimed once (iterative shooting) and reused: with no wind the flight is
translation invariant, so a library pitch can be shifted to cross any target point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .ball_flight import _aero_consts, _deriv
from .constants import PhysicsSpec

KMH = 1 / 3.6


@torch.no_grad()
def fly_to_x(pos, vel, omega, spec: PhysicsSpec, x_target: float, dt: float = 0.004, t_max: float = 1.5):
    """RK4 toward -X until x <= x_target. Returns (position at the plane, velocity there, time)."""
    pos, vel, omega = pos.clone(), vel.clone(), omega.clone()
    c = _aero_consts(spec, pos)
    n = pos.shape[0]
    done = torch.zeros(n, dtype=torch.bool, device=pos.device)
    p_hit, v_hit = pos.clone(), vel.clone()
    t_hit = torch.full((n,), t_max, device=pos.device, dtype=pos.dtype)
    t = 0.0
    for _ in range(int(t_max / dt)):
        a1, w1 = _deriv(vel, omega, spec, c)
        a2, w2 = _deriv(vel + 0.5 * dt * a1, omega + 0.5 * dt * w1, spec, c)
        a3, w3 = _deriv(vel + 0.5 * dt * a2, omega + 0.5 * dt * w2, spec, c)
        a4, w4 = _deriv(vel + dt * a3, omega + dt * w3, spec, c)
        np_ = pos + dt / 6 * (vel + 2 * (vel + 0.5 * dt * a1) + 2 * (vel + 0.5 * dt * a2) + (vel + dt * a3))
        nv = vel + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        omega = omega + dt / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        cross = (~done) & (np_[:, 0] <= x_target)
        f = ((pos[:, 0] - x_target) / (pos[:, 0] - np_[:, 0]).clamp_min(1e-9)).unsqueeze(-1)
        p_hit = torch.where(cross.unsqueeze(-1), pos + f * (np_ - pos), p_hit)
        v_hit = torch.where(cross.unsqueeze(-1), vel + f * (nv - vel), v_hit)
        t_hit = torch.where(cross, t + f.squeeze(-1) * dt, t_hit)
        done |= cross
        pos, vel = np_, nv
        t += dt
        if bool(done.all()):
            break
    return p_hit, v_hit, t_hit


@dataclass
class PitchLibrary:
    release: torch.Tensor  # (N,3) release point
    target: torch.Tensor  # (N,3) point the pitch crosses (aimed)
    v0: torch.Tensor  # (N,3) release velocity
    omega: torch.Tensor  # (N,3) spin
    flight_time: torch.Tensor  # (N,) release -> target
    speed_kmh: torch.Tensor  # (N,) release speed

    def to(self, device):
        return PitchLibrary(*(getattr(self, k).to(device) for k in self.__dataclass_fields__))


@torch.no_grad()
def build_library(n: int, spec: PhysicsSpec, speed_kmh=(75.0, 155.0), backspin_rpm=(1500.0, 2500.0),
                  spin_tilt_deg=30.0, seed: int = 0, iters: int = 5) -> PitchLibrary:
    """Right-handed-pitcher fastballs released ~16.9 m from the plate, aimed at targets around the zone."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
    release = torch.stack([torch.full((n,), 16.9), u(0.25, 0.75), u(1.55, 1.95)], -1)
    target = torch.stack([torch.zeros(n), u(-0.35, 0.35), u(0.45, 1.15)], -1)
    speed = u(*speed_kmh) * KMH
    spin = u(*backspin_rpm) * 2 * math.pi / 60
    tilt = torch.deg2rad(u(-spin_tilt_deg, spin_tilt_deg))
    # backspin for travel toward -X: omega along +Y (Magnus = omega x v -> +Z lift); tilt the axis about X
    omega = torch.stack([torch.zeros(n), torch.cos(tilt), torch.sin(tilt)], -1) * spin.unsqueeze(-1)
    aim = target.clone()
    for _ in range(iters):  # shooting: correct the aim point by the miss at the target plane
        d = torch.nn.functional.normalize(aim - release, dim=-1)
        p_hit, _, t_hit = fly_to_x(release, d * speed.unsqueeze(-1), omega, spec, 0.0)
        aim[:, 1:] += target[:, 1:] - p_hit[:, 1:]
    d = torch.nn.functional.normalize(aim - release, dim=-1)
    v0 = d * speed.unsqueeze(-1)
    p_hit, _, t_hit = fly_to_x(release, v0, omega, spec, 0.0)
    err = (p_hit[:, 1:] - target[:, 1:]).norm(dim=-1)
    assert float(err.max()) < 0.01, f"pitch aiming did not converge (max miss {float(err.max()):.3f} m)"
    order = torch.argsort(speed)
    return PitchLibrary(release=release[order], target=target[order], v0=v0[order], omega=omega[order],
                        flight_time=t_hit[order], speed_kmh=(speed * 3.6)[order])
