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
                  spin_tilt_deg=30.0, seed: int = 0, iters: int = 5, release_point=None) -> PitchLibrary:
    """Right-handed-pitcher fastballs released ~16.9 m from the plate, aimed at targets around the zone.
    release_point (x, y, z): every pitch from this point instead (the game fallback: the RL pitcher's own release,
    hf_avg3 7300 + H5: (17.51, +1.17, 1.86) in this frame)."""
    g = torch.Generator().manual_seed(seed)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)  # noqa: E731
    release = torch.stack([torch.full((n,), 16.9), u(0.25, 0.75), u(1.55, 1.95)], -1)
    if release_point is not None:
        release = torch.tensor(release_point, dtype=release.dtype).expand(n, 3).clone()
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


GAME_RELEASE_POINT = (17.51, 1.17, 1.86)  # the RL pitcher's release (hf_avg3 7300 + H5) in this frame, as for the fallback


def read_pitch_table(path: str):
    """Rows of an eval_stage --pitch_csv table that reached the plate: (speed_mps, plate_y, plate_z) in the batting frame.
    The pitching frame's +y is the first-base side; the batting frame's +y is the third-base side (TablePitcher's flip)."""
    import csv

    with open(path, newline="") as f:
        rows = [(float(r["speed_mps"]), -float(r["plate_y"]), float(r["plate_z"]))
                for r in csv.DictReader(f) if int(float(r.get("crossed", 1)))]
    assert rows, f"no pitches that reached the plate in {path}"
    return rows


def build_library_from_table(rows, spec: PhysicsSpec, release_point=GAME_RELEASE_POINT, backspin_rpm: float = 2000.0,
                             iters: int = 5) -> PitchLibrary:
    """One aimed pitch per table row (release speed, plate y / z) from the game's release point, with the same drag +
    Magnus flight as build_library; flight_time is the release -> plate time of that flight."""
    t = torch.tensor(rows, dtype=torch.float32)
    n = t.shape[0]
    release = torch.tensor(release_point, dtype=torch.float32).expand(n, 3).clone()
    target = torch.stack([torch.zeros(n), t[:, 1], t[:, 2]], -1)
    speed = t[:, 0]
    omega = torch.stack([torch.zeros(n), torch.ones(n), torch.zeros(n)], -1) * (backspin_rpm * 2 * math.pi / 60)
    aim = target.clone()
    for _ in range(iters):
        d = torch.nn.functional.normalize(aim - release, dim=-1)
        p_hit, _, t_hit = fly_to_x(release, d * speed.unsqueeze(-1), omega, spec, 0.0)
        aim[:, 1:] += target[:, 1:] - p_hit[:, 1:]
    d = torch.nn.functional.normalize(aim - release, dim=-1)
    v0 = d * speed.unsqueeze(-1)
    p_hit, _, t_hit = fly_to_x(release, v0, omega, spec, 0.0)
    err = (p_hit[:, 1:] - target[:, 1:]).norm(dim=-1)
    assert float(err.max()) < 0.02, f"table pitch aiming did not converge (max miss {float(err.max()):.3f} m)"
    return PitchLibrary(release=release, target=target, v0=v0, omega=omega, flight_time=t_hit, speed_kmh=speed * 3.6)


PITCH_PLATE_DISTANCE = 18.44  # m, rubber -> plate in the pitching env's plate frame


def build_library_from_release(path: str, spec: PhysicsSpec, plate_distance: float = PITCH_PLATE_DISTANCE) -> PitchLibrary:
    """One pitch per released + thrown row of an eval_stage --release_csv table, flown from its own release state.
    The pitching plate frame has the rubber at the origin, +x toward the plate (at plate_distance), +y the first-base side,
    z above the ground; this frame has the plate at x = 0, the ball travelling -x and +y the third-base side: a 180 deg
    turn about the vertical (x -> plate_distance - x, y -> -y; vectors flip x and y). Rows whose flight does not reach
    the plate are dropped."""
    import csv

    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(float(r["released"])) and int(float(r["thrown"]))]
    assert rows, f"no released + thrown pitches in {path}"
    g = lambda k: torch.tensor([float(r[k]) for r in rows], dtype=torch.float32)  # noqa: E731
    release = torch.stack([plate_distance - g("x"), -g("y"), g("z")], -1)
    if "vx" in rows[0]:
        v0 = torch.stack([-g("vx"), -g("vy"), g("vz")], -1)
        omega = torch.stack([-g("wx"), -g("wy"), g("wz")], -1)
    else:  # older tables (before the full release state): velocity from speed / heading / elevation, no spin
        print(f"[pitch_gen] {path}: no vx..wz columns - velocity from speed/heading/elevation, spin 0", flush=True)
        sp, hd, el = g("speed"), torch.deg2rad(g("heading")), torch.deg2rad(g("elevation"))
        v0 = torch.stack([-sp * torch.cos(el) * torch.cos(hd), -sp * torch.cos(el) * torch.sin(hd), sp * torch.sin(el)], -1)
        omega = torch.zeros_like(v0)
    p_hit, _, t_hit = fly_to_x(release, v0, omega, spec, 0.0)
    ok = (p_hit[:, 0] <= 0.01) & (p_hit[:, 2] > 0.0)
    assert ok.any(), f"no pitch in {path} reaches the plate"
    k = ok.nonzero(as_tuple=False).squeeze(-1)
    target = torch.stack([torch.zeros(len(k)), p_hit[k, 1], p_hit[k, 2]], -1)
    return PitchLibrary(release=release[k], target=target, v0=v0[k], omega=omega[k], flight_time=t_hit[k],
                        speed_kmh=v0[k].norm(dim=-1) * 3.6)
