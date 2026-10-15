"""Ball on natural grass: bounces with spin and friction, then sliding and rolling to rest (one-month plan, W2).

Bounce (impulse model of a spinning sphere at a rough surface, after R. Cross and A. M. Nathan):
- normal: |v_z'| = e |v_z|, with e dropping on soft turf as the impact gets harder:
  e = clip(e0 - e_slope |v_z|, e_min, e0)
- tangential: the contact point slips at u = v_t + omega x (-R z). Friction gives the centre a velocity change
  of at most mu (1 + e) |v_z| against u, capped at the change that ends the slip (rolling):
  |u| / (1 + 1 / k) for I = k m R^2. The same impulse changes the spin.
On the ground: sliding at mu_slide g until the contact point stops slipping, then rolling at mu_roll g.
Topspin (a typical grounder) makes the ball skip forward; backspin checks it.

The coefficients are approximate values for an MLB infield / outfield cut; calibrate them against measured
bounces (W2) - nothing here is fitted yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .constants import BallSpec


@dataclass(frozen=True)
class GrassSpec:
    e0: float = 0.55  # COR of a gentle bounce
    e_slope: float = 0.012  # per m/s of normal impact speed
    e_min: float = 0.30
    mu_bounce: float = 0.50  # friction during a bounce
    mu_slide: float = 0.40  # sliding on the grass
    mu_roll: float = 0.25  # rolling resistance (decel = mu_roll g)
    roll_below_mps: float = 0.8  # normal speed under which the ball stops bouncing


def restitution(vz_in: float, grass: GrassSpec) -> float:
    return min(grass.e0, max(grass.e_min, grass.e0 - grass.e_slope * abs(vz_in)))


def _slip(v, w, r):
    """Contact-point velocity (x, y) of a ball at the ground: v_t + omega x (0, 0, -r)."""
    return v[0] - r * w[1], v[1] + r * w[0]


def _friction(v, w, dv_max, ball: BallSpec):
    """Apply a tangential velocity change of at most dv_max against the slip, never past rolling."""
    r, k = ball.radius, ball.inertia_factor
    ux, uy = _slip(v, w, r)
    u = math.hypot(ux, uy)
    if u < 1e-9:
        return v, w, True
    dv = min(dv_max, u / (1.0 + 1.0 / k))
    dvx, dvy = -dv * ux / u, -dv * uy / u
    v = (v[0] + dvx, v[1] + dvy, v[2])
    w = (w[0] + dvy / (k * r), w[1] - dvx / (k * r), w[2])  # r x (m dv) / I with r = (0, 0, -R)
    return v, w, dv < dv_max


def bounce(v, w, grass: GrassSpec, ball: BallSpec):
    """Velocity and spin after one bounce; v[2] < 0 coming in."""
    e = restitution(v[2], grass)
    v, w, _ = _friction(v, w, grass.mu_bounce * (1.0 + e) * abs(v[2]), ball)
    return (v[0], v[1], e * abs(v[2])), w


def ground_path(t: float, x: float, y: float, v, w, grass: GrassSpec, ball: BallSpec, g: float, dt: float,
                stop: Callable[[float, float], bool] | None = None, t_max: float = 30.0) -> list:
    """Samples (t, x, y, z) after the ball first lands at (x, y) at time t with velocity v (v[2] < 0) and spin w,
    until it is at rest or stop(x, y) is true (e.g. at the wall)."""
    out = []
    while True:  # bounces
        v, w = bounce(v, w, grass, ball)
        if v[2] <= grass.roll_below_mps:
            break
        tb = 2.0 * v[2] / g
        n = max(1, math.ceil(tb / dt))
        for i in range(1, n + 1):
            tt = tb * i / n
            px, py = x + v[0] * tt, y + v[1] * tt
            out.append((t + tt, px, py, max(0.0, v[2] * tt - 0.5 * g * tt * tt)))
            if stop and stop(px, py):
                return out
        x, y, t = x + v[0] * tb, y + v[1] * tb, t + tb
        v = (v[0], v[1], -v[2])
    v = (v[0], v[1], 0.0)
    rolling = False
    while t < t_max:  # slide, then roll
        speed = math.hypot(v[0], v[1])
        if speed < 0.05:
            break
        if not rolling:
            v2, w, rolling = _friction(v, w, grass.mu_slide * g * dt, ball)
        if rolling:  # rolling resistance; the spin follows the speed
            s = max(0.0, speed - grass.mu_roll * g * dt) / speed
            v2 = (v[0] * s, v[1] * s, 0.0)
            w = (-v2[1] / ball.radius, v2[0] / ball.radius, w[2])
        x, y, t = x + 0.5 * (v[0] + v2[0]) * dt, y + 0.5 * (v[1] + v2[1]) * dt, t + dt
        v = v2
        out.append((t, x, y, 0.0))
        if stop and stop(x, y):
            break
    return out
