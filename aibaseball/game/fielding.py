"""Batted-ball resolution with scripted fielders (one-month plan, W3).

The ball flies on the physics flight model (drag + Magnus), then bounces, slides and rolls on the grass model
(physics/grass.py: friction and spin at each bounce). Every fielder runs straight at the ball after a reaction time;
the first one who can reach it fields it - in the air it is a catch. Base running compares each runner's time to a
base with the throw's time there. No errors, double plays or tag-ups other than a sacrifice fly from third.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..physics import PhysicsSpec, aero_acceleration, launch_state
from ..physics.grass import GrassSpec, ground_path
from .agents import BattedBall
from .field import ALIGNMENT, BASE_PATH, BASES, OUTFIELD, FieldSpec, spray_deg

T = torch.float64


@dataclass(frozen=True)
class FieldingSpec:
    reaction_s: float = 0.4
    accel_mps2: float = 5.0  # fielders build up to their top speed
    speed_mps: dict = field(default_factory=lambda: {
        "P": 5.5, "C": 5.5, "1B": 6.5, "2B": 6.5, "3B": 6.5, "SS": 6.5, "LF": 7.5, "CF": 7.5, "RF": 7.5})
    reach_m: float = 1.5  # glove reach
    catch_height_m: float = 2.4
    transfer_s: float = 0.7  # glove to release
    throw_mps: float = 30.0  # average speed over the throw
    runner_mps: float = 7.5
    batter_start_s: float = 0.6  # end of the swing to the first stride
    turn_s: float = 0.4  # per base rounded
    safe_margin_s: float = 0.3  # the runner only tries an extra base with this much time to spare
    sac_fly_m: float = 60.0  # a catch at least this deep scores the runner from third
    contact_height_m: float = 0.9
    dt: float = 0.02


@dataclass
class Play:
    kind: str  # foul | fly_out | ground_out | single | double | triple | home_run
    batter_to: int = 0  # 0 = out, 1-3 = base, 4 = scored
    runner_to: tuple = (None, None, None)  # destination of the runner on 1B, 2B, 3B
    outs: int = 0
    fielder: str | None = None
    t_fielded: float | None = None
    at: tuple | None = None  # where the ball was fielded / left the park (x, y, z)
    path: list = field(default_factory=list)  # (t, x, y, z, in_flight) samples, for the replay


def ball_path(ball: BattedBall, fld: FieldSpec, grass: GrassSpec, fs: FieldingSpec, phys: PhysicsSpec):
    """Samples (t, x, y, z, in_flight) from contact until the ball stops, and how the flight ended:
    'landed' | 'wall' | 'home_run'. Spin is held constant through the flight."""
    one = lambda v: torch.tensor([v], dtype=T)  # noqa: E731
    vel, omega = launch_state(one(ball.exit_speed_mps), one(ball.launch_deg), one(ball.backspin_rpm), one(ball.spray_deg))
    g = torch.tensor([0.0, 0.0, -phys.gravity], dtype=T)
    acc = lambda v: aero_acceleration(v, omega, phys) + g  # noqa: E731
    pos, dt, t = torch.tensor([[0.0, 0.0, fs.contact_height_m]], dtype=T), fs.dt, 0.0
    out = [(0.0, 0.0, 0.0, fs.contact_height_m, True)]
    while t < 12.0:
        a1 = acc(vel)
        a2 = acc(vel + 0.5 * dt * a1)
        a3 = acc(vel + 0.5 * dt * a2)
        a4 = acc(vel + dt * a3)
        new = pos + dt / 6.0 * (vel + 2.0 * (vel + 0.5 * dt * a1) + 2.0 * (vel + 0.5 * dt * a2) + (vel + dt * a3))
        vel = vel + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        t += dt
        x, y, z = new[0].tolist()
        if fld.is_fair(x, y) and math.hypot(x, y) >= fld.fence_distance(spray_deg(x, y)):
            if z >= fld.fence_height_m:
                out.append((t, x, y, z, True))
                return out, "home_run"
            s = (fld.fence_distance(spray_deg(x, y)) - 0.5) / math.hypot(x, y)  # off the wall, dead at its foot
            out.append((t + 0.3, x * s, y * s, 0.0, False))
            return out, "wall"
        if z <= 0.0:
            z0 = pos[0, 2].item()
            f = z0 / (z0 - z)
            t -= dt * (1.0 - f)
            x, y = (pos[0, :2] + f * (new[0, :2] - pos[0, :2])).tolist()
            out.append((t, x, y, 0.0, True))
            break
        out.append((t, x, y, z, True))
        pos = new

    def at_wall(px, py):
        return fld.is_fair(px, py) and math.hypot(px, py) >= fld.fence_distance(spray_deg(px, py)) - 0.5

    rest = ground_path(t, x, y, tuple(vel[0].tolist()), tuple(omega[0].tolist()), grass, phys.ball, phys.gravity, dt, at_wall)
    return out + [(*s, False) for s in rest], "landed"


def _run_time(fs: FieldingSpec, pos: str, d: float) -> float:
    v, a = fs.speed_mps[pos], fs.accel_mps2
    d_acc = v * v / (2.0 * a)
    return math.sqrt(2.0 * d / a) if d <= d_acc else v / a + (d - d_acc) / v


def _arrival(fs: FieldingSpec, pos: str, x: float, y: float) -> float:
    fx, fy = ALIGNMENT[pos]
    return fs.reaction_s + _run_time(fs, pos, max(0.0, math.hypot(x - fx, y - fy) - fs.reach_m))


def resolve(ball: BattedBall, occupied: tuple, outs: int, fld: FieldSpec | None = None, fs: FieldingSpec | None = None,
            grass: GrassSpec | None = None, phys: PhysicsSpec | None = None) -> Play:
    """The play a batted ball makes. occupied: (runner on 1B, 2B, 3B) as bools; outs before the play."""
    fld, fs, grass, phys = fld or FieldSpec(), fs or FieldingSpec(), grass or GrassSpec(), phys or PhysicsSpec()
    path, end = ball_path(ball, fld, grass, fs, phys)
    land = [s for s in path if s[4]][-1] if end == "landed" else None  # first touch of the ground
    foul = land is not None and not fld.is_fair(land[1], land[2])
    hold = tuple(i + 1 if occupied[i] else None for i in range(3))

    fielded = None  # (t, x, y, z, in_flight, position)
    for s in path:
        t, x, y, z, flying = s
        if z > fs.catch_height_m:
            continue
        if flying and fld.foul_depth(x, y) > fld.foul_room_m:
            continue  # into the stands
        if not flying and foul:
            break  # a foul ball once it touches the ground
        best = min((_arrival(fs, p, x, y), p) for p in ALIGNMENT)
        if best[0] <= t:
            fielded = (t, x, y, z, flying, best[1])
            break

    if fielded and fielded[4]:  # caught in the air
        t, x, y, z, _, pos = fielded
        to = list(hold)
        if outs + 1 < 3 and occupied[2] and math.hypot(x, y) >= fs.sac_fly_m:
            to[2] = 4
        return Play("fly_out", 0, tuple(to), 1, pos, t, (x, y, z), path)
    if end == "home_run":
        return Play("home_run", 4, tuple(4 if o else None for o in occupied), 0, None, None, path[-1][1:4], path)
    if foul:
        return Play("foul", path=path)
    if fielded is None:  # nobody got there while it moved: first to the ball at rest
        t_end, x, y = path[-1][0], path[-1][1], path[-1][2]
        arr, pos = min((_arrival(fs, p, x, y), p) for p in ALIGNMENT)
        fielded = (max(t_end, arr), x, y, 0.0, False, pos)
    t_f, x, y, z, _, pos = fielded

    def run_t(k):
        return fs.batter_start_s + k * BASE_PATH / fs.runner_mps + (k - 1) * fs.turn_s

    def throw_t(k):
        bx, by = BASES[k % 4]
        d = math.hypot(x - bx, y - by)
        return t_f + min(fs.transfer_s + d / fs.throw_mps, _run_time(fs, pos, max(0.0, d - fs.reach_m)))

    if throw_t(1) < run_t(1):  # thrown out at first; forced runners move up
        forced = [occupied[0], occupied[0] and occupied[1], occupied[0] and occupied[1] and occupied[2]]
        to = tuple(None if not occupied[i] else i + 2 if forced[i] else i + 1 for i in range(3))
        return Play("ground_out", 0, to, 1, pos, t_f, (x, y, z), path)
    k = 1
    for kk in (2, 3, 4):
        if run_t(kk) + fs.safe_margin_s < throw_t(kk):
            k = kk
        else:
            break
    extra = lambda b: 1 if b == 2 and k == 1 and pos in OUTFIELD else 0  # noqa: E731
    to = tuple(min(4, i + 1 + k + extra(i + 1)) if occupied[i] else None for i in range(3))
    kind = ("single", "double", "triple", "home_run")[k - 1]
    return Play(kind, k, to, 0, pos, t_f, (x, y, z), path)
