"""Batched baseball flight model (gravity + drag + Magnus + spin decay), torch, Z-up world."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .constants import PhysicsSpec


@dataclass
class FlightResult:
    landing_pos: torch.Tensor  # (N, 3), z == ground height
    carry: torch.Tensor  # (N,) horizontal distance from launch point
    flight_time: torch.Tensor  # (N,)
    apex: torch.Tensor  # (N,) max height
    landed: torch.Tensor  # (N,) bool


def aero_acceleration(vel: torch.Tensor, omega: torch.Tensor, spec: PhysicsSpec, _consts: dict | None = None) -> torch.Tensor:
    """Drag + Magnus acceleration (no gravity). vel, omega: (N, 3) world frame."""
    c = _consts or _aero_consts(spec, vel)
    v_rel = vel - c["wind"]
    speed = v_rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w_mag = omega.norm(dim=-1, keepdim=True)

    cd = c["cd0"] + c["cd_per_radps"] * w_mag
    s = c["r"] * w_mag / speed
    cl = c["cl2"] * s / (c["cl0"] + c["cl1"] * s)

    drag = -c["k"] * cd * speed * v_rel
    # Magnus direction: omega_hat x v_hat (only the spin component perpendicular to v lifts)
    w_hat = omega / w_mag.clamp_min(1e-9)
    magnus = c["k"] * cl * speed * torch.cross(w_hat, v_rel, dim=-1)
    return drag + magnus


def _aero_consts(spec: PhysicsSpec, like: torch.Tensor) -> dict:
    ball, aero = spec.ball, spec.aero
    return dict(
        wind=torch.tensor(aero.wind, dtype=like.dtype, device=like.device),
        g=torch.tensor([0.0, 0.0, -spec.gravity], dtype=like.dtype, device=like.device),
        k=0.5 * aero.rho * ball.area / ball.mass,
        r=ball.radius,
        cd0=aero.cd0,
        cd_per_radps=aero.cd_spin * 60.0 / (2.0 * math.pi) / 1000.0,
        cl0=aero.cl0,
        cl1=aero.cl1,
        cl2=aero.cl2,
        decay=spec.aero.spin_decay_k / ball.radius,
    )


def spin_decay_rate(vel: torch.Tensor, spec: PhysicsSpec) -> torch.Tensor:
    """d(omega)/dt / omega, shape (N, 1)."""
    return -spec.aero.spin_decay_k * vel.norm(dim=-1, keepdim=True) / spec.ball.radius


def _deriv(vel, omega, spec, c):
    acc = aero_acceleration(vel, omega, spec, c) + c["g"]
    domega = -c["decay"] * vel.norm(dim=-1, keepdim=True) * omega
    return acc, domega


@torch.no_grad()
def simulate_flight(
    pos: torch.Tensor,
    vel: torch.Tensor,
    omega: torch.Tensor,
    spec: PhysicsSpec,
    ground_z: float = 0.0,
    dt: float = 0.005,
    t_max: float = 12.0,
) -> FlightResult:
    """Integrate until each ball crosses ground_z (RK4). Returns landing info per ball."""
    pos, vel, omega = pos.clone(), vel.clone(), omega.clone()
    n = pos.shape[0]
    g_vec = _aero_consts(spec, pos)
    start = pos.clone()
    landed = torch.zeros(n, dtype=torch.bool, device=pos.device)
    landing = pos.clone()
    t_land = torch.full((n,), t_max, dtype=pos.dtype, device=pos.device)
    apex = pos[:, 2].clone()
    t = 0.0
    steps = int(t_max / dt)
    for k in range(steps):
        a1, w1 = _deriv(vel, omega, spec, g_vec)
        a2, w2 = _deriv(vel + 0.5 * dt * a1, omega + 0.5 * dt * w1, spec, g_vec)
        a3, w3 = _deriv(vel + 0.5 * dt * a2, omega + 0.5 * dt * w2, spec, g_vec)
        a4, w4 = _deriv(vel + dt * a3, omega + dt * w3, spec, g_vec)
        dpos = dt / 6.0 * (vel + 2 * (vel + 0.5 * dt * a1) + 2 * (vel + 0.5 * dt * a2) + (vel + dt * a3))
        new_pos = pos + dpos
        vel = vel + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        omega = omega + dt / 6.0 * (w1 + 2 * w2 + 2 * w3 + w4)

        crossed = (~landed) & (new_pos[:, 2] <= ground_z) & (pos[:, 2] > ground_z)
        frac = ((pos[:, 2] - ground_z) / (pos[:, 2] - new_pos[:, 2]).clamp_min(1e-9)).unsqueeze(-1)
        hit = pos + frac * (new_pos - pos)
        landing = torch.where(crossed.unsqueeze(-1), hit, landing)
        t_land = torch.where(crossed, t + frac.squeeze(-1) * dt, t_land)
        landed = landed | crossed
        apex = torch.where(landed, apex, torch.maximum(apex, new_pos[:, 2]))
        pos = torch.where(landed.unsqueeze(-1), pos, new_pos)
        t += dt
        # host sync only every few steps (GPU friendly)
        if k % 10 == 9 and bool(landed.all()):
            break

    landing = torch.where(landed.unsqueeze(-1), landing, pos)
    landing[:, 2] = torch.where(landed, torch.full_like(landing[:, 2], ground_z), landing[:, 2])
    carry = (landing[:, :2] - start[:, :2]).norm(dim=-1)
    return FlightResult(landing_pos=landing, carry=carry, flight_time=t_land, apex=apex, landed=landed)


class GraphedFlight:
    """CUDA-graph version of `simulate_flight` for training (fixed capacity, ~ms per call).

    Kernel launch overhead dominates the small per-step RK4 math (especially under Windows WDDM),
    so a block of RK4 steps is captured once and replayed.
    """

    def __init__(self, capacity: int, spec: PhysicsSpec, device, dt: float = 0.02, block: int = 25,
                 ground_z: float = 0.0, t_max: float = 12.0):
        self.n, self.spec, self.dt, self.block, self.ground_z, self.t_max = capacity, spec, dt, block, ground_z, t_max
        f = dict(dtype=torch.float32, device=device)
        self.pos = torch.zeros(capacity, 3, **f)
        self.vel = torch.zeros(capacity, 3, **f)
        self.omega = torch.zeros(capacity, 3, **f)
        self.start = torch.zeros(capacity, 3, **f)
        self.landing = torch.zeros(capacity, 3, **f)
        self.t_land = torch.zeros(capacity, **f)
        self.apex = torch.zeros(capacity, **f)
        self.t = torch.zeros(1, **f)
        self.landed = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.c = _aero_consts(spec, self.pos)
        self.use_graph = torch.device(device).type == "cuda"
        self.graph = None
        if self.use_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._block()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._block()

    def _block(self):
        dt, gz, c, spec = self.dt, self.ground_z, self.c, self.spec
        for _ in range(self.block):
            pos, vel, omega = self.pos, self.vel, self.omega
            a1, w1 = _deriv(vel, omega, spec, c)
            a2, w2 = _deriv(vel + 0.5 * dt * a1, omega + 0.5 * dt * w1, spec, c)
            a3, w3 = _deriv(vel + 0.5 * dt * a2, omega + 0.5 * dt * w2, spec, c)
            a4, w4 = _deriv(vel + dt * a3, omega + dt * w3, spec, c)
            new_pos = pos + dt / 6.0 * (vel + 2 * (vel + 0.5 * dt * a1) + 2 * (vel + 0.5 * dt * a2) + (vel + dt * a3))
            new_vel = vel + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            new_omega = omega + dt / 6.0 * (w1 + 2 * w2 + 2 * w3 + w4)
            crossed = (~self.landed) & (new_pos[:, 2] <= gz) & (pos[:, 2] > gz)
            frac = ((pos[:, 2] - gz) / (pos[:, 2] - new_pos[:, 2]).clamp_min(1e-9)).unsqueeze(-1)
            self.landing.copy_(torch.where(crossed.unsqueeze(-1), pos + frac * (new_pos - pos), self.landing))
            self.t_land.copy_(torch.where(crossed, self.t + frac.squeeze(-1) * dt, self.t_land))
            self.landed.logical_or_(crossed)
            keep = self.landed.unsqueeze(-1)
            self.apex.copy_(torch.where(self.landed, self.apex, torch.maximum(self.apex, new_pos[:, 2])))
            self.pos.copy_(torch.where(keep, pos, new_pos))
            self.vel.copy_(torch.where(keep, vel, new_vel))
            self.omega.copy_(torch.where(keep, omega, new_omega))
            self.t.add_(dt)

    @torch.no_grad()
    def run(self, pos: torch.Tensor, vel: torch.Tensor, omega: torch.Tensor) -> FlightResult:
        m = pos.shape[0]
        assert m <= self.n
        self.pos.zero_()
        self.pos[:, 2] = self.ground_z - 1.0  # unused slots: already on the ground
        self.vel.zero_()
        self.omega.zero_()
        self.landed.fill_(True)
        self.landed[:m] = False
        self.pos[:m] = pos
        self.vel[:m] = vel
        self.omega[:m] = omega
        self.start.copy_(self.pos)
        self.landing.copy_(self.pos)
        self.apex.copy_(self.pos[:, 2])
        self.t_land.fill_(self.t_max)
        self.t.zero_()
        n_blocks = int(self.t_max / (self.dt * self.block)) + 1
        for _ in range(n_blocks):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._block()
            if bool(self.landed[:m].all()):
                break
        landed = self.landed[:m].clone()
        landing = torch.where(landed.unsqueeze(-1), self.landing[:m], self.pos[:m]).clone()
        landing[:, 2] = torch.where(landed, torch.full_like(landing[:, 2], self.ground_z), landing[:, 2])
        carry = (landing[:, :2] - self.start[:m, :2]).norm(dim=-1)
        return FlightResult(landing_pos=landing, carry=carry, flight_time=self.t_land[:m].clone(),
                            apex=self.apex[:m].clone(), landed=landed)


@torch.no_grad()
def simulate_to_plane(pos: torch.Tensor, vel: torch.Tensor, omega: torch.Tensor, spec: PhysicsSpec, plane_x: float,
                      dt: float = 0.02, t_max: float = 1.0):
    """Integrate (RK4, drag + Magnus) until each ball reaches x = plane_x (pitch toward +X).

    Returns (crossing position (N,3), speed at the plane (N,), time (N,), reached (N,) bool).
    """
    pos, vel, omega = pos.clone(), vel.clone(), omega.clone()
    n = pos.shape[0]
    c = _aero_consts(spec, pos)
    done = torch.zeros(n, dtype=torch.bool, device=pos.device)
    cross = pos.clone()
    speed = vel.norm(dim=-1)
    t_cross = torch.full((n,), t_max, dtype=pos.dtype, device=pos.device)
    t = 0.0
    for k in range(int(t_max / dt)):
        a1, w1 = _deriv(vel, omega, spec, c)
        a2, w2 = _deriv(vel + 0.5 * dt * a1, omega + 0.5 * dt * w1, spec, c)
        a3, w3 = _deriv(vel + 0.5 * dt * a2, omega + 0.5 * dt * w2, spec, c)
        a4, w4 = _deriv(vel + dt * a3, omega + dt * w3, spec, c)
        new_pos = pos + dt / 6.0 * (vel + 2 * (vel + 0.5 * dt * a1) + 2 * (vel + 0.5 * dt * a2) + (vel + dt * a3))
        new_vel = vel + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        omega = omega + dt / 6.0 * (w1 + 2 * w2 + 2 * w3 + w4)
        crossed = (~done) & (new_pos[:, 0] >= plane_x) & (pos[:, 0] < plane_x)
        frac = ((plane_x - pos[:, 0]) / (new_pos[:, 0] - pos[:, 0]).clamp_min(1e-9)).unsqueeze(-1)
        cross = torch.where(crossed.unsqueeze(-1), pos + frac * (new_pos - pos), cross)
        speed = torch.where(crossed, (vel + frac * (new_vel - vel)).norm(dim=-1), speed)
        t_cross = torch.where(crossed, t + frac.squeeze(-1) * dt, t_cross)
        done = done | crossed | (new_pos[:, 2] < 0.0)  # hit the ground first: stop (not reached)
        pos, vel = torch.where(done.unsqueeze(-1), pos, new_pos), torch.where(done.unsqueeze(-1), vel, new_vel)
        t += dt
        if k % 5 == 4 and bool(done.all()):
            break
    reached = cross[:, 0] >= plane_x - 1e-6
    # balls that hit the ground first: report where they stopped (lets callers measure the miss)
    cross = torch.where(reached.unsqueeze(-1), cross, pos)
    return cross, speed, t_cross, reached


def launch_state(
    exit_speed: torch.Tensor,
    launch_angle_deg: torch.Tensor,
    backspin_rpm: torch.Tensor,
    spray_deg: torch.Tensor | None = None,
    sidespin_rpm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity / angular velocity for a launch toward +X (center field), Z up.

    Backspin makes the ball rise: for v along +X, backspin is omega along -Y... check:
    Magnus ~ omega x v; (-Y) x (+X) = +Z  -> lift. So backspin => omega = -Y * w.
    """
    la = torch.deg2rad(launch_angle_deg)
    spray = torch.deg2rad(spray_deg) if spray_deg is not None else torch.zeros_like(la)
    v = torch.stack(
        [
            exit_speed * torch.cos(la) * torch.cos(spray),
            exit_speed * torch.cos(la) * torch.sin(spray),
            exit_speed * torch.sin(la),
        ],
        dim=-1,
    )
    w_back = backspin_rpm * 2.0 * math.pi / 60.0
    # backspin axis: horizontal, perpendicular to the direction of travel, pointing to the batter's left
    axis = torch.stack([torch.sin(spray), -torch.cos(spray), torch.zeros_like(spray)], dim=-1)
    omega = axis * w_back.unsqueeze(-1)
    if sidespin_rpm is not None:
        omega = omega + torch.stack(
            [torch.zeros_like(la), torch.zeros_like(la), sidespin_rpm * 2.0 * math.pi / 60.0], dim=-1
        )
    return v, omega
