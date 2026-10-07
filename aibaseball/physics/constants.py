"""Physical constants for MLB baseball simulation (SI units).

Sources:
- MLB Official Baseball Rules 3.01: ball weight 5–5.25 oz, circumference 9–9.25 in.
- MLB Official Baseball Rules 3.02: bat max diameter 2.61 in, max length 42 in.
- Alan M. Nathan, "Trajectory Calculator" (baseball.physics.illinois.edu): drag / lift fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OZ = 0.028349523125  # kg
INCH = 0.0254  # m
MPH = 0.44704  # m/s
FT = 0.3048  # m

GRAVITY = 9.80665  # m/s^2

TARGET_DISTANCE = 150.3  # m, task goal (carry distance, horizontal from tee)


@dataclass(frozen=True)
class BallSpec:
    """MLB baseball. Midpoint of the legal range."""

    mass: float = 5.125 * OZ  # 0.1453 kg
    circumference: float = 9.125 * INCH  # 0.2318 m
    # Moment of inertia factor I = k m r^2 (cork/rubber core + yarn; ~0.4 measured).
    inertia_factor: float = 0.4

    @property
    def radius(self) -> float:
        return self.circumference / (2.0 * math.pi)

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    @property
    def inertia(self) -> float:
        return self.inertia_factor * self.mass * self.radius**2


def air_density(temp_c: float = 21.0, pressure_pa: float = 101325.0, rel_humidity: float = 0.5) -> float:
    """Moist-air density from the ideal gas law (Tetens saturation pressure)."""
    t_k = temp_c + 273.15
    p_sat = 610.78 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    p_v = rel_humidity * p_sat
    p_d = pressure_pa - p_v
    return p_d / (287.058 * t_k) + p_v / (461.495 * t_k)


@dataclass(frozen=True)
class AeroSpec:
    """Indoor stadium: still air, 21 C, sea-level pressure, 50% RH (~1.195 kg/m^3).

    Drag / lift coefficient fits from A. M. Nathan's trajectory calculator:
        Cd = cd0 + cd_spin * (spin_rpm / 1000)
        Cl = cl2 * S / (cl0 + cl1 * S),   S = r * omega / v
    Spin decay: d(omega)/dt = -omega * spin_decay_k * v / r.
    """

    temp_c: float = 21.0
    pressure_pa: float = 101325.0
    rel_humidity: float = 0.5
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)  # indoor: no wind
    cd0: float = 0.3008
    cd_spin: float = 0.0292
    cl0: float = 0.583
    cl1: float = 2.333
    cl2: float = 1.120
    spin_decay_k: float = 0.00002

    @property
    def rho(self) -> float:
        return air_density(self.temp_c, self.pressure_pa, self.rel_humidity)


@dataclass(frozen=True)
class ImpactSpec:
    """Ball–bat collision parameters (wood bat, MLB ball).

    COR vs. normal relative speed: MLB ball spec is COR 0.546 +/- 0.032 at 58 mph against
    a rigid wall; it falls roughly linearly with speed (~0.50 at 100 mph). Off the sweet spot
    energy goes into bat vibration, modelled as a Gaussian falloff of the effective COR.
    """

    cor_ref: float = 0.546
    cor_ref_speed: float = 58.0 * MPH
    cor_slope: float = -0.0011 / MPH  # per m/s
    cor_min: float = 0.30
    cor_max: float = 0.58
    vib_loss_max: float = 0.35  # fraction of COR lost far from the sweet spot
    vib_width: float = 0.09  # m
    friction: float = 0.50  # ball–wood sliding friction
    tangential_cor: float = 0.0  # 0 => ball ends contact rolling (no tangential rebound)


@dataclass(frozen=True)
class PhysicsSpec:
    ball: BallSpec = field(default_factory=BallSpec)
    aero: AeroSpec = field(default_factory=AeroSpec)
    impact: ImpactSpec = field(default_factory=ImpactSpec)
    gravity: float = GRAVITY
