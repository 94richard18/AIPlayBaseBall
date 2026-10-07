"""MLB-legal one-piece maple bat (34 in / 32 oz), axisymmetric along its local +X axis.

Local frame: origin at the knob end, +X toward the barrel end. Mass properties are
integrated from the radius profile with a uniform density chosen to hit the target mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import INCH, OZ

# (x from knob end [m], radius [m]); linear interpolation between stations.
# Max diameter 2.61 in (MLB rule) -> we use 2.55 in barrel.
DEFAULT_PROFILE: tuple[tuple[float, float], ...] = (
    (0.000, 0.0200),  # knob
    (0.012, 0.0200),
    (0.022, 0.0125),  # handle starts
    (0.300, 0.0135),
    (0.450, 0.0215),  # taper
    (0.600, 0.0318),
    (0.640, 0.0324),  # barrel
    (0.840, 0.0324),
    (0.858, 0.0290),  # end cup rounding
    (0.8636, 0.0240),
)


@dataclass
class BatSpec:
    length: float = 34.0 * INCH  # 0.8636 m
    mass: float = 32.0 * OZ  # 0.907 kg
    profile: tuple[tuple[float, float], ...] = DEFAULT_PROFILE
    sweet_spot_from_tip: float = 6.0 * INCH  # ~ node of the fundamental bending mode
    # Grip stations (distance from knob end): bottom hand next to the knob, top hand above it.
    bottom_hand_x: float = 0.060
    top_hand_x: float = 0.150
    n_stations: int = 400
    _props: dict = field(default=None, init=False, repr=False)

    def radius_at(self, x):
        xs, rs = zip(*self.profile)
        return np.interp(x, xs, rs, left=0.0, right=0.0)

    @property
    def sweet_spot_x(self) -> float:
        return self.length - self.sweet_spot_from_tip

    def mass_properties(self) -> dict:
        """Returns density, CM (along x), and inertia about CM (axial, transverse)."""
        if self._props is not None:
            return self._props
        x = np.linspace(0.0, self.length, self.n_stations + 1)
        xm = 0.5 * (x[1:] + x[:-1])
        dx = np.diff(x)
        r = self.radius_at(xm)
        vol = math.pi * r**2 * dx
        density = self.mass / vol.sum()
        dm = density * vol
        cm = float((dm * xm).sum() / self.mass)
        i_axial = float((0.5 * dm * r**2).sum())
        # thin disks: about own diameter m r^2/4 + parallel axis
        i_trans = float((dm * (r**2 / 4.0 + (xm - cm) ** 2)).sum())
        i_knob6 = i_trans + self.mass * (cm - 6.0 * INCH) ** 2  # MOI about the 6-inch point (industry metric)
        self._props = dict(
            density=float(density),
            volume=float(vol.sum()),
            cm_x=cm,
            inertia_axial=i_axial,
            inertia_transverse=i_trans,
            inertia_6in=i_knob6,
        )
        return self._props

    def collision_capsules(self, n: int = 6) -> list[tuple[float, float, float]]:
        """Approximate the bat by n capsules along x: list of (x_start, x_end, radius)."""
        bounds = np.linspace(0.02, self.length - 0.01, n + 1)
        caps = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            xs = np.linspace(a, b, 8)
            caps.append((float(a), float(b), float(self.radius_at(xs).max())))
        return caps

    def mesh(self, n_seg: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Surface of revolution: (vertices (V,3), triangle indices (F,3)), local frame."""
        xs, rs = zip(*self.profile)
        xs, rs = np.array(xs), np.array(rs)
        ang = np.linspace(0, 2 * math.pi, n_seg, endpoint=False)
        verts = [[0.0, 0.0, 0.0]]
        for x, r in zip(xs, rs):
            for a in ang:
                verts.append([x, r * math.cos(a), r * math.sin(a)])
        verts.append([xs[-1], 0.0, 0.0])
        verts = np.array(verts)
        faces = []
        n_ring = len(xs)
        for j in range(n_seg):  # knob cap
            faces.append([0, 1 + (j + 1) % n_seg, 1 + j])
        for i in range(n_ring - 1):
            for j in range(n_seg):
                a = 1 + i * n_seg + j
                b = 1 + i * n_seg + (j + 1) % n_seg
                c = a + n_seg
                d = b + n_seg
                faces += [[a, b, d], [a, d, c]]
        last = len(verts) - 1
        base = 1 + (n_ring - 1) * n_seg
        for j in range(n_seg):  # end cap
            faces.append([last, base + j, base + (j + 1) % n_seg])
        return verts, np.array(faces, dtype=np.int64)
