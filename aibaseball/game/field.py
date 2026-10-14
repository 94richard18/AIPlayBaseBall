"""Field geometry for the game engine (one-month plan, W3).

Frame: home plate at the origin, +X toward center field, +Y toward the third-base side, Z up - the frame of
physics.launch_state (spray > 0 = toward left field). MLB base paths; the fence is a smooth curve from the foul
poles to center field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BASE_PATH = 27.432  # m (90 ft)
MOUND_DIST = 18.44  # m, rubber to the plate
FOUL_ANGLE_DEG = 45.0
_D = BASE_PATH / math.sqrt(2.0)
BASES = ((0.0, 0.0), (_D, -_D), (2.0 * _D, 0.0), (_D, _D))  # home, 1B, 2B, 3B


def spray_deg(x: float, y: float) -> float:
    """Angle from the center-field line, + toward left field."""
    return math.degrees(math.atan2(y, x))


def polar(dist: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return dist * math.cos(a), dist * math.sin(a)


POSITIONS = ("P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF")
ALIGNMENT = {  # standard depths (m from the plate, deg from the center-field line)
    "P": (MOUND_DIST, 0.0),
    "C": (-1.0, 0.0),
    "1B": polar(32.0, -38.0),
    "2B": polar(44.0, -14.0),
    "3B": polar(32.0, 38.0),
    "SS": polar(44.0, 14.0),
    "LF": polar(88.0, 28.0),
    "CF": polar(96.0, 0.0),
    "RF": polar(88.0, -28.0),
}
OUTFIELD = ("LF", "CF", "RF")


@dataclass(frozen=True)
class FieldSpec:
    fence_line_m: float = 100.0  # at the foul poles
    fence_center_m: float = 122.0
    fence_height_m: float = 2.5
    foul_room_m: float = 15.0  # foul territory where a fly can still be caught (stands beyond)

    def fence_distance(self, angle_deg: float) -> float:
        a = min(abs(angle_deg), FOUL_ANGLE_DEG)
        return self.fence_line_m + (self.fence_center_m - self.fence_line_m) * math.cos(math.radians(2.0 * a))

    def is_fair(self, x: float, y: float) -> bool:
        return x > 0.0 and abs(spray_deg(x, y)) <= FOUL_ANGLE_DEG

    def foul_depth(self, x: float, y: float) -> float:
        """How far into foul territory a point is (0 when fair): distance past the nearer foul line."""
        if self.is_fair(x, y):
            return 0.0
        a = abs(spray_deg(x, y))
        d = math.hypot(x, y)
        return d * math.sin(math.radians(a - FOUL_ANGLE_DEG)) if a <= 135.0 else d
