"""ONE OUTS (甲斐谷忍): one pitcher against one batter for one plate appearance, no fielders, a wall with a painted strike
zone behind the plate (rules as given by the user, 2026-10-23).

Batter wins: a walk, a fair ball whose first bounce is beyond the infield, or a home run.
Pitcher wins: a strikeout, or a fair ball whose first bounce is in the infield (no fielders - it stands for the out).
Hit by pitch: the batter wins (user, 2026-10-23). The right-handed batter's body is a box on the third-base side of the
plate (BODY_Y_M / BODY_Z_M at the plate plane); a pitch crossing inside it hits him, swing or not.
Not in the given rules, default here: a foul is a strike except with two strikes (never the third).
Infield: within 95 ft (28.96 m) of the pitcher's rubber - the edge of the MLB infield dirt.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..physics import PhysicsSpec
from ..physics.grass import GrassSpec
from .agents import BatterAgent, PitcherAgent, StrikeZone
from .field import MOUND_DIST, FieldSpec
from .fielding import FieldingSpec, ball_path
from .state import GameState

INFIELD_RADIUS_M = 28.96  # 95 ft from the rubber
BODY_Y_M = (0.50, 0.90)  # m from the plate centre toward third base: a right-handed batter's body in his stance
BODY_Z_M = (0.20, 1.70)


@dataclass
class OneOutsResult:
    winner: str  # "batter" | "pitcher"
    how: str  # "walk" | "hit_by_pitch" | "outfield_hit" | "home_run" | "strikeout" | "infield_ball"
    pitches: list  # one dict per pitch


def first_bounce(ball, fld: FieldSpec, fs: FieldingSpec, grass: GrassSpec, phys: PhysicsSpec):
    """(x, y) of the first touch of the ground, or None for a home run / a ball off the wall."""
    path, end = ball_path(ball, fld, grass, fs, phys)
    if end != "landed":
        return end, None
    land = [s for s in path if s[4]][-1]
    return end, (land[1], land[2])


def play_one_outs(pitcher: PitcherAgent, batter: BatterAgent, rng: random.Random, zone: StrikeZone | None = None,
                  fld: FieldSpec | None = None) -> OneOutsResult:
    zone, fld = zone or StrikeZone(), fld or FieldSpec()
    fs, grass, phys = FieldingSpec(), GrassSpec(), PhysicsSpec()
    state = GameState()  # only the count is used (the agents' interface takes a state)
    balls = strikes = 0
    log = []
    while True:
        state.balls, state.strikes = balls, strikes
        p = pitcher.pitch(state, rng)
        if BODY_Y_M[0] <= p.plate_y <= BODY_Y_M[1] and BODY_Z_M[0] <= p.plate_z <= BODY_Z_M[1]:
            log.append(dict(balls=balls, strikes=strikes, pitch=p, swing="hit_by_pitch"))
            return OneOutsResult("batter", "hit_by_pitch", log)
        sw = batter.swing(state, p, zone, rng)
        ev = dict(balls=balls, strikes=strikes, pitch=p, swing=sw.kind)
        log.append(ev)
        if sw.kind == "take":
            if zone.contains(p.plate_y, p.plate_z):  # hits the painted box on the wall
                strikes += 1
            else:
                balls += 1
        elif sw.kind == "miss":
            strikes += 1
        else:
            end, land = first_bounce(sw.ball, fld, fs, grass, phys)
            ev["land"], ev["ball"], ev["end"] = land, sw.ball, end
            if end in ("home_run", "wall"):
                return OneOutsResult("batter", "home_run" if end == "home_run" else "outfield_hit", log)
            x, y = land
            if not fld.is_fair(x, y):
                strikes = min(strikes + 1, 2)  # foul: a strike, never the third
                continue
            if math.hypot(x - MOUND_DIST, y) <= INFIELD_RADIUS_M or math.hypot(x, y) <= MOUND_DIST:
                return OneOutsResult("pitcher", "infield_ball", log)
            return OneOutsResult("batter", "outfield_hit", log)
        if balls >= 4:
            return OneOutsResult("batter", "walk", log)
        if strikes >= 3:
            return OneOutsResult("pitcher", "strikeout", log)
