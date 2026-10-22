"""Pitcher / batter interfaces of the game engine, and statistical stand-ins.

The RL pitcher and batter (Isaac Sim) plug in later through the same two calls; until then the stand-ins draw
pitches and batted balls from rough MLB-like distributions so the engine can play whole games.
Mirror match: one pitcher agent and one batter agent serve both teams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .state import GameState


@dataclass(frozen=True)
class StrikeZone:
    half_width_m: float = 0.2159 + 0.0366  # half the plate + the ball's radius
    bottom_m: float = 0.50  # 1.85 m batter: hollow of the knee
    top_m: float = 1.10  # midpoint between shoulders and belt

    def contains(self, y: float, z: float) -> bool:
        return abs(y) <= self.half_width_m and self.bottom_m <= z <= self.top_m


@dataclass(frozen=True)
class Pitch:
    speed_mps: float  # at release
    plate_y: float  # m, where the ball crosses the front of the plate (+ = third-base side)
    plate_z: float  # m above the ground


@dataclass(frozen=True)
class BattedBall:
    exit_speed_mps: float
    launch_deg: float
    spray_deg: float  # + toward left field
    backspin_rpm: float  # < 0 = topspin


@dataclass(frozen=True)
class Swing:
    kind: str  # "take" | "miss" | "contact"
    ball: BattedBall | None = None


class PitcherAgent(Protocol):
    def pitch(self, state: GameState, rng: random.Random) -> Pitch: ...


class BatterAgent(Protocol):
    def swing(self, state: GameState, pitch: Pitch, zone: StrikeZone, rng: random.Random) -> Swing: ...


@dataclass
class StatPitcher:
    """Aims at the middle of the zone with a Gaussian miss (about 60% strikes)."""

    speed_mps: float = 32.0  # ~115 km/h, the human-strength pitcher so far
    speed_sd: float = 1.5
    miss_sd_m: float = 0.22

    def pitch(self, state: GameState, rng: random.Random) -> Pitch:
        return Pitch(rng.gauss(self.speed_mps, self.speed_sd), rng.gauss(0.0, self.miss_sd_m), rng.gauss(0.80, self.miss_sd_m))


@dataclass
class TablePitcher:
    """Replays the RL pitcher's evaluated pitches: rows of a CSV with columns speed_mps, plate_y, plate_z, crossed
    (crossed = 0: the ball never reached the plate - a ball for the count, placed far outside the zone).
    The pitching env's frame has the pitcher facing +x, so its +y is the pitcher's left = the first-base side;
    the engine's plate_y is + toward third base, hence the sign flip."""

    rows: list

    @classmethod
    def from_csv(cls, path: str) -> "TablePitcher":
        import csv

        with open(path, newline="") as f:
            rows = [(float(r["speed_mps"]), -float(r["plate_y"]), float(r["plate_z"]), int(float(r.get("crossed", 1))))
                    for r in csv.DictReader(f)]
        assert rows, f"no pitches in {path}"
        return cls(rows)

    def pitch(self, state: GameState, rng: random.Random) -> Pitch:
        speed, y, z, crossed = rng.choice(self.rows)
        return Pitch(speed, y, z) if crossed else Pitch(speed, 2.0, 0.0)


@dataclass
class FallbackPitcher:
    """The game's fallback pitch (user rule, adopted 2026-10-21 after hk): the RL pitcher's motion is played back and
    the ball after the release is a simulated fastball at the batter's training speeds - 88-112 km/h at release
    (build_library, drag + Magnus), 80-102 km/h at the plate. Location: StatPitcher's Gaussian miss around mid-zone."""

    speed_kmh: tuple = (88.0, 112.0)
    miss_sd_m: float = 0.22

    def pitch(self, state: GameState, rng: random.Random) -> Pitch:
        return Pitch(rng.uniform(*self.speed_kmh) / 3.6, rng.gauss(0.0, self.miss_sd_m), rng.gauss(0.80, self.miss_sd_m))


@dataclass
class TableBatter:
    """Replays the RL batter's evaluated batted balls (scripts/showcase.py --eval_envs -> videos/showcase_stats.json
    "swings": exit_velo_mps, launch_deg, spray_deg, backspin_rpm). Swing decisions stay StatBatter's; contact uses the
    measured hit rate (diag_bat_contact) in the zone, scaled by StatBatter's out/in-zone contact ratio outside it."""

    rows: list
    hit_rate: float
    swing_in_zone: float = 0.68
    swing_out_zone: float = 0.28
    out_zone_contact_scale: float = 0.52 / 0.78

    @classmethod
    def from_json(cls, path: str, hit_rate: float) -> "TableBatter":
        import json

        with open(path, encoding="utf-8") as f:
            swings = json.load(f)["swings"]
        rows = [(float(s["exit_velo_mps"]), float(s["launch_deg"]), float(s["spray_deg"]), float(s["backspin_rpm"]))
                for s in swings]
        assert rows, f"no batted balls in {path}"
        return cls(rows, hit_rate)

    def swing(self, state: GameState, pitch: Pitch, zone: StrikeZone, rng: random.Random) -> Swing:
        in_zone = zone.contains(pitch.plate_y, pitch.plate_z)
        if abs(pitch.plate_y) > 1.0:
            return Swing("take")
        if rng.random() >= (self.swing_in_zone if in_zone else self.swing_out_zone):
            return Swing("take")
        if rng.random() >= self.hit_rate * (1.0 if in_zone else self.out_zone_contact_scale):
            return Swing("miss")
        ev, la, spray, spin = rng.choice(self.rows)
        return Swing("contact", BattedBall(ev, la, spray, spin))


@dataclass
class StatBatter:
    """Swing / contact rates and batted-ball spread loosely after MLB averages."""

    swing_in_zone: float = 0.68
    swing_out_zone: float = 0.28
    contact_in_zone: float = 0.78
    contact_out_zone: float = 0.52
    exit_speed_mps: float = 38.0
    exit_speed_sd: float = 7.0
    launch_deg: float = 12.0
    launch_sd: float = 24.0
    spray_sd: float = 32.0

    def swing(self, state: GameState, pitch: Pitch, zone: StrikeZone, rng: random.Random) -> Swing:
        in_zone = zone.contains(pitch.plate_y, pitch.plate_z)
        if abs(pitch.plate_y) > 1.0:  # nowhere near the plate
            return Swing("take")
        if rng.random() >= (self.swing_in_zone if in_zone else self.swing_out_zone):
            return Swing("take")
        if rng.random() >= (self.contact_in_zone if in_zone else self.contact_out_zone):
            return Swing("miss")
        la = rng.gauss(self.launch_deg, self.launch_sd)
        spin = min(3500.0, 1200.0 + 40.0 * la) if la > 0.0 else -800.0
        ev = min(50.0, max(8.0, rng.gauss(self.exit_speed_mps, self.exit_speed_sd)))
        return Swing("contact", BattedBall(ev, la, rng.gauss(0.0, self.spray_sd), spin))
