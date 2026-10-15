"""Plays a whole game pitch by pitch (one-month plan, W3) and keeps an event log for the W4 replay."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..physics import PhysicsSpec
from .agents import BatterAgent, PitcherAgent, StatBatter, StatPitcher, StrikeZone
from .field import FieldSpec
from .fielding import FieldingSpec, GrassSpec, resolve
from .state import GameState


@dataclass
class Game:
    pitcher: PitcherAgent = field(default_factory=StatPitcher)  # mirror match: both teams
    batter: BatterAgent = field(default_factory=StatBatter)
    seed: int = 0
    innings: int = 9
    field_spec: FieldSpec = field(default_factory=FieldSpec)
    zone: StrikeZone = field(default_factory=StrikeZone)
    fielding: FieldingSpec = field(default_factory=FieldingSpec)
    grass: GrassSpec = field(default_factory=GrassSpec)
    phys: PhysicsSpec = field(default_factory=PhysicsSpec)

    def __post_init__(self):
        self.state = GameState(innings=self.innings)
        self.rng = random.Random(self.seed)
        self.log: list[dict] = []

    def pitch(self) -> dict:
        s = self.state
        ev = dict(inning=s.inning, top=s.top, batter=s.batter, balls=s.balls, strikes=s.strikes, outs=s.outs,
                  bases=tuple(b is not None for b in s.bases), score=tuple(s.score))
        p = self.pitcher.pitch(s, self.rng)
        sw = self.batter.swing(s, p, self.zone, self.rng)
        ev.update(pitch=p, swing=sw.kind, play=None)
        if sw.kind == "take":
            ev["result"] = s.strike() if self.zone.contains(p.plate_y, p.plate_z) else s.ball()
        elif sw.kind == "miss":
            ev["result"] = s.strike()
        else:
            play = resolve(sw.ball, ev["bases"], s.outs, self.field_spec, self.fielding, self.grass, self.phys)
            ev["play"] = play
            if play.kind == "foul":
                ev["result"] = s.strike(foul=True)
            else:
                ev["runs"] = s.apply_play(play.batter_to, play.runner_to, play.outs)
                ev["result"] = play.kind
        self.log.append(ev)
        return ev

    def play(self, max_pitches: int = 2000) -> GameState:
        while not self.state.final and len(self.log) < max_pitches:
            self.pitch()
        return self.state


def line_score(s: GameState) -> str:
    n = max(len(s.line[0]), len(s.line[1]))
    cells = lambda row: " ".join(f"{r:>2}" for r in row) + "   " * (n - len(row))  # noqa: E731
    head = "      " + " ".join(f"{i + 1:>2}" for i in range(n)) + "    R"
    return "\n".join([head, f"Away  {cells(s.line[0])}  {s.score[0]:>3}", f"Home  {cells(s.line[1])}  {s.score[1]:>3}"])
