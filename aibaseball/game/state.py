"""Game state and the scoring rules (simplified MLB: no steals, bunts, pickoffs, challenges, hit-by-pitch,
errors or double plays; runs on a third out at a base never count)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameState:
    innings: int = 9
    max_innings: int = 15  # a tie after this ends the game
    scoreless_tie_ends: bool = True  # 0-0 after `innings`: the game ends as a tie (no extra innings)
    inning: int = 1
    top: bool = True
    outs: int = 0
    balls: int = 0
    strikes: int = 0
    bases: list = field(default_factory=lambda: [None, None, None])  # lineup slot on 1B, 2B, 3B
    score: list = field(default_factory=lambda: [0, 0])  # away, home
    line: list = field(default_factory=lambda: [[0], []])  # runs per half inning, per team
    next_batter: list = field(default_factory=lambda: [0, 0])  # lineup slot, per team
    final: bool = False

    @property
    def batting(self) -> int:
        return 0 if self.top else 1

    @property
    def batter(self) -> int:
        return self.next_batter[self.batting]

    # --- pitch-level rules -------------------------------------------------------------------------------
    def ball(self) -> str:
        self.balls += 1
        if self.balls < 4:
            return "ball"
        self.walk()
        return "walk"

    def strike(self, foul: bool = False) -> str:
        if foul and self.strikes == 2:
            return "foul"
        self.strikes += 1
        if self.strikes < 3:
            return "foul" if foul else "strike"
        self.end_pa()
        self.add_outs(1)
        return "strikeout"

    def walk(self) -> None:
        runs = 0
        if self.bases[0] is not None:
            if self.bases[1] is not None:
                if self.bases[2] is not None:
                    runs = 1
                self.bases[2] = self.bases[1]
            self.bases[1] = self.bases[0]
        self.bases[0] = self.batter
        self.end_pa()
        self.score_runs(runs)

    # --- ball in play --------------------------------------------------------------------------------------
    def apply_play(self, batter_to: int, runner_to: tuple, outs: int) -> int:
        """batter_to: 0 = out, 1-3 = base reached, 4 = scored; runner_to[i]: destination of the runner on base
        i + 1 (None = no runner). Returns the runs that counted."""
        new, runs = [None, None, None], 0
        for i in (2, 1, 0):
            r = self.bases[i]
            if r is None:
                continue
            if runner_to[i] >= 4:
                runs += 1
            else:
                assert new[runner_to[i] - 1] is None, "two runners on one base"
                new[runner_to[i] - 1] = r
        if batter_to >= 4:
            runs += 1
        elif batter_to >= 1:
            assert new[batter_to - 1] is None, "two runners on one base"
            new[batter_to - 1] = self.batter
        self.end_pa()
        if self.outs + outs >= 3:
            self.add_outs(outs)  # third out: no runs count
            return 0
        self.bases = new
        self.add_outs(outs)
        self.score_runs(runs)
        return runs

    # --- bookkeeping ---------------------------------------------------------------------------------------
    def end_pa(self) -> None:
        self.balls = self.strikes = 0
        self.next_batter[self.batting] = (self.next_batter[self.batting] + 1) % 9

    def add_outs(self, n: int) -> None:
        self.outs += n
        if self.outs >= 3:
            self.end_half()

    def score_runs(self, n: int) -> None:
        if n <= 0:
            return
        self.score[self.batting] += n
        self.line[self.batting][-1] += n
        if not self.top and self.inning >= self.innings and self.score[1] > self.score[0]:
            self.final = True  # walk-off

    def end_half(self) -> None:
        self.outs, self.bases = 0, [None, None, None]
        if self.top:
            if self.inning >= self.innings and self.score[1] > self.score[0]:
                self.final = True  # home team leads: no bottom half
                return
            self.top = False
        else:
            if self.inning >= self.innings and self.score[0] != self.score[1]:
                self.final = True
                return
            if self.inning >= self.innings and self.scoreless_tie_ends and self.score == [0, 0]:
                self.final = True  # user rule 2026-10-23: a 0-0 game after the regulation innings is a tie
                return
            if self.inning >= self.max_innings:
                self.final = True
                return
            self.inning += 1
            self.top = True
        self.line[self.batting].append(0)
