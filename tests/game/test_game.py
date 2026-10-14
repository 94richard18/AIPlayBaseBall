"""Game engine tests (CPU only, no Isaac Sim). Run: python tests/game/test_game.py (or pytest)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aibaseball.game import BattedBall, FieldSpec, Game, GameState, resolve  # noqa: E402


def test_count():
    s = GameState()
    assert [s.ball() for _ in range(3)] == ["ball"] * 3
    assert s.strike() == "strike" and s.strike(foul=True) == "foul" and s.strike(foul=True) == "foul"
    assert s.strikes == 2
    assert s.strike() == "strikeout" and s.outs == 1 and s.balls == 0 and s.batter == 1


def test_walk_forces_runs_only_when_bases_loaded():
    s = GameState()
    s.bases = [0, None, 2]
    s.next_batter[0] = 3
    for _ in range(4):
        s.ball()
    assert s.bases == [3, 0, 2] and s.score == [0, 0]
    for _ in range(4):
        s.ball()
    assert s.score == [1, 0] and s.bases == [4, 3, 0]


def test_three_outs_switch_sides_and_third_out_runs_do_not_count():
    s = GameState()
    s.bases, s.outs = [None, None, 5], 2
    s.apply_play(0, (None, None, 4), 1)  # runner crosses on the third out at first
    assert s.score == [0, 0] and not s.top and s.bases == [None, None, None] and s.outs == 0


def test_home_team_skips_bottom_ninth_when_leading_and_walk_off():
    s = GameState()
    s.inning, s.top, s.score, s.outs = 9, True, [2, 3], 2
    s.strike(), s.strike(), s.strike()
    assert s.final
    s = GameState()
    s.inning, s.top, s.score, s.line = 9, False, [4, 4], [[0] * 9, [0] * 9]
    s.bases = [None, None, 7]
    s.apply_play(1, (None, None, 4), 0)
    assert s.final and s.score == [4, 5]


def test_extra_innings_on_a_tie():
    s = GameState(innings=1)
    for _ in range(18):  # six strikeouts: 0-0 after one inning
        s.strike()
    assert not s.final and s.inning == 2 and s.top


def test_batted_balls():
    fld = FieldSpec()
    no_one = (False, False, False)
    hr = resolve(BattedBall(46.0, 28.0, 0.0, 2000.0), no_one, 0)
    assert hr.kind == "home_run" and hr.batter_to == 4
    pop = resolve(BattedBall(28.0, 60.0, 5.0, 2500.0), no_one, 0)
    assert pop.kind == "fly_out" and pop.fielder not in ("LF", "CF", "RF")
    grounder = resolve(BattedBall(35.0, -8.0, 12.0, -800.0), (True, False, False), 0)
    assert grounder.kind == "ground_out" and grounder.runner_to == (2, None, None)
    foul = resolve(BattedBall(35.0, 5.0, 60.0, -800.0), no_one, 0)
    assert foul.kind == "foul"
    assert all(not fld.is_fair(x, y) for x, y in [(-1.0, 0.0), (10.0, 11.0), (10.0, -11.0)])


def test_sac_fly_scores_from_third():
    deep = resolve(BattedBall(41.0, 32.0, 0.0, 2300.0), (False, False, True), 0)
    assert deep.kind == "fly_out" and deep.runner_to[2] == 4


def test_full_game():
    g = Game(seed=1)
    s = g.play()
    assert s.final and s.inning >= 9
    assert sum(s.line[0]) == s.score[0] and sum(s.line[1]) == s.score[1]
    assert len(s.line[0]) == s.inning and len(s.line[1]) in (s.inning, s.inning - 1)
    assert all(e["result"] for e in g.log)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
