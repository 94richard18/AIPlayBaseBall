"""Play whole games with the game engine (CPU, stand-in pitcher / batter) and print the line score and outcomes.

    python scripts/sim_game.py [--games 1] [--seed 0] [--pitches pitches.csv]
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibaseball.game import FallbackPitcher, Game, TablePitcher, line_score  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pitches", default=None, help="CSV of the RL pitcher's evaluated pitches (speed_mps, plate_y, plate_z, crossed)")
parser.add_argument("--fallback", action="store_true", help="the fallback pitch (88-112 km/h at release, adopted 2026-10-21)")
args = parser.parse_args()

outcomes, runs, t0 = collections.Counter(), [], time.time()
for i in range(args.games):
    kw = {"pitcher": TablePitcher.from_csv(args.pitches)} if args.pitches else {"pitcher": FallbackPitcher()} if args.fallback else {}
    g = Game(seed=args.seed + i, **kw)
    s = g.play()
    runs += s.score
    outcomes.update(e["result"] for e in g.log)
    print(f"game {i} (seed {args.seed + i}, {len(g.log)} pitches)\n{line_score(s)}\n")
n = 2 * args.games
print(f"{(time.time() - t0) / args.games:.1f} s per game, {sum(runs) / n:.2f} runs per team")
print("per team per game: " + ", ".join(f"{k} {v / n:.1f}" for k, v in outcomes.most_common()))
