"""Play whole games with the game engine (CPU, stand-in pitcher / batter) and print the line score and outcomes.

    python scripts/sim_game.py [--games 1] [--seed 0]
"""

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibaseball.game import Game, line_score  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

outcomes, runs, t0 = collections.Counter(), [], time.time()
for i in range(args.games):
    g = Game(seed=args.seed + i)
    s = g.play()
    runs += s.score
    outcomes.update(e["result"] for e in g.log)
    print(f"game {i} (seed {args.seed + i}, {len(g.log)} pitches)\n{line_score(s)}\n")
n = 2 * args.games
print(f"{(time.time() - t0) / args.games:.1f} s per game, {sum(runs) / n:.2f} runs per team")
print("per team per game: " + ", ".join(f"{k} {v / n:.1f}" for k, v in outcomes.most_common()))
