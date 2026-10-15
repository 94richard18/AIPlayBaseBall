"""Play one game with the game engine and render its top-view replay to an mp4 (CPU only).

    python scripts/render_game.py [--seed 0] [--out videos/game_seed0.mp4] [--max_events N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibaseball.game import Game, line_score  # noqa: E402
from aibaseball.game.replay import render  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--max_events", type=int, default=None)
args = parser.parse_args()

g = Game(seed=args.seed)
s = g.play()
print(line_score(s))
out = args.out or f"videos/game_seed{args.seed}.mp4"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
t0 = time.time()
n = render(g.log, out, fps=args.fps, max_events=args.max_events)
print(f"{out}: {n} frames ({n / args.fps / 60:.1f} min), rendered in {time.time() - t0:.0f} s")
