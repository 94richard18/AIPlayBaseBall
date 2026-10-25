"""ONE OUTS match video (CPU only): play N matches with the engine and render them one after another.

    python scripts/render_one_outs.py [--matches 10] [--seed 0] [--batter_table assets/game/batter_h9_12300_aim012.json]
        [--pitches <eval_stage --pitch_csv>] [--out videos/one_outs_fallback.mp4]
Default pitcher: the fallback pitch (FallbackPitcher); default batter: the RL batter table (Human9 model_12300 + aim -0.12, hit rate 0.691, 10-25; 12500 / -0.08 reverted after its retest).
"""

import argparse
import collections
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aibaseball.game import FallbackPitcher, TableBatter, TablePitcher  # noqa: E402
from aibaseball.game.one_outs import play_one_outs  # noqa: E402
from aibaseball.game.one_outs_replay import render_matches  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--matches", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batter_table", default="assets/game/batter_h9_12300_aim012.json")
parser.add_argument("--hit_rate", type=float, default=0.691)
parser.add_argument("--pitches", default=None, help="the RL pitcher's evaluated pitches (eval_stage --pitch_csv); default: the fallback pitch")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", default="videos/one_outs_fallback.mp4")
args = parser.parse_args()

rng = random.Random(args.seed)
pitcher = TablePitcher.from_csv(args.pitches) if args.pitches else FallbackPitcher()
batter = TableBatter.from_json(args.batter_table, args.hit_rate)
matches = [play_one_outs(pitcher, batter, rng) for _ in range(args.matches)]
wins = collections.Counter(m.winner for m in matches)
print(f"{args.matches} matches: batter {wins['batter']}, pitcher {wins['pitcher']}; "
      + ", ".join(f"{k} {v}" for k, v in collections.Counter(m.how for m in matches).most_common()))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
t0 = time.time()
n = render_matches(matches, args.out, fps=args.fps)
print(f"{args.out}: {n} frames ({n / args.fps:.0f} s), rendered in {time.time() - t0:.0f} s")
