"""ONE OUTS matches: the fallback pitcher against a batter, one plate appearance each (CPU only).

    python scripts/one_outs.py [--matches 1000] [--batter_table assets/game/batter_h7_aim012.json] [--hit_rate 0.73]
"""

import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aibaseball.game import FallbackPitcher, StatBatter, TableBatter, TablePitcher  # noqa: E402
from aibaseball.game.one_outs import play_one_outs  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--matches", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batter_table", default=None, help="RL batter table (default: the statistical batter)")
parser.add_argument("--pitches", default=None, help="the RL pitcher's evaluated pitches (eval_stage --pitch_csv); default: the fallback pitch")
parser.add_argument("--hit_rate", type=float, default=0.73)
parser.add_argument("--show", type=int, default=3, help="print this many matches pitch by pitch")
args = parser.parse_args()

rng = random.Random(args.seed)
pitcher = TablePitcher.from_csv(args.pitches) if args.pitches else FallbackPitcher()
batter = TableBatter.from_json(args.batter_table, args.hit_rate) if args.batter_table else StatBatter()
wins, hows, n_pitches = collections.Counter(), collections.Counter(), 0
for i in range(args.matches):
    r = play_one_outs(pitcher, batter, rng)
    wins[r.winner] += 1
    hows[r.how] += 1
    n_pitches += len(r.pitches)
    if i < args.show:
        seq = " ".join(f"{e['balls']}-{e['strikes']}:{e['swing']}" + (f"@({e['land'][0]:.0f},{e['land'][1]:.0f})"
                       if e.get("land") else "") for e in r.pitches)
        print(f"match {i}: {r.winner} wins by {r.how} | {seq}")
n = args.matches
print(f"{n} matches: batter {wins['batter'] / n:.1%}, pitcher {wins['pitcher'] / n:.1%}; {n_pitches / n:.1f} pitches each")
print("how: " + ", ".join(f"{k} {v / n:.1%}" for k, v in hows.most_common()))
