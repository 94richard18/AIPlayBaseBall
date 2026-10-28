"""M2 offline analysis (batter session, 10-28): the joint scene's live pitches (logs/m2/m2_s*.csv, scripts/joint_smoke.py
--after_done hold) against the M2 bars.

- crossing: the joint env's integrator crossing (j_cross, pitcher plate frame: y mirrored) vs the hand-off's fly_to_x
  prediction (pred_y / pred_z, the game's convention), < 1 cm
- in-zone hit rate (StrikeZone on the predicted crossing; a hit has no j_cross) and falls after a hit inside the
  training episode (batter_fell_before_done)
- ONE OUTS at-bats with the engine rng: each pitch is a live row drawn at random; the take / swing decision is
  TableBatter.decides_to_swing (applied offline: the live batter always swings); a swing keeps the row's own outcome
  (contact -> its batted ball through judge_pitch, else a miss). Walk rate and the batter's win rate.

    python scripts/m2_analysis.py --rows "logs/m2/m2_s*.csv" --at_bats 20000
"""

import argparse
import csv
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aibaseball.game.agents import BattedBall, Pitch, StrikeZone, Swing, TableBatter  # noqa: E402
from aibaseball.game.one_outs import play_one_outs  # noqa: E402


def f(x):
    return float(x) if x not in ("", "nan") else float("nan")


class ReplayPitcher:
    def __init__(self, rows):
        self.rows, self.last = rows, None

    def pitch(self, state, rng):
        self.last = rng.choice(self.rows)
        r = self.last
        return Pitch(f(r["kmh"]) / 3.6, f(r["pred_y"]), f(r["pred_z"]))


class ReplayBatter:
    def __init__(self, pitcher):
        self.p, self.t = pitcher, TableBatter([], 0.0)

    def swing(self, state, pitch, zone, rng):
        if not self.t.decides_to_swing(pitch, zone, rng):
            return Swing("take")
        r = self.p.last
        if r["hit"] != "True":
            return Swing("miss")
        la = f(r["launch_deg"])
        spin = f(r.get("backspin_rpm", "nan"))
        if math.isnan(spin):  # rows from before the backspin column: StatBatter's spin model
            spin = min(3500.0, 1200.0 + 40.0 * la) if la > 0.0 else -800.0
        return Swing("contact", BattedBall(f(r["exit_mps"]), la, f(r["spray_deg"]), spin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="logs/m2/m2_s*.csv")
    ap.add_argument("--at_bats", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [r for p in sorted(glob.glob(a.rows)) for r in csv.DictReader(open(p))]
    rows = [r for r in rows if r["handed"] == "True"]
    zone = StrikeZone()
    n = len(rows)
    # j_cross y is the pitcher's plate frame (+y toward first base: the batter stands at plate y -0.9); pred_y is the
    # batter frame B = the game's convention (Pitch.plate_y, BODY_Y_M: + toward third base) -> compare with the sign flipped
    dy = [abs(-f(r["plate_y"]) - f(r["pred_y"])) for r in rows if not math.isnan(f(r["plate_y"]))]
    dz = [abs(f(r["plate_z"]) - f(r["pred_z"])) for r in rows if not math.isnan(f(r["plate_z"]))]
    inz = [zone.contains(f(r["pred_y"]), f(r["pred_z"])) for r in rows]
    hit = [r["hit"] == "True" for r in rows]
    fell = [r["batter_fell_before_done"] == "True" for r in rows]
    n_in = sum(inz)
    hits_in = sum(h and i for h, i in zip(hit, inz))
    print(f"[m2] {n} live pitches from {len(glob.glob(a.rows))} files; in zone {n_in} ({n_in / n:.3f})")
    print(f"[m2] crossing vs fly_to_x ({len(dy)} crossed): |dy| max {max(dy) * 100:.2f} cm, |dz| max {max(dz) * 100:.2f} cm "
          f"(bar < 1 cm)")
    hr = hits_in / max(n_in, 1)
    se = math.sqrt(hr * (1 - hr) / max(n_in, 1))
    print(f"[m2] in-zone hit rate {hr:.3f} +- {se:.3f} ({hits_in}/{n_in}; bar 0.46 +- 0.05); all pitches {sum(hit) / n:.3f}")
    nh = sum(hit)
    fh = sum(x and h for x, h in zip(fell, hit))
    print(f"[m2] falls after a hit (inside the training episode) {fh}/{nh} = {fh / max(nh, 1):.1%} (bar 43 +- 8%); "
          f"miss falls inside the episode {sum(x and not h for x, h in zip(fell, hit))}")
    rng = random.Random(a.seed)
    pit = ReplayPitcher(rows)
    bat = ReplayBatter(pit)
    res = [play_one_outs(pit, bat, rng, zone=zone) for _ in range(a.at_bats)]
    hows = {}
    for r in res:
        hows[r.how] = hows.get(r.how, 0) + 1
    w = hows.get("walk", 0) / a.at_bats
    bw = sum(r.winner == "batter" for r in res) / a.at_bats
    print(f"[m2] ONE OUTS x {a.at_bats} (engine rng seed {a.seed}): walks {w:.1%} (bar 10 +- 2.3%), batter wins {bw:.1%}, "
          f"by outcome { {k: round(v / a.at_bats, 3) for k, v in sorted(hows.items())} }")


main()
