"""Selection route (planning 14, 10-30): a measurement for the user's decision, not a proposal. CPU only.

The delivery 12500's sweet spot lands at a ~fixed horizontal spot (axis_off vs plate_y slope ~-0.8): find that spot y0
(plate_y where the fitted axis_off line crosses 0, in-zone pitches) and, for bands |plate_y - y0| <= 3 / 5 / 8 / 12 cm,
report the in-zone share inside the band, the hit rate, |dx|, exit, launch, carry (game flight model: drag + Magnus from
0.9 m, aibaseball.game.fielding.ball_path; first bounce or wall / home run) and the count with carry >= 38.9 m. Then
ONE OUTS (1000 matches x 5 seeds) vs the P3c pitcher table with an oracle batter that swings only at in-zone pitches
inside the band (takes everything else, even strikes); contact at the band's hit rate, batted balls from the band's hits.

Input: diag_bat_contact --contact_log per_pitch JSON with spray_contact / backspin_contact (diag1024 seed 0, P3c t18,
jitter 0, bias 0.055, aim 0). diag plate_y is the engine's plate_y (+ = third-base side; TablePitcher flips the pitch CSV).

    python scripts/band_select.py --diag logs/diag_bat_contact_band12500.json --out logs/band_select
"""
import argparse
import collections
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aibaseball.game import TablePitcher  # noqa: E402
from aibaseball.game.agents import BattedBall, StrikeZone, Swing  # noqa: E402
from aibaseball.game.field import FieldSpec  # noqa: E402
from aibaseball.game.fielding import FieldingSpec, ball_path  # noqa: E402
from aibaseball.game.one_outs import play_one_outs  # noqa: E402
from aibaseball.physics import PhysicsSpec  # noqa: E402
from aibaseball.physics.grass import GrassSpec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--pitches", default="assets/game/p3c_identical_t18_pitches.csv")
ap.add_argument("--bands", default="0.03,0.05,0.08,0.12")
ap.add_argument("--matches", type=int, default=1000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--out", default="logs/band_select")
args = ap.parse_args()
SS, OUTFIELD = 0.7112, 38.9
P = {k: np.array(v, dtype=float) for k, v in json.load(open(args.diag))["per_pitch"].items()}
zone = StrikeZone()
z = np.array([zone.contains(a, b) for a, b in zip(P["plate_y"], P["plate_z"])])
h = P["hit"].astype(bool)
ax = P["x_on_bat"] - SS
L = [f"band selection (delivery 12500 @ 1.5x, {os.path.basename(args.diag)}; in-zone n {z.sum()}, hits {(h & z).sum()})"]
b = np.polyfit(P["plate_y"][z], ax[z], 1)
bh = np.polyfit(P["plate_y"][h & z], ax[h & z], 1)
y0 = -b[1] / b[0]
L.append(f"axis_off = {b[0]:+.3f} plate_y {b[1]:+.4f} (in-zone hits + misses) -> y0 {y0:+.4f} m; hits only: "
         f"{bh[0]:+.3f} / {bh[1]:+.4f} -> y0 {-bh[1] / bh[0]:+.4f}")

fld, fs, grass, phys = FieldSpec(), FieldingSpec(), GrassSpec(), PhysicsSpec()


def carry(i):
    ball = BattedBall(P["ev_any"][i], P["launch_any"][i], P["spray_contact"][i], P["backspin_contact"][i])
    path, end = ball_path(ball, fld, grass, fs, phys)
    if end != "landed":
        return 99.0, end, ball
    land = [s for s in path if s[4]][-1]
    return math.hypot(land[1], land[2]), "fair" if fld.is_fair(land[1], land[2]) else "foul", ball


hall = h.copy()
h = h & np.isfinite(P["ev_any"])  # a hit whose episode ended on its own impact step has no launch record
C = {int(i): carry(i) for i in np.flatnonzero(h)}
q = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]" if len(v) else "-"  # noqa: E731


class BandBatter:
    """Oracle: swing only at in-zone pitches with |plate_y - y0| <= w; contact at the band's hit rate; balls from its hits."""

    def __init__(self, rows, hit_rate, y0, w):
        self.rows, self.hit_rate, self.y0, self.w = rows, hit_rate, y0, w

    def swing(self, state, pitch, zone_, rng):
        if not (zone_.contains(pitch.plate_y, pitch.plate_z) and abs(pitch.plate_y - self.y0) <= self.w) or not self.rows:
            return Swing("take")
        if rng.random() >= self.hit_rate:
            return Swing("miss")
        return Swing("contact", rng.choice(self.rows))


pitcher = TablePitcher.from_csv(args.pitches)
tz = [(r[1], r[2]) for r in pitcher.rows if r[3]]
L.append(f"P3c pitch table: {len(pitcher.rows)} rows, in-zone {sum(zone.contains(a, c) for a, c in tz) / len(pitcher.rows):.3f}")
res = {}
for w in [float(x) for x in args.bands.split(",")] + [9.0]:
    m = z & (np.abs(P["plate_y"] - y0) <= w)
    mh = m & h
    ids = np.flatnonzero(mh)
    cr = np.array([C[int(i)][0] for i in ids])
    fair = np.array([C[int(i)][1] in ("fair", "wall", "home_run") for i in ids])
    hr = (m & hall).sum() / max(m.sum(), 1)  # every hit counts; the balls come from the recorded ones
    lab = "all in-zone" if w > 1 else f"|plate_y - y0| <= {w * 100:.0f} cm"
    L.append(f"{lab}: hits without a launch record {int((m & hall & ~h).sum())}; share of in-zone {m.sum() / z.sum():.3f} (n {m.sum()}); hit {hr:.3f}; |dx| med "
             f"{np.median(np.abs(P['dx_any'][mh])) if mh.any() else float('nan'):.4f}; exit {q(P['ev_any'][mh])}; launch "
             f"{q(P['launch_any'][mh])}; carry {q(cr)} m; fair carry >= {OUTFIELD} m: {int(((cr >= OUTFIELD) & fair).sum())} / {len(ids)}")
    rows = [C[int(i)][2] for i in ids]
    zb = sum(zone.contains(a, c) and abs(a - y0) <= w for a, c in tz) / len(pitcher.rows)
    wins, hows = [], collections.Counter()
    for s in range(args.seeds):
        rng = random.Random(s)
        bat = BandBatter(rows, hr, y0, w)
        n_w = 0
        for _ in range(args.matches):
            r = play_one_outs(pitcher, bat, rng)
            n_w += r.winner == "batter"
            hows[r.how] += 1
        wins.append(n_w / args.matches)
    n = args.matches * args.seeds
    L.append(f"  ONE OUTS (P3c table, pitches in the band {zb:.3f}): batter {np.mean(wins):.1%} (seeds "
             + " ".join(f"{x:.1%}" for x in wins) + "); " + ", ".join(f"{k} {v / n:.1%}" for k, v in hows.most_common()))
    res[lab] = dict(share=float(m.sum() / z.sum()), hit=float(hr), batter_win=float(np.mean(wins)), seeds=wins,
                    how={k: v / n for k, v in hows.items()}, n_hits=int(len(ids)), outfield=int(((cr >= OUTFIELD) & fair).sum()))
L.append("reference: the delivery (joint M2, 20000 PAs) batter 9.8% (all walks); 'all in-zone' above = the same oracle "
         "rule without a band (swing at every in-zone pitch, take every ball)")
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(L) + "\n")
json.dump(dict(y0=float(y0), fit=[float(b[0]), float(b[1])], bands=res), open(os.path.join(args.out, "band_select.json"), "w"), indent=1)
print("\n".join(L))
