"""ONE OUTS plate appearances from the joint M2 rows + the M3 showcase PA plan (batter session, 10-28; CPU, no GPU run).

One joint process = 64 first pitches, one per env, and the pitcher's pitch per env does not depend on the seed (pred_y /
pred_z / kmh identical over M2 seeds 1-10): the M2 pitch set is these 64 pitches; the seed is the batter's randomness.
A PA (planning 14's rule) draws pitches = envs with the engine rng, without replacement inside a PA:

- hit by pitch (is_hit_by_pitch on the predicted crossing) ends it, as in play_one_outs;
- the take decision is TableBatter.decides_to_swing on that crossing (stochastic per PA, as in the engine); a take is a
  ball / strike by StrikeZone on the same crossing;
- a swing draws one of that env's M2 rows (one per seed): contact -> its batted ball through judge_pitch, else a miss.

The showcase PA (the first engine rng seed >= --pa_seed passing the pick rule: a ball in play with carry >= 15 m to end
it, no HBP, pitcher standing, swings without a fall before done, 3..7 pitches, a called strike and a swinging strike
preferred) is written as logs/m3/pa_plan.csv, one row per pitch: the film is one joint_smoke_20 run per row
(--seed seed --swing_csv swing_csv --after_done --timing_csv --video_env env --count_csv of the before-count), checked
against the planned row of that seed (exit_mps / launch_deg / carry_m / spray_deg / fair).

    python scripts/joint_pa.py --rows "logs/m2/m2_s*.csv" --out logs/m3
"""

import argparse
import csv
import functools
import glob
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aibaseball.game.one_outs as oo  # noqa: E402
from aibaseball.game.agents import BattedBall, Pitch, StrikeZone, TableBatter  # noqa: E402
from aibaseball.game.field import FieldSpec  # noqa: E402
from aibaseball.game.fielding import FieldingSpec  # noqa: E402
from aibaseball.physics import PhysicsSpec  # noqa: E402
from aibaseball.physics.grass import GrassSpec  # noqa: E402

# the batted ball's flight + bounce is the slow part and depends only on the row: once per row
_bounce = functools.lru_cache(maxsize=None)(lambda key, fb=oo.first_bounce: fb(BattedBall(*key), *SPECS))
SPECS = (FieldSpec(), FieldingSpec(), GrassSpec(), PhysicsSpec())
oo.first_bounce = lambda ball, *s: _bounce((ball.exit_speed_mps, ball.launch_deg, ball.spray_deg, ball.backspin_rpm))


def f(x):
    return float(x) if x not in ("", "nan", None) else float("nan")


def b(r, c):
    return r.get(c) == "True"


def batted(r):
    la = f(r["launch_deg"])
    spin = f(r.get("backspin_rpm", "nan"))
    if math.isnan(spin):  # rows from before the backspin column: StatBatter's spin model (as m2_analysis)
        spin = min(3500.0, 1200.0 + 40.0 * la) if la > 0.0 else -800.0
    return BattedBall(f(r["exit_mps"]), la, f(r["spray_deg"]), spin)


def fair(r):
    c, s = f(r["carry_m"]), f(r["spray_deg"])
    return c == c and s == s and FieldSpec().is_fair(c * math.cos(math.radians(s)), c * math.sin(math.radians(s)))


def play_pa(envs, rng, zone):
    """envs: {env: [rows, one per seed]}. Returns the pitch log and (winner, how)."""
    left = sorted(envs)
    dec = TableBatter([], 0.0)
    balls = strikes = 0
    log = []
    while left:
        e = left.pop(rng.randrange(len(left)))
        r0 = envs[e][0]
        p = Pitch(f(r0["kmh"]) / 3.6, f(r0["pred_y"]), f(r0["pred_z"]))
        ev = dict(env=e, balls_before=balls, strikes_before=strikes, kmh=f(r0["kmh"]), in_zone=zone.contains(p.plate_y, p.plate_z))
        log.append(ev)
        if oo.is_hit_by_pitch(p.plate_y, p.plate_z):
            ev.update(swing=0, kind="hit_by_pitch", expected="hit_by_pitch", row=r0)
            return log, ("batter", "hit_by_pitch")
        sw = dec.decides_to_swing(p, zone, rng)
        r = rng.choice(envs[e]) if sw else r0
        kind = "take" if not sw else "contact" if b(r, "hit") else "miss"
        ev.update(swing=int(sw), kind=kind, row=r)
        b2, s2, res, extra = oo.judge_pitch(balls, strikes, p, kind, batted(r) if kind == "contact" else None, zone, *SPECS)
        ev["expected"] = ("called_strike" if s2 > strikes else "ball") if kind == "take" else "swinging_strike" if kind == "miss" \
            else "in_play" if res is not None else "foul"
        if res is not None:
            ev["how"] = res[1]
        balls, strikes = b2, s2
        if res is not None:
            return log, res
    return log, (None, "out_of_pitches")


def showcase_ok(log, how, lo, hi, want="in_play"):
    """want: in_play (an end by a ball in play, carry >= 15 m), pitcher (an infield ball), batter (the batter wins: an
    outfield hit / home run, or a walk - M2's 182 hits have no outfield ball, its longest carries ~40 m judge infield)."""
    last = log[-1]
    clean = (lo <= len(log) <= hi and how != "hit_by_pitch" and not any(b(x["row"], "pitcher_fell") for x in log)
             and not any(x["swing"] and b(x["row"], "batter_fell_before_done") for x in log))
    in_play = last["expected"] == "in_play" and f(last["row"]["carry_m"]) >= 15.0
    if want == "batter":
        return clean and (how == "walk" or (in_play and how in ("outfield_hit", "home_run")))
    return clean and in_play and (want == "in_play" or how == "infield_ball")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="logs/m2/m2_s*.csv")
    ap.add_argument("--out", default="logs/m3", help="directory: pa_plan.csv + pa_swing_s<seed>_e<env>.csv per pitch")
    ap.add_argument("--stats", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0, help="engine rng seed of the stats PAs")
    ap.add_argument("--pa_seed", type=int, default=0, help="first engine rng seed tried for the showcase PA")
    ap.add_argument("--tries", type=int, default=5000)
    ap.add_argument("--min_pitches", type=int, default=3)
    ap.add_argument("--max_pitches", type=int, default=7)
    ap.add_argument("--want", default="in_play", choices=("in_play", "pitcher", "batter"),
                    help="the showcase PA's end: any ball in play, an infield ball (pitcher wins), an outfield hit (batter wins)")
    ap.add_argument("--plan", default="pa_plan.csv", help="plan file name inside --out")
    ap.add_argument("--take_mode", type=int, default=0, help="BatterSide take mode written into the masks for a take (see BatterSide swing)")
    ap.add_argument("--after_done", default="policy")
    ap.add_argument("--timing_csv", default="logs/m2lite/timing.csv")
    a = ap.parse_args()
    envs = {}
    files = sorted(glob.glob(a.rows), key=lambda p: int(re.search(r"_s(\d+)\.csv", p).group(1)))
    for p in files:
        s = int(re.search(r"_s(\d+)\.csv", p).group(1))
        for r in csv.DictReader(open(p)):
            if b(r, "handed"):
                r["seed"] = s
                envs.setdefault(int(r["env"]), []).append(r)
    for e, rs in envs.items():  # the pitch is the env's, whatever the seed
        assert len({(r["kmh"], r["pred_y"], r["pred_z"]) for r in rs}) == 1, f"env {e}: the pitch differs between seeds"
    zone = StrikeZone()
    n_hbp = sum(oo.is_hit_by_pitch(f(rs[0]["pred_y"]), f(rs[0]["pred_z"])) for rs in envs.values())

    rng = random.Random(a.seed)
    hows, npitch = {}, 0
    for _ in range(a.stats):
        log, (_, how) = play_pa(envs, rng, zone)
        hows[how] = hows.get(how, 0) + 1
        npitch += len(log)
    bw = sum(v for k, v in hows.items() if k in ("walk", "hit_by_pitch", "outfield_hit", "home_run")) / a.stats
    print(f"[pa] {len(envs)} pitches x {len(files)} seeds ({sum(map(len, envs.values()))} rows), HBP pitches {n_hbp}; "
          f"{a.stats} PAs (engine rng seed {a.seed}, no repeat inside a PA): batter wins {bw:.1%}, {npitch / a.stats:.2f} "
          f"pitches per PA, by outcome { {k: round(v / a.stats, 3) for k, v in sorted(hows.items())} }")

    best = None
    for d in range(a.pa_seed, a.pa_seed + a.tries):
        log, (win, how) = play_pa(envs, random.Random(d), zone)
        if not showcase_ok(log, how, a.min_pitches, a.max_pitches, a.want):
            continue
        rich = any(x["expected"] == "called_strike" for x in log) and any(x["expected"] == "swinging_strike" for x in log)
        if best is None or rich:
            best = (d, log, win, how)
        if rich:
            break
    if best is None:
        print(f"[pa] no PA passes the pick rule in engine rng seeds {a.pa_seed}..{a.pa_seed + a.tries - 1}")
        sys.exit(1)
    d, log, win, how = best
    os.makedirs(a.out, exist_ok=True)
    keys = ["pitch", "seed", "env", "balls_before", "strikes_before", "swing", "expected", "swing_csv", "after_done",
            "timing_csv", "exit_mps", "launch_deg", "carry_m", "spray_deg", "fair"]
    n_env = max(int(r["env"]) for p in files[:1] for r in csv.DictReader(open(p))) + 1
    with open(os.path.join(a.out, a.plan), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for i, x in enumerate(log):
            r = x["row"]
            m = os.path.join(a.out, f"pa_swing_s{r['seed']}_e{x['env']}.csv").replace("\\", "/")
            with open(m, "w", newline="") as fm:  # every env swings except this pitch's env when it is a take
                mw = csv.writer(fm)
                mw.writerow(["env", "swing"])
                mw.writerows([e, (1 if x["swing"] else a.take_mode) if e == x["env"] else 1] for e in range(n_env))
            row = dict(pitch=i + 1, seed=r["seed"], env=x["env"], balls_before=x["balls_before"],
                       strikes_before=x["strikes_before"], swing=x["swing"], expected=x["expected"], swing_csv=m,
                       after_done=a.after_done, timing_csv=a.timing_csv)
            if x["expected"] in ("in_play", "foul"):
                row.update(exit_mps=r["exit_mps"], launch_deg=r["launch_deg"], carry_m=r["carry_m"], spray_deg=r["spray_deg"], fair=fair(r))
            w.writerow(row)
    print(f"[pa] showcase PA (engine rng seed {d}): {len(log)} pitches, {win} wins by {how}")
    for i, x in enumerate(log):
        r = x["row"]
        extra = f" EV {f(r['exit_mps']):.1f} carry {f(r['carry_m']):.1f} m" if x["expected"] in ("in_play", "foul") else ""
        print(f"[pa]   {i + 1}. seed {r['seed']:2d} env {x['env']:2d} {x['balls_before']}-{x['strikes_before']} "
              f"{x['kmh']:.1f} km/h {'zone' if x['in_zone'] else 'out '} -> {x['expected']}{extra}")
    print(f"[pa] wrote {a.out}/{a.plan} + {len(log)} swing masks")


main()
