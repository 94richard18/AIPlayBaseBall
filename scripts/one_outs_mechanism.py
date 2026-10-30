"""ONE OUTS mechanism split (planning 14, 10-31): per batter config (3 showcase tables x 5 seeds x 1000 matches, the same
runs as the 3-table metric), the foul / fair share of contacts and the PA outcome shares, plus pitches per PA. CPU only.

    python scripts/one_outs_mechanism.py
"""
import collections
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aibaseball.game import TableBatter, TablePitcher  # noqa: E402
from aibaseball.game.field import FieldSpec  # noqa: E402
from aibaseball.game.one_outs import play_one_outs  # noqa: E402

T = "assets/game/p3c_identical_t18_pitches.csv"
CONFIGS = {  # name: (hit rate = the diag's 'slow in zone', 3 tables)
    "12500": (0.458, ["logs/b_table_e15_12500.json", "logs/b_table_tchk_plain.json", "logs/b_table_t3_e15_3.json"]),
    "12500+saim": (0.526, [f"logs/saim/table_{j}.json" for j in (1, 2, 3)]),
    "h29": (0.908, [f"logs/h29adopt/table_h29_{j}.json" for j in (1, 2, 3)]),
    "h29s": (0.700, [f"logs/h29adopt/table_h29s_{j}.json" for j in (1, 2, 3)]),
}
fld = FieldSpec()
pitcher = TablePitcher.from_csv(T)
print("config | contacts per PA, foul / fair of contacts | PA: batter win, strikeout, walk, HBP, infield out, outfield hit | "
      "pitches per PA | fair contacts: launch >= 10 deg, exit med")
for name, (hr, tables) in CONFIGS.items():
    hows, n, npit, ncon, nfoul = collections.Counter(), 0, 0, 0, 0
    la_fair, ev_fair = [], []
    for tb in tables:
        batter = TableBatter.from_json(tb, hr)
        for s in range(5):
            rng = random.Random(s)
            for _ in range(1000):
                r = play_one_outs(pitcher, batter, rng)
                n += 1
                hows[(r.winner, r.how)] += 1
                npit += len(r.pitches)
                for e in r.pitches:
                    if e["swing"] != "contact":
                        continue
                    ncon += 1
                    land = e.get("land")
                    if e.get("end") not in ("home_run", "wall") and land is not None and not fld.is_fair(*land):
                        nfoul += 1
                    else:
                        la_fair.append(e["ball"].launch_deg)
                        ev_fair.append(e["ball"].exit_speed_mps)
    sh = lambda k: sum(v for (w, h), v in hows.items() if h == k) / n  # noqa: E731
    bw = sum(v for (w, h), v in hows.items() if w == "batter") / n
    print(f"{name:10s} | {ncon / n:.2f}, foul {nfoul / max(ncon, 1):.3f} fair {1 - nfoul / max(ncon, 1):.3f} | "
          f"win {bw:.3f}, K {sh('strikeout'):.3f}, BB {sh('walk'):.3f}, HBP {sh('hit_by_pitch'):.3f}, "
          f"in-play out {sh('infield_ball'):.3f}, OF hit {sh('outfield_hit') + sh('home_run'):.3f} | {npit / n:.2f} | "
          f"{sum(x >= 10 for x in la_fair) / max(len(la_fair), 1):.3f}, {st.median(ev_fair) if ev_fair else float('nan'):.1f}")
