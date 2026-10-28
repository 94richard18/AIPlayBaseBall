"""Pre-contact 'falls' (planning 14, 10-29): rows whose bat_fall_t comes before the contact (hits: hit_t) or before the
ball reaches the plate plane (misses: t_plane; else the predicted crossing handoff_t + flight_pred), i.e. a fall that
could veto BatterSide's hit window (batter_side.py:414) or end the episode before the pitch arrives."""
import csv, sys
for p in sys.argv[1:]:
    R = list(csv.DictReader(open(p))); f = lambda x: float(x) if x not in ("", "nan") else float("nan")
    hits = [r for r in R if r["hit"] == "True"]; miss = [r for r in R if r["hit"] != "True"]
    hb = [r for r in hits if f(r["bat_fall_t"]) < f(r["hit_t"])]
    arr = lambda r: f(r["t_plane"]) if f(r["t_plane"]) == f(r["t_plane"]) else f(r["handoff_t"]) + f(r["flight_pred"])
    mb = [r for r in miss if f(r["bat_fall_t"]) < arr(r)]
    ab = [r for r in R if f(r["bat_fall_t"]) < arr(r) or (r["hit"] == "True" and f(r["bat_fall_t"]) < f(r["hit_t"]))]
    print(f"PREFALL {p}: hits fallen before contact {len(hb)}/{len(hits)}; misses fallen before the plate {len(mb)}/{len(miss)}; "
          f"all rows {len(ab)}/{len(R)}; earliest fall minus arrival {min(f(r['bat_fall_t']) - arr(r) for r in R):.3f} s")
    for r in mb[:8]: print(f"  env {r['env']} fall {r['bat_fall_t']} arrival {arr(r):.3f} start {r['batter_start']}")
