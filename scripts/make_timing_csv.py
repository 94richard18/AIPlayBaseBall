"""Two-pass oracle timing, pass 1 -> pass 2 (batter session, 10-28): per env (hand-off time, contact-plane time) of the
live pitch, from joint_smoke row CSVs of the same pitcher / release table. The pitch does not depend on the batter or
the seed (m2lite s1 vs s2: kmh / handoff_t / pred columns 64/64 identical, t_plane identical where both crossed), so
rows from several runs are merged. Per env, in order: a recorded t_plane (any run), handoff_t + flight_pred (fly_to_x at
the hand-off, rows from after that column), else a flight-vs-kmh fit over the crossed envs (marked "fit").

    python scripts/make_timing_csv.py --rows "logs/m2lite/m2lite_s?.csv" --out logs/m2lite/timing.csv
"""

import argparse
import csv
import glob

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
runs = [list(csv.DictReader(open(p))) for p in sorted(glob.glob(a.rows))]
n = len(runs[0])
assert all(len(r) == n for r in runs), [len(r) for r in runs]
ok = lambda x: x not in (None, "", "nan")  # noqa: E731
out, src = [], {"t_plane": 0, "flight_pred": 0, "fit": 0}
for e in range(n):
    rs = [r[e] for r in runs]
    assert len({(r["kmh"], r["handoff_t"]) for r in rs}) == 1, (e, [(r["kmh"], r["handoff_t"]) for r in rs])
    h = float(rs[0]["handoff_t"])
    tp = [float(r["t_plane"]) for r in rs if ok(r.get("t_plane"))]
    fp = [float(r["flight_pred"]) for r in rs if ok(r.get("flight_pred"))]
    if tp:
        assert max(tp) - min(tp) < 1e-3, (e, tp)
        out.append([e, h, tp[0], "t_plane"])
    elif fp:
        out.append([e, h, h + fp[0], "flight_pred"])
    else:
        out.append([e, h, float("nan"), "fit"])
    src[out[-1][3]] += 1
known = [(float(runs[0][o[0]]["kmh"]), o[2] - o[1]) for o in out if o[3] != "fit"]
k, c = np.polyfit(*zip(*known), 1)
res = np.array([f - (k * v + c) for v, f in known])
for o in out:
    if o[3] == "fit":
        o[2] = o[1] + k * float(runs[0][o[0]]["kmh"]) + c
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["env", "handoff_t", "t_plane", "source"])
    w.writerows([[o[0], round(o[1], 4), round(o[2], 4), o[3]] for o in out])
print(f"[timing] {n} envs from {len(runs)} runs: {src}; fit flight {k * 1000:+.1f} ms per km/h, residual sd "
      f"{res.std() * 1000:.1f} ms -> {a.out}", flush=True)
