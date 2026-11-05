"""Take pre-check summary (11-06): one line per variant from the joint csv + the --take_npz of scripts/take_precheck.py.
falls, take_bat_vmax max / median, and the start snap: torso tilt range, root displacement (from the window's first step)
and max |body joint vel| over start_at -0.05..+0.05 s (and over start..+0.5 s), median / max over envs.

    python logs/take_precheck_sum.py <label> <csv> <npz>"""
import csv
import sys

import numpy as np

lab, cp, npz = sys.argv[1:4]
rows = list(csv.DictReader(open(cp)))
b = lambda r, c: r[c] in ("True", "1")  # noqa: E731
d = np.load(npz)
t, root, tilt, jv, st = d["t"], d["root"], d["tilt"], d["jv"], d["start_at"]


def win(lo, hi, at=None):
    at = st if at is None else at
    tr, rd, jm = [], [], []
    for e in range(len(at)):
        if np.isnan(at[e]):
            continue
        m = (t[e] >= at[e] + lo - 1e-6) & (t[e] <= at[e] + hi + 1e-6)
        if not m.any():
            continue
        tr.append(np.ptp(tilt[e, m]))
        rd.append(np.linalg.norm(root[e, m] - root[e, m][0], axis=-1).max() * 100.0)
        jm.append(jv[e, m].max())
    q = lambda x: f"{np.median(x):.1f} / {np.max(x):.1f}"  # noqa: E731
    return f"torso {q(tr)} deg, root {q(rd)} cm, |qd| {q(jm)} rad/s (median / max)"


def pose():  # where it settles: torso tilt + bat CM height at start, start + 0.5 s and the run's end (median over envs)
    if "batz" not in d.files:
        return "no bat z"
    bz = d["batz"]
    out = []
    for lab_, dt in (("start", 0.0), ("+0.5 s", 0.5), ("+2.0 s", 2.0), ("end", None)):
        k = [(-1 if dt is None else int(np.abs(t[e] - (st[e] + dt)).argmin())) for e in range(len(st))]
        out.append(f"{lab_} torso {np.median([tilt[e, i] for e, i in enumerate(k)]):.1f} deg bat z "
                   f"{np.median([bz[e, i] for e, i in enumerate(k)]):.2f} m")
    return "; ".join(out)


vm = np.array([float(r["take_bat_vmax"]) for r in rows])
fell = sum(b(r, "batter_fell") for r in rows)
dev = sum(r["take_dev_t"] not in ("", "nan") for r in rows)
pf = sum(b(r, "pitcher_fell") for r in rows)
modes = sorted({r["take_mode"] for r in rows})
dn = np.array([float(r["done_t"]) if r["done_t"] not in ("", "nan") else np.nan for r in rows])  # mode 11: release to after_done
started = int((~np.isnan(st)).sum())
print(f"{lab}: take_mode {modes}, started {started}/{len(rows)} | batter_fell {fell}/{len(rows)} (joint 0.3 rad off target {dev}), "
      f"pitcher_fell {pf} | take_bat_vmax max {vm.max():.2f} median {np.median(vm):.2f} m/s | "
      f"start +-0.05 s: {win(-0.05, 0.05)} | start..+0.5 s: {win(0.0, 0.5)} | pose: {pose()}"
      + (f" | done (pitch passed) ..+0.5 s: {win(0.0, 0.5, dn)}" if "11" in modes else ""))
