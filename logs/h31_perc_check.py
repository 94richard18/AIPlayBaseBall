"""13997 joint rows (s6 by default; args: joint csvs, pooled): perceived crossing z (filter) at t_plane - 0.10 s and at the commit minus the true crossing (pred_z),
and dz at contact / closest approach for hits and misses; the diag's dz on the same 4 clips (it records no percept z). CPU."""
import csv
import json

import numpy as np

f = lambda x: float(x) if x not in ("", "nan", "None") else np.nan  # noqa: E731
import sys
b = [x for c in (sys.argv[1:] or ["logs/h31/joint_s6_13997.csv"]) for x in csv.DictReader(open(c))]  # 13997 joint csvs (pooled)
J = {k: np.array([f(x[k]) for x in b]) for k in ("pred_z", "perc_z_pre", "perc_z_commit", "dz_m", "t_commit", "t_gap", "kmh", "flight_pred")}
hit = np.array([x["hit"] == "True" for x in b]); sw = np.array([x["swing"] == "True" for x in b])
s = lambda a: f"bias {np.nanmedian(a) * 100:+.1f} cm, mean {np.nanmean(a) * 100:+.1f}, sd {np.nanstd(a) * 100:.1f}, q25/75 {np.nanpercentile(a, 25) * 100:+.1f}/{np.nanpercentile(a, 75) * 100:+.1f} (n {np.isfinite(a).sum()})"  # noqa: E731
for nm, sel in (("all", np.ones(len(b), bool)), ("hits", hit), ("misses", ~hit)):
    print(f"joint {nm:6s} perc(t_plane-0.10) - true: {s(J['perc_z_pre'][sel] - J['pred_z'][sel])}")
    print(f"joint {nm:6s} perc(commit)       - true: {s(J['perc_z_commit'][sel] - J['pred_z'][sel])}")
    print(f"joint {nm:6s} dz (contact / closest):  {s(J['dz_m'][sel])}")
e = J["perc_z_commit"] - J["pred_z"]; ok = np.isfinite(e) & np.isfinite(J["dz_m"])
print(f"joint r(perc err at commit, dz) {np.corrcoef(e[ok], J['dz_m'][ok])[0, 1]:+.2f}; r(perc err, kmh) {np.corrcoef(e[ok], J['kmh'][ok])[0, 1]:+.2f}; "
      f"commit lead (t_gap - t_commit) median {np.nanmedian(J['t_gap'] - J['t_commit']) * 1000:.0f} ms")
D = {k: np.array(v, float) for k, v in json.load(open("logs/diag_bat_contact_h31_13997.json"))["per_pitch"].items()}
z = (np.abs(D["plate_y"]) <= 0.2525) & (D["plate_z"] >= 0.50) & (D["plate_z"] <= 1.10); h = D["hit"] > 0
for nm, sel in (("all", np.isfinite(D["dz"])), ("in-zone", z), ("hits", h), ("misses", ~h)):
    print(f"diag  {nm:7s} dz (contact / closest): {s(D['dz'][sel])}")
