"""Hand-off pose vs the recover bank (planning 14, 10-30): per env of a joint_smoke run, the nearest-neighbour distance
(rad, L2) of the joint pose at the recover trigger to logs/recover_bank_s0.npz, on the arm joints and on all body joints,
split by on the ground (min_pelvis_z_2s < 0.35) or not, plus the |aim offset| at the trigger. CPU only.

    python scripts/trig_nn.py --trig logs/h29adopt/trig_h29s.npz --csv logs/h29adopt/joint_h29s.csv --label h29s
"""
import argparse
import csv

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trig", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--bank", default="logs/recover_bank_s0.npz")
ap.add_argument("--label", default="")
a = ap.parse_args()
B, T = np.load(a.bank), np.load(a.trig)
bn = [str(n) for n in B["joint_names"]]
tn = [str(n) for n in T["joint_names"]]
body = [bn[i] for i in B["body_joint_ids"]]
arm = [n for n in body if any(k in n for k in ("shoulder", "elbow", "wrist"))]
f = lambda x: float(x) if x not in ("", "nan") else float("nan")
rows = list(csv.DictReader(open(a.csv)))
g = np.array([f(r["min_pelvis_z_2s"]) < 0.35 for r in rows])
hit = np.array([r["hit"] == "True" for r in rows])
off = np.asarray(T["off"], dtype=float)
out = [f"trig NN {a.label}:"]
for name, js in (("arm", arm), ("body", body)):
    bq = B["jp"][:, [bn.index(n) for n in js]]
    tq = T["jp"][:, [tn.index(n) for n in js]]
    d = np.sqrt(((tq[:, None, :] - bq[None]) ** 2).sum(-1)).min(1)
    s = lambda m: f"{np.nanmedian(d[m]):.3f}" if m.any() else "nan"
    out.append(f"{name} ground {s(g)} (n {g.sum()}) other {s(~g)} miss-other {s(~g & ~hit)};")
s = lambda m: f"{np.nanmedian(off[m]):.3f} max {np.nanmax(off[m]):.3f}" if m.any() and np.isfinite(off[m]).any() else "nan"
out.append(f"|offset| at trigger ground {s(g)} other {s(~g)}")
print(" ".join(out))
