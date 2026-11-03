"""ema_gate.py <ref diag tag> <test diag tag> <ref joint csv> <test joint csv>: the AIB_ACT_EMA gate (14, 11-03 22:xx).
Row: hit / in-zone fair exit + carry medians / fell by end / da, wait_da, dqt, torso_hp, wait torso_hp (diag, jitter 0) and the joint
row (hits / fair / OF / fell before done). PASS = hit and exit within 1 SE of the reference (SE: binomial / bootstrap of the median),
fell-before-done not above the reference, wait_da cut >= 40 %."""
import csv
import json
import sys

import numpy as np


def diag(tag):
    P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"].items()
         if isinstance(v, list) and v and not isinstance(v[0], (list, dict, str))}
    h = P["hit"] > 0
    fair = h & (np.abs(P["spray_contact"]) <= 45)
    return P, h, fair


def joint(path):
    b = list(csv.DictReader(open(path)))
    T = lambda x: x == "True"
    f = lambda x: float(x) if x not in ("", "nan") else float("nan")
    hits = [x for x in b if T(x["hit"])]
    fair = [x for x in hits if abs(f(x["spray_deg"])) <= 45]
    return dict(hits=len(hits), fair=len(fair), of=sum(f(x["carry_m"]) >= 38.9 for x in fair),
                fbd=sum(T(x["batter_fell_before_done"]) for x in b), fell=sum(T(x["batter_fell"]) for x in b))


rng = np.random.default_rng(0)
(Pr, hr, fr), (Pt, ht, ft) = diag(sys.argv[1]), diag(sys.argv[2])
ev_r, ev_t = (x[np.isfinite(x)] for x in (Pr["ev_contact"][fr], Pt["ev_contact"][ft]))
se_hit = np.sqrt(hr.mean() * (1 - hr.mean()) / len(hr))
se_ev = np.std([np.median(rng.choice(ev_r, len(ev_r))) for _ in range(500)])
m = lambda P, h, k: np.nanmean(P[k][np.isfinite(P[k])]) if k.startswith("jit") else np.nanmedian(P[k][h])
row = []
for name, (P, h, fa) in (("ref", (Pr, hr, fr)), ("test", (Pt, ht, ft))):
    row.append(f"{name}: hit {h.mean():.3f} exit {np.nanmedian(P['ev_contact'][fa]):.1f} launch {np.nanmedian(P['launch_contact'][fa]):+.1f} fell by end {P['fell'].mean():.3f} "
               f"da {m(P, h, 'jit_da'):.3f} wait_da {m(P, h, 'jit_wait_da'):.3f} dqt {m(P, h, 'jit_dqt'):.3f} torso_hp {m(P, h, 'jit_torso_hp'):.2f} wait_torso_hp {m(P, h, 'jit_wait_torso_hp'):.2f}")
Jr, Jt = joint(sys.argv[3]), joint(sys.argv[4])
row.append("joint ref " + " ".join(f"{k} {v}" for k, v in Jr.items()) + " | joint test " + " ".join(f"{k} {v}" for k, v in Jt.items()))
cut = 1 - m(Pt, ht, "jit_wait_da") / m(Pr, hr, "jit_wait_da")
ok = (ht.mean() >= hr.mean() - se_hit and np.median(ev_t) >= np.median(ev_r) - se_ev and Jt["fbd"] <= Jr["fbd"] and cut >= 0.40)
print(f"{'PASS' if ok else 'FAIL'} (hit within {se_hit:.3f}, exit within {se_ev:.2f}, fell-before-done {Jt['fbd']} <= {Jr['fbd']}, wait_da cut {cut:.0%} >= 40%) | "
      + " | ".join(row))
