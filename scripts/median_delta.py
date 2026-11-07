"""Release-speed change judged on LAUNCHES (docs/plans/fastball_restart.md 5-2).

Unit = one eval_stage launch of noisy 128 (start_joint_noise 0.02); its statistic = the released-pitch release-speed
median. Judged quantity = mean of launch medians (not the pooled median). Delta = mean(cand) - mean(base),
SE = sqrt(s_c^2 / k_c + s_b^2 / k_b), Welch t 90% two-sided interval [lo, hi] (Welch-Satterthwaite df).
Stand share = mean over the candidate launches of (at rest and thrown) / n.
  --point falsifier : REGRESS if hi < -0.05; OK if lo >= -0.05; else UNDETERMINED
  --point segment   : PASS if delta >= 0.3 and lo > 0 and stand share >= 0.3125; FAIL if delta < 0.3 and hi < 0.3;
                      else UNDETERMINED (mutually exclusive)
A single launch on one side uses the other side's sd (pooled assumption, flagged in the line).

python scripts/median_delta.py --base L1.csv L2.csv ... --cand C1.csv C2.csv ... --point {falsifier,segment}
"""
import argparse
import csv

import numpy as np
from scipy import stats

STAND_SHARE = 0.3125  # 40/128


def launch(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    sp = [float(r["speed"]) for r in rows if float(r["released"]) > 0]
    st = sum(float(r["at_rest"]) > 0 and float(r["thrown"]) > 0 for r in rows)
    return float(np.median(sp)) if sp else float("nan"), st / max(len(rows), 1)


def welch(b, c, conf=0.90):
    kb, kc = len(b), len(c)
    sb = np.std(b, ddof=1) if kb > 1 else np.nan
    sc = np.std(c, ddof=1) if kc > 1 else np.nan
    flag = ""
    if np.isnan(sb):
        sb, flag = sc, " (base sd from cand)"
    if np.isnan(sc):
        sc, flag = sb, " (cand sd from base)"
    vb, vc = sb ** 2 / kb, sc ** 2 / kc
    se = np.sqrt(vb + vc)
    df = (vb + vc) ** 2 / ((vb ** 2 / max(kb - 1, 1)) + (vc ** 2 / max(kc - 1, 1)))
    t = stats.t.ppf(0.5 + conf / 2, df)
    d = float(np.mean(c) - np.mean(b))
    return d, d - t * se, d + t * se, se, df, flag


def verdict(point, d, lo, hi, stand):
    if point == "falsifier":
        return "REGRESS" if hi < -0.05 else ("OK" if lo >= -0.05 else "UNDETERMINED")
    if d >= 0.3 and lo > 0 and stand >= STAND_SHARE:
        return "PASS"
    if d < 0.3 and hi < 0.3:
        return "FAIL"
    return "UNDETERMINED"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--point", choices=("falsifier", "segment"), required=True)
    a = ap.parse_args()
    B = [launch(p) for p in a.base]
    C = [launch(p) for p in a.cand]
    b, c = np.array([x[0] for x in B]), np.array([x[0] for x in C])
    stand = float(np.mean([x[1] for x in C]))
    d, lo, hi, se, df, flag = welch(b, c)
    v = verdict(a.point, d, lo, hi, stand)
    print(f"median_delta {a.point}: delta {d:+.2f} [{lo:+.2f}, {hi:+.2f}] se {se:.2f} df {df:.1f}{flag} | base k={len(b)} "
          f"mean {b.mean():.2f} | cand k={len(c)} mean {c.mean():.2f} {np.round(c, 2).tolist()} stand {stand:.3f} -> {v}")
