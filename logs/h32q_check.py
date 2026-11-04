"""h32q_check.py <start log> <s_lo> <s_hi> <check log> <c_lo> <c_hi>: H32Q falsifier on 50-it means (14, 11-04 01:4x; the
single-line H32P check killed H32Q on a noise dip). Means over iterations [lo, hi] of bat/ss_ca, bat/hit_rate, bat/end_fall,
bat/rt_wt_unit. Kill: check ss_ca < 24.3 (or argv[7]), hit < start hit - 0.08, fall > 1.5 x start and > start + 0.01. Prints 'KILL ...' / 'OK ...'."""
import re
import sys

import numpy as np

keys = {"ss": "bat/ss_ca", "hit": "bat/hit_rate", "fall": "bat/end_fall", "wt_unit": "bat/rt_wt_unit"}


def means(f, lo, hi):
    txt = open(f, encoding="utf-8", errors="replace").read()
    R = []
    for blk in re.split(r"Learning iteration ", txt)[1:]:
        it = int(blk.split("/")[0])
        if lo <= it <= hi:
            R.append([float((re.search(re.escape(n) + r":\s*([-0-9.e]+)", blk) or [None, "nan"])[1]) for n in keys.values()])
    return dict(zip(keys, np.nanmean(np.array(R), 0))) if R else dict.fromkeys(keys, float("nan"))


s, c = means(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])), means(sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
SS = float(sys.argv[7]) if len(sys.argv) > 7 else 24.3  # optional 7th arg: ss_ca floor
bad = []
if not c["ss"] >= SS: bad.append(f"ss_ca {c['ss']:.2f} < {SS}")
if not c["hit"] >= s["hit"] - 0.08: bad.append(f"hit {c['hit']:.3f} < start {s['hit']:.3f} - 0.08")
if not (c["fall"] <= 1.5 * s["fall"] or c["fall"] <= s["fall"] + 0.01): bad.append(f"fall share {c['fall']:.3f} > 1.5 x start {s['fall']:.3f}")
f = lambda r: " ".join(f"{k} {v:.3f}" for k, v in r.items())
print(("KILL " + "; ".join(bad) if bad else "OK") + f" | start {sys.argv[2]}-{sys.argv[3]}: {f(s)} | check {sys.argv[5]}-{sys.argv[6]}: {f(c)}")
