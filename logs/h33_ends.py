"""rsl_rl train log -> episode-end shares (bat/end_*), miss term (bat/rt_miss) and mean episode length at given iterations.
Usage: h33_ends.py train.txt it1 it2 ... -> one line per iteration"""
import re
import sys

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
rows = {}
for blk in re.split(r"Learning iteration ", txt)[1:]:
    it = int(blk.split("/")[0])
    g = lambda n: (re.search(re.escape(n) + r":\s*([-0-9.]+)", blk) or [None, "nan"])[1]
    rows[it] = {k: float(g("bat/end_" + k)) for k in ("hit", "fall", "passed", "lost", "timeout", "other")}
    rows[it]["miss"] = float(g("bat/rt_miss")); rows[it]["eplen"] = float(g("Mean episode length"))
for t in map(int, sys.argv[2:]):
    it = min(rows, key=lambda i: abs(i - t))
    print(f"it {it} " + " ".join(f"{k} {v:.3f}" for k, v in rows[it].items()))
