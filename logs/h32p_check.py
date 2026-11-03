"""h32p_check.py <train.txt> <start_it> <check_it>: H32P falsifier at +250 (14, 11-03 23:3x). Start = the logged values at
start_it (stats EMA warmed up), check = at check_it: train ss_contact (bat/ss_ca), hit (bat/hit_rate), fall share (bat/end_fall),
pre-commit da (bat/rt_dapre_unit). Falsifier: ss_ca < 24.3, hit < start - 0.08 (train hit is stochastic-policy, seg 3 band
0.66-0.76), fall share > 1.5 x start and > start + 0.01. Prints 'KILL ...' or 'OK ...'."""
import re
import sys

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
keys = {"ss": "bat/ss_ca", "hit": "bat/hit_rate", "fall": "bat/end_fall", "dapre": "bat/rt_dapre_unit", "da": "bat/rt_da_unit"}
rows = {}
for blk in re.split(r"Learning iteration ", txt)[1:]:
    it = int(blk.split("/")[0])
    rows[it] = {k: float((re.search(re.escape(n) + r":\s*([-0-9.e]+)", blk) or [None, "nan"])[1]) for k, n in keys.items()}
pick = lambda t: rows[min(rows, key=lambda i: abs(i - t))]
s, c = pick(int(sys.argv[2])), pick(int(sys.argv[3]))
bad = []
if not c["ss"] >= 24.3: bad.append(f"ss_ca {c['ss']:.2f} < 24.3")
if not c["hit"] >= s["hit"] - 0.08: bad.append(f"hit {c['hit']:.3f} < start {s['hit']:.3f} - 0.08")
if not (c["fall"] <= 1.5 * s["fall"] or c["fall"] <= s["fall"] + 0.01): bad.append(f"fall share {c['fall']:.3f} > 1.5 x start {s['fall']:.3f}")
f = lambda r: " ".join(f"{k} {v:.3f}" for k, v in r.items())
print(("KILL " + "; ".join(bad) if bad else "OK") + f" | start ({sys.argv[2]}): {f(s)} | check ({sys.argv[3]}): {f(c)}")
