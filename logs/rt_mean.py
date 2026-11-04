"""rt_mean.py <train log> <lo> <hi> <key> [key ...]: means of bat/<key> over iterations [lo, hi] (as h32q_check.py), one line."""
import re
import sys

import numpy as np

txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
lo, hi, keys = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
R = []
for blk in re.split(r"Learning iteration ", txt)[1:]:
    if lo <= int(blk.split("/")[0]) <= hi:
        R.append([float((re.search(re.escape("bat/" + k) + r":\s*([-0-9.e]+)", blk) or [None, "nan"])[1]) for k in keys])
M = np.nanmean(np.array(R), 0) if R else [float("nan")] * len(keys)
print(f"{lo}-{hi}: " + " ".join(f"{k} {v:.3f}" for k, v in zip(keys, M)))
