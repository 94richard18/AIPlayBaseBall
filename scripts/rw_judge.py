"""RW operation check (plan docs/plans/fastball_restart.md 5-19 / 5-20) from diag_chain dumps.

    python scripts/rw_judge.py <cand.npz> <m0.npz>     # m0 = ho2 7600 on the k = 0 task (logs/rw/zs_k0.npz)

Valid envs = diag_chain rows (released, window complete, no fall before the release) that have a stride event.
(i) tracking: median over envs with |target| > 1 deg of robot / target >= 0.5;
(ii) change vs the old motion: (median robot - m0) has the sign of the median target and |.| >= 0.5 |median target|.
Effective = (i) and (ii), both counts >= 96; any count < 96 -> undetermined; else not confirmed (no attribution).
"""

import sys

import numpy as np

c, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
tg, rb = np.asarray(c["d_tgt"], float), np.asarray(c["d_rob"], float)
ev = np.isfinite(tg) & np.isfinite(rb)
tk = ev & (np.abs(tg) > 1.0)
b_t, b_r = np.asarray(b["d_tgt"], float), np.asarray(b["d_rob"], float)
m0 = float(np.median(b_r[np.isfinite(b_r) & np.isfinite(b_t)]))
ratio = float(np.median(rb[tk] / tg[tk])) if tk.any() else float("nan")
mt, mr = (float(np.median(x[ev])) if ev.any() else float("nan") for x in (tg, rb))
i_ok = ratio >= 0.5
ii_ok = np.sign(mr - m0) == np.sign(mt) and abs(mr - m0) >= 0.5 * abs(mt)
if ev.sum() < 96 or tk.sum() < 96:
    v = "undetermined (n < 96)"
elif i_ok and ii_ok:
    v = "effective"
else:
    v = "not confirmed"
print(f"[rwop] k {float(c['kyaw']):.2f}: n event {int(ev.sum())}, n tracking {int(tk.sum())} (excluded |target| <= 1 deg "
      f"{int((ev & ~tk).sum())}) | target median {mt:+.1f}, robot median {mr:+.1f}, m0 {m0:+.1f} deg | (i) robot/target "
      f"{ratio:.2f} {'ok' if i_ok else 'no'} | (ii) change {mr - m0:+.1f} vs 0.5 x |target| {0.5 * abs(mt):.1f} "
      f"{'ok' if ii_ok else 'no'} -> {v}")
