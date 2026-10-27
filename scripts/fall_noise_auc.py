"""Offline (CPU) read of diag_fall_release --per_env: fall time after the release, fall direction, and which start-noise
joints predict a fall (AUC per joint of the signed noise, fell vs not; 0.5 = no signal, |AUC - 0.5| > ~0.12 at 128 envs).

    python scripts/fall_noise_auc.py logs/falldiag_7600_env.npz
"""

import sys

import numpy as np

z = np.load(sys.argv[1])
fell, rest, rel_t, fell_t = z["fell"], z["rest"], z["rel_t"], z["fell_t"]
print(f"{len(fell)} envs: fell {fell.sum()}, at rest {rest.sum()}, released {(~np.isnan(rel_t)).sum()}")
dt = fell_t - rel_t
pre = fell & ~(dt > 0)
print(f"falls before / without the release {pre.sum()}; after it {(fell & (dt > 0)).sum()}, time after the release p10/50/90 "
      + "/".join(f"{v:.2f}" for v in np.nanpercentile(dt[fell & (dt > 0)], [10, 50, 90])) + " s")
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 9)):
    print(f"  {lo:.1f}-{hi:.1f} s after the release: {int((fell & (dt >= lo) & (dt < hi)).sum())}")
d = z["fall_dir"][fell]
ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))  # 0 = toward the plate, +90 = first base
for lab, m in (("plate", np.abs(ang) < 45), ("first base", (ang >= 45) & (ang < 135)), ("back", np.abs(ang) >= 135),
               ("third base", (ang <= -45) & (ang > -135))):
    print(f"  fall direction {lab}: {int(m.sum())}")


def auc(x, y):
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


q = z["q_noise"]
a = np.array([auc(q[:, j], fell) for j in range(q.shape[1])])
o = np.argsort(-np.abs(a - 0.5))
print("start-noise joints by |AUC - 0.5| (fell vs not; > 0.5: more positive noise falls more):")
for j in o[:10]:
    print(f"  {str(z['joint_names'][j]):22s} AUC {a[j]:.2f}")
