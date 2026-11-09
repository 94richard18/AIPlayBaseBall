"""Event-aligned summary of a diag_fall_timeline trace (plan 5-25). CPU only, numpy only.

    python scripts/fall_timeline_report.py <trace.npz> [<trace2.npz> ...]

Events per env:
  - lead touchdown (TD): the first step after 0.2 s where the lead foot is planted and its link origin has earlier risen
    >= 0.10 m above its start height (the stride landing); NaN if never.
  - release (REL): the first step with the released flag.
  - fall: the first step with the fallen flag (episodes keep running, so the flag can be re-evaluated later; first only).
Samples at TD, TD + 0.1, REL - 0.1, REL, REL + 0.1 / 0.2 / 0.3 / 0.5 s, per outcome group (fell / at rest / up, not at
rest), medians over the envs with that sample. A sample taken at or after the env's fall is counted separately
("fallen by then") and left out of the medians. Quantities:
  - COM - lead foot: COM (xy) minus the lead-foot LINK ORIGIN (xy), not the support polygon; x toward the plate,
    y toward first base.
  - COM velocity (x, y, z); trunk tilt (deg from vertical) and lean (torso up-vector x / y); lead knee (deg);
    planted share lead / back (contact force > contact_force_n).
Also the first time after the release the trunk tilt reaches 20 / 30 / 45 deg. Descriptive only, no thresholds judged.
"""

import sys

import numpy as np


def first_true(m):
    """Index of the first True along axis 1, -1 if none."""
    has = m.any(1)
    return np.where(has, m.argmax(1), -1)


def report(path):
    z = np.load(path, allow_pickle=True)
    dt = float(z["dt"])
    n, T = z["rel"].shape[:2]
    t = (np.arange(T) + 1) * dt
    lead = z["lead"]
    lift = lead[:, :, 2] - lead[:, :1, 2]
    rose = np.maximum.accumulate(lift >= 0.10, axis=1)
    td_k = first_true(rose & (z["planted"][:, :, 0] > 0) & (t[None] > 0.2) & ~(lift >= 0.10))
    rel_k = first_true(z["rel"][:, :, 0] > 0)
    fall_k = first_true(z["fallen"][:, :, 0] > 0)
    fell = fall_k >= 0
    rest = z["at_rest"] & ~fell
    groups = (("fell", fell), ("at rest", rest), ("up, not at rest", ~fell & ~rest))
    print(f"== {path}: {z['checkpoint']} | {z['task']}, {n} envs, dt {dt:.4f}")
    print(f"  outcome: fell {fell.sum()}, at rest {rest.sum()}, up not at rest {(~fell & ~rest).sum()}; "
          f"lead touchdown found {int((td_k >= 0).sum())}, released {int((rel_k >= 0).sum())}")
    tdt = np.where(td_k >= 0, t[td_k], np.nan)
    relt = np.where(rel_k >= 0, t[rel_k], np.nan)
    fallt = np.where(fell, t[fall_k], np.nan)
    print(f"  time median: touchdown {np.nanmedian(tdt):.3f} s, release {np.nanmedian(relt):.3f} s, "
          f"release - touchdown {np.nanmedian(relt - tdt) * 1000:+.0f} ms")
    if fell.any():
        dfr = (fallt - relt)[fell]
        dft = (fallt - tdt)[fell]
        print(f"  [fell] fall - release median {np.nanmedian(dfr):+.2f} s (p10 {np.nanpercentile(dfr, 10):+.2f}, p90 "
              f"{np.nanpercentile(dfr, 90):+.2f}); falls before the release {int(np.nansum(dfr < 0))}, before the touchdown "
              f"{int(np.nansum(dft < 0))}, without a release {int(np.isnan(relt[fell]).sum())}")
    tilt = z["tilt"][:, :, 0]
    for thr in (20, 30, 45):
        k = first_true((tilt >= thr) & (t[None] >= np.nan_to_num(relt, nan=1e9)[:, None]))
        tt = np.where(k >= 0, t[k], np.nan) - relt
        print(f"  first trunk tilt >= {thr} deg after the release (s after the release), median: " + ", ".join(
            f"{lab} {np.nanmedian(tt[m]):+.2f} ({int(np.isfinite(tt[m]).sum())}/{m.sum()})" for lab, m in groups if m.any()))
    pre_tilt = np.array([tilt[i, :rel_k[i] + 1].max() if rel_k[i] >= 0 else np.nan for i in range(n)])
    print("  max trunk tilt up to the release, median: " + ", ".join(
        f"{lab} {np.nanmedian(pre_tilt[m]):.0f} deg" for lab, m in groups if m.any()))
    # TD -> REL window (Codex R39): steps [TD, REL), cut at the env's fall step; envs with no TD / no REL / REL <= TD
    # are left out and counted. Per env: lead planted share, highest lead-foot origin over its start, lead-foot
    # displacement (at the last step of the window), first step after TD with the lead foot unloaded.
    pl = z["planted"][:, :, 0] > 0
    win = []
    skip = {"no TD": 0, "no REL": 0, "REL <= TD": 0, "fell before REL (window cut)": 0}
    for i in range(n):
        if td_k[i] < 0:
            skip["no TD"] += 1
            continue
        if rel_k[i] < 0:
            skip["no REL"] += 1
            continue
        if rel_k[i] <= td_k[i]:
            skip["REL <= TD"] += 1
            continue
        end = rel_k[i]
        if fell[i] and fall_k[i] < end:
            skip["fell before REL (window cut)"] += 1
            end = fall_k[i]
        if end <= td_k[i]:
            continue
        s = slice(td_k[i], end)
        off = np.where(~pl[i, td_k[i] + 1:end])[0]
        win.append((i, pl[i, s].mean(), lift[i, s].max(), lead[i, end - 1, 0] - lead[i, td_k[i], 0],
                    lead[i, end - 1, 1] - lead[i, td_k[i], 1], (off[0] + 1) * dt if len(off) else np.nan))
    print("  TD -> REL window [TD, REL), cut at the fall; left out: " + ", ".join(f"{k} {v}" for k, v in skip.items()))
    if win:
        wa = np.array(win)
        for lab, m in groups:
            sel = m[wa[:, 0].astype(int)]
            if not sel.any():
                continue
            w_ = wa[sel]
            fu = w_[:, 5]
            print(f"    [{lab}, {sel.sum()} windows] lead planted share median {np.median(w_[:, 1]) * 100:.0f}% (p10 "
                  f"{np.percentile(w_[:, 1], 10) * 100:.0f}, p90 {np.percentile(w_[:, 1], 90) * 100:.0f}); lead foot highest over "
                  f"its start {np.median(w_[:, 2]) * 100:+.1f} cm; moved x {np.median(w_[:, 3]) * 100:+.1f} y "
                  f"{np.median(w_[:, 4]) * 100:+.1f} cm; first unloaded after TD in {int(np.isfinite(fu).sum())}/{sel.sum()}, "
                  f"median {np.nanmedian(fu) * 1000 if np.isfinite(fu).any() else np.nan:.0f} ms after TD")
    for off in (-0.3, -0.2, -0.1, 0.0):  # before the touchdown: COM y velocity and torso lean y, not-yet-fallen envs only
        k = np.where(td_k >= 0, td_k + int(round(off / dt)), -1)
        ok = (k >= 0) & ~(fell & (fall_k <= k))
        if ok.any():
            idx = np.where(ok)[0]
            print(f"  TD{off:+.1f} s ({ok.sum()} envs): COM v y median {np.median(z['comv'][idx, k[idx], 1]):+.2f} m/s, "
                  f"torso lean y {np.median(z['lean'][idx, k[idx], 1]):+.2f}, back foot planted "
                  f"{z['planted'][idx, k[idx], 1].mean() * 100:.0f}%")
    samples = (("TD", td_k, 0.0), ("TD+0.1", td_k, 0.1), ("REL-0.1", rel_k, -0.1), ("REL", rel_k, 0.0),
               ("REL+0.1", rel_k, 0.1), ("REL+0.2", rel_k, 0.2), ("REL+0.3", rel_k, 0.3), ("REL+0.5", rel_k, 0.5))
    for lab, m in groups:
        if not m.any():
            continue
        print(f"  [{lab}, {m.sum()}]  sample: COM-lead x/y cm | COM v x/y/z m/s | tilt deg, lean x/y | knee deg | "
              f"planted lead/back % | fallen by then")
        for sl, ek, off in samples:
            k = np.where(ek >= 0, ek + int(round(off / dt)), -1)
            ok = m & (k >= 0) & (k < T)
            if not ok.any():
                print(f"    {sl:8s} -")
                continue
            idx = np.where(ok)[0]
            kk = k[idx]
            down = fell[idx] & (fall_k[idx] <= kk)
            use = idx[~down]
            ku = kk[~down]
            if len(use) == 0:
                print(f"    {sl:8s} all {len(idx)} fallen by then")
                continue
            c = z["com"][use, ku]
            lf = z["lead"][use, ku]
            v = z["comv"][use, ku]
            pl = z["planted"][use, ku]
            print(f"    {sl:8s} {np.median((c[:, 0] - lf[:, 0])) * 100:+6.1f} / {np.median((c[:, 1] - lf[:, 1])) * 100:+6.1f} | "
                  f"{np.median(v[:, 0]):+.2f} / {np.median(v[:, 1]):+.2f} / {np.median(v[:, 2]):+.2f} | "
                  f"{np.median(tilt[use, ku]):4.0f}, {np.median(z['lean'][use, ku, 0]):+.2f} / {np.median(z['lean'][use, ku, 1]):+.2f} | "
                  f"{np.median(z['knee'][use, ku, 0]):4.0f} | {pl[:, 0].mean() * 100:3.0f} / {pl[:, 1].mean() * 100:3.0f} | "
                  f"{int(down.sum())}/{len(idx)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
