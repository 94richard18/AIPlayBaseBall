"""Human28 ArmAim gates (planning 14, 10-30) on a diag_bat_contact --contact_log per-pitch JSON (diag1024 seed 0, P3c t18,
jitter 0, bias 0.055, aim 0). In-zone pitches; every hit counts (*_any keys). axis_off = dx_any on hits (the contact
station), x_on_bat - sweet spot (closest approach) on misses.
  +300 : slope of axis_off vs plate_y on hits flatter than -0.50 AND hits |axis_off| median <= 0.08 AND hit >= 0.36
  +1000: slope flatter than -0.30, |dx| median <= 0.06, hit >= 0.40, exit median >= 20, D share in [0, 2.5 cm] >= 0.25
Second line: the same slope on the reachable pitches only (step 0: plate_y >= y0 - 0.13, any on the + side).

    python scripts/armaim_gate.py --diag logs/diag_bat_contact_<tag>.json --stage 300 [--label model_X]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--diag", required=True)
ap.add_argument("--stage", choices=["base", "300", "1000"], default="base")
ap.add_argument("--label", default="")
ap.add_argument("--y0", type=float, default=-0.055)
ap.add_argument("--reach_minus", type=float, default=0.13)
args = ap.parse_args()
SS = 0.7112
P = {k: np.array(v, dtype=float) for k, v in json.load(open(args.diag))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
h = P["hit"].astype(bool)
ax = np.where(h & np.isfinite(P["dx_any"]), P["dx_any"], P["x_on_bat"] - SS)
reach = P["plate_y"] >= args.y0 - args.reach_minus


def slope(m):
    y, a = P["plate_y"][m], ax[m]
    k = np.isfinite(y) & np.isfinite(a)
    if k.sum() < 3:
        return float("nan"), float("nan"), 0
    return float(np.polyfit(y[k], a[k], 1)[0]), float(np.corrcoef(y[k], a[k])[0, 1]), int(k.sum())


hz = h & z
Z = hz.sum() / z.sum()
s_h, r_h, n_h = slope(hz)
s_m, r_m, _ = slope(~h & z)
s_r, r_r, n_r = slope(hz & reach)
AH = np.nanmedian(np.abs(ax[hz]))
DX = np.nanmedian(np.abs(P["dx_any"][hz]))
EV = np.nanmedian(P["ev_any"][hz])
D = P["D_any"][hz]
D = D[np.isfinite(D)]
DS = ((D >= 0) & (D <= 0.025)).mean() if len(D) else float("nan")
LA = np.nanmedian(P["launch_any"][hz])
gap = np.nanmedian(np.clip(P["gap"][z], 0, None))
med = (f"start medians (in-zone, all pitches): |axis_off| {np.nanmedian(np.abs(ax[z])):.3f}, gap (clamped >= 0) {gap:.3f}; "
       f"hits: |dx| {DX:.3f}, |D - 0.010| {np.nanmedian(np.abs(D - 0.010)):.3f}")
if args.stage == "300":
    ok = s_h > -0.50 and AH <= 0.08 and Z >= 0.36
elif args.stage == "1000":
    ok = s_h > -0.30 and DX <= 0.06 and Z >= 0.40 and EV >= 20.0 and DS >= 0.25
else:
    ok = None
tag = "BASE" if ok is None else ("PASS" if ok else "FAIL")
print(f"{tag} +{args.stage} {args.label}: in-zone hit {Z:.3f} (n {z.sum()}); |axis_off| hits {AH:.4f}; |dx| {DX:.4f}; exit {EV:.1f}; "
      f"D share {DS:.3f}; launch {LA:.1f}; slope hits {s_h:+.2f} (r {r_h:+.2f}, n {n_h}), misses {s_m:+.2f} (r {r_m:+.2f})")
print(f"  reachable pitches only (plate_y >= {args.y0 - args.reach_minus:+.3f}; {(z & reach).sum()} of {z.sum()} in-zone): hit "
      f"{(hz & reach).sum() / max((z & reach).sum(), 1):.3f}; slope hits {s_r:+.2f} (r {r_r:+.2f}, n {n_r}); |axis_off| hits "
      f"{np.nanmedian(np.abs(ax[hz & reach])):.4f}")
print("  " + med)
