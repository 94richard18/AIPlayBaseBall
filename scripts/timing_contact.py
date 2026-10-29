"""Timing vs contact station (planning 14, user 10-30 "新線": a measurement, no training). CPU only.

Inputs: diag_bat_contact --contact_log per-pitch JSONs (delivery 12500 @ 1.5x, diag1024 seed 0, P3c t18, jitter 0, aim 0;
one per swing-start bias) + the joint contact_offset rows (35 joint hits, seeds 6 / 1).
Timing signs (diag_bat_contact's definitions):
  timing   = t_square - t_plane (s): the bat is squared up (axis most across the pitch line, sweet spot > 10 m/s) minus the
             ball crossing the contact plane; < 0 = the bat squares BEFORE the ball arrives = bat EARLY.
  ref_err  = reference clock when the ball crosses the contact plane minus the clip's contact time (s); + = the swing
             reference is already past contact = bat EARLY.
  ref_phase_hit = the same at the hit step (hits only); + = early.
  timing_hit = t_square - t_hit (s, hits: the ball never crosses the plane, so t_plane / timing are NaN); < 0 = early.
Sweet-spot miss vector at the closest approach of the ball to the bat axis over the swing (until contact; for hits that is
the contact): along the axis = x_on_bat - sweet spot (+ tip), vertical = dz (ball above the axis point), horizontal along
the swing = dperp (+ = ball ahead of the bat in the swing direction: the bat is behind = late / short; - = past = early),
surface gap (< 0 = penetration = contact). Plane: swing_pitch_deg = angle between the sweet-spot velocity and the reversed
pitch direction.
dx = contact station - sweet spot (m, + toward the tip; - = handle side); D = normal offset (+ = undercut).
Misses: closest approach over the swing (x_on_bat - sweet spot, gap = surface distance, dz, dperp = along the swing).

    python scripts/timing_contact.py --diag 0.030:<json> 0.055:<json> ... --joint logs/contact_offset/contact_offset.csv
        --joint_csv logs/contact_offset/s6.csv logs/contact_offset/s1.csv --out logs/timing_contact
"""
import argparse
import csv
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--diag", nargs="+", required=True, help="bias:path pairs; the 0.055 one is the baseline for Q1 / Q2")
ap.add_argument("--joint", default="logs/contact_offset/contact_offset.csv")
ap.add_argument("--joint_csv", nargs="*", default=["logs/contact_offset/s6.csv", "logs/contact_offset/s1.csv"])
ap.add_argument("--out", default="logs/timing_contact")
args = ap.parse_args()
SS = 0.7112  # sweet spot, m from the knob (BatSpec)
L = []
q = lambda v: f"{np.nanmedian(v):+.4f} [{np.nanpercentile(v, 25):+.4f}, {np.nanpercentile(v, 75):+.4f}]" if np.isfinite(v).any() else "-"  # noqa: E731


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")


def load(path):
    P = {k: np.array(v, dtype=float) for k, v in json.load(open(path))["per_pitch"].items()}
    P["zone"] = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
    P["hit"] = P["hit"].astype(bool)
    P["timing_hit"] = P["t_square"] - P["t_hit"]
    return P


D = {float(b): load(p) for b, p in (s.split(":", 1) for s in args.diag)}
rows = []
B = D[0.055]
L.append("Q1 hits: timing vs contact geometry (delivery 12500, bias 0.055, diag1024 seed 0)")
for nm, m in (("hits", B["hit"]), ("in-zone hits", B["hit"] & B["zone"])):
    dx, Dn, tm, rp = B["dx_contact"][m], B["D_contact"][m], B["timing_hit"][m], B["ref_phase_hit"][m]
    L.append(f"  {nm} n {int(m.sum())}: corr(dx, timing) {corr(dx, tm):+.2f}, corr(D, timing) {corr(Dn, tm):+.2f}, "
             f"corr(dx, ref_phase_hit) {corr(dx, rp):+.2f}, corr(D, ref_phase_hit) {corr(Dn, rp):+.2f}, corr(|dx|, ref_phase_hit) "
             f"{corr(np.abs(dx), rp):+.2f}")
    L.append(f"    timing_hit (s, < 0 = bat squared before contact) {q(tm)}; ref_phase_hit (s, + = early) {q(rp)}; plane "
             f"swing_pitch_deg {q(B['swing_pitch_deg'][m])}")
    t1, t2 = np.nanpercentile(rp, [100 / 3, 200 / 3])
    for lab, s in (("latest third (ref_phase_hit low)", rp <= t1), ("middle third", (rp > t1) & (rp <= t2)),
                   ("earliest third (ref_phase_hit high)", rp > t2)):
        L.append(f"    {lab:36s} n {int(s.sum()):3d}: ref_phase_hit {q(rp[s])}  dx {q(dx[s])}  |dx| med {np.nanmedian(np.abs(dx[s])):.4f}"
                 f"  D {q(Dn[s])}")
# joint hits (contact_offset rows + their joint CSV timing columns)
if os.path.exists(args.joint):
    J = list(csv.DictReader(open(args.joint)))
    tab = {}
    for p in args.joint_csv:
        for r in csv.DictReader(open(p)):
            tab[(os.path.splitext(os.path.basename(p))[0], int(r["env"]))] = r
    f = lambda r, k: float(r[k]) if r.get(k, "") not in ("", "nan") else float("nan")  # noqa: E731
    dx = np.array([float(r["dx_m"]) for r in J])
    Dn = np.array([float(r["D_m"]) for r in J])
    tm = np.array([f(tab[(r["run"], int(r["env"]))], "timing_s") for r in J])
    re_ = np.array([f(tab[(r["run"], int(r["env"]))], "ref_err") for r in J])
    L.append("  joint hits: timing_s / ref_err are NaN on every joint hit row (the ball never crosses the plane) - no joint timing")
    if False:
        L.append(f"  joint hits n {len(J)}: corr(dx, timing_s) {corr(dx, tm):+.2f}, corr(D, timing_s) {corr(Dn, tm):+.2f}, "
             f"corr(dx, ref_err) {corr(dx, re_):+.2f}, corr(D, ref_err) {corr(Dn, re_):+.2f}; timing_s {q(tm)}, ref_err {q(re_)}")
L.append("Q2 misses at closest approach (bias 0.055; all in-zone misses / all misses)")
for nm, m in (("misses", ~B["hit"]), ("in-zone misses", ~B["hit"] & B["zone"])):
    L.append(f"  {nm} n {int(m.sum())}: station from the sweet spot (m, + tip) {q(B['x_on_bat'][m] - SS)}; surface gap (m) "
             f"{q(B['gap'][m])}; dz (ball above the axis point, m) {q(B['dz'][m])}; dperp (along the swing, m) {q(B['dperp'][m])}")
    L.append(f"    timing (ms, < 0 = bat squares before the ball crosses the plane) {q(B['timing'][m] * 1000)}; ref_err (ms, + = "
             f"early) {q(B['ref_err'][m] * 1000)}; bat at the tip end of the barrel or past it (station > +0.10 m) "
             f"{int(((B['x_on_bat'][m] - SS) > 0.10).sum())}, at the handle side (< -0.10 m) {int(((B['x_on_bat'][m] - SS) < -0.10).sum())}")
L.append("Q2b sweet-spot miss vector at closest approach (bias 0.055), hits and misses, in-zone")
for nm, m in (("in-zone hits", B["hit"] & B["zone"]), ("in-zone misses", ~B["hit"] & B["zone"]), ("all hits", B["hit"]),
              ("all misses", ~B["hit"])):
    ax_, vz, hz = B["x_on_bat"][m] - SS, B["dz"][m], B["dperp"][m]
    L.append(f"  {nm:15s} n {int(m.sum()):3d}: along axis {q(ax_)} (|.| med {np.nanmedian(np.abs(ax_)):.4f}); vertical {q(vz)} "
             f"(|.| {np.nanmedian(np.abs(vz)):.4f}); along swing {q(hz)} (|.| {np.nanmedian(np.abs(hz)):.4f}); gap {q(B['gap'][m])}")
L.append("Q3 swing-start bias sweep (same policy, diag1024 seed 0, contact_log; in-zone)")
for b in sorted(D):
    P = D[b]
    z = P["zone"]
    h = P["hit"] & z
    mm = ~P["hit"] & z
    Dh = P["D_contact"][h]
    L.append(f"  bias {b:.3f}: in-zone hit {h.sum() / max(z.sum(), 1):.3f} (n {int(z.sum())}); |dx| med {np.nanmedian(np.abs(P['dx_contact'][h])):.4f}"
             f"; dx med {np.nanmedian(P['dx_contact'][h]):+.4f}; D med {np.nanmedian(Dh):+.4f}, in [0, 2.5 cm] {((Dh >= 0) & (Dh <= 0.025)).mean():.3f}"
             f"; launch med {np.nanmedian(P['launch_contact'][h]):.1f}; exit med {np.nanmedian(P['ev_contact'][h]):.1f}"
             f"; misses early {int((mm & (P['timing'] < 0)).sum())} late {int((mm & (P['timing'] > 0)).sum())}"
             f"; ref_phase_hit med {np.nanmedian(P['ref_phase_hit'][h]) * 1000:+.0f} ms; timing_hit med "
             f"{np.nanmedian(P['timing_hit'][h]) * 1000:+.0f} ms; miss vector |axis| / |vertical| / |swing| med "
             f"{np.nanmedian(np.abs(P['x_on_bat'][z] - SS)):.3f} / {np.nanmedian(np.abs(P['dz'][z])):.3f} / {np.nanmedian(np.abs(P['dperp'][z])):.3f} (all in-zone)")
    for i in np.flatnonzero(P["hit"]):
        rows.append(dict(bias=b, env=int(i), in_zone=bool(P["zone"][i]), dx=P["dx_contact"][i], D=P["D_contact"][i],
                         timing_hit=P["timing_hit"][i], axis_off=P["x_on_bat"][i] - SS, dz=P["dz"][i], dperp=P["dperp"][i], ref_phase_hit=P["ref_phase_hit"][i], ref_err=P["ref_err"][i],
                         launch=P["launch_contact"][i], ev=P["ev_contact"][i]))
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "hits.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
