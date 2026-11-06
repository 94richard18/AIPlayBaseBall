"""Four-seam fastball line (user 11-06, criterion option (a)): one line per eval_stage --release_csv.

Release speed p10 / 50 / 90 (m/s and km/h), at rest and thrown, at rest and T3+, spin rpm p10 / 50 / 90 and the
backspin share: the angle between the spin axis and pure backspin (horizontal, perpendicular to the velocity, Magnus
lift up = v_hat x z_hat), median and the count within 30 deg. Gates: M1 90 / M2 105 / M3 120 km/h release median,
stand >= 48/64, T3+ and stand >= 32/64 (identical start); spin is reported, not gated. --ref prints the delta vs a
reference csv (P3c: assets/game/p3c_identical_t18_release.csv).

python scripts/fastball_summary.py RELEASE_CSV [--ref REF_CSV] [--tag TAG]
"""
import argparse
import csv

import numpy as np

KMH = 3.6
GATES = (("M1", 90.0), ("M2", 105.0), ("M3", 120.0))


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    f = lambda k: np.array([float(r[k]) for r in rows])
    rel, rest, thrown, tier = f("released") > 0, f("at_rest") > 0, f("thrown") > 0, f("tier")
    v = np.stack([f("vx"), f("vy"), f("vz")], 1)
    w = np.stack([f("wx"), f("wy"), f("wz")], 1)
    return dict(n=len(rows), rel=rel, stand=rest & thrown, t3=rest & thrown & (tier >= 3), speed=f("speed"), v=v, w=w)


def stats(d):
    r = d["rel"]
    sp = d["speed"][r]
    v, w = d["v"][r], d["w"][r]
    rpm = np.linalg.norm(w, axis=1) * 60 / (2 * np.pi)
    back = np.cross(v, np.array([0.0, 0.0, 1.0]))
    back /= np.linalg.norm(back, axis=1, keepdims=True) + 1e-9
    wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
    ang = np.degrees(np.arccos(np.clip((wn * back).sum(1), -1, 1)))
    q = lambda x: np.percentile(x, (10, 50, 90)) if len(x) else np.full(3, np.nan)
    return dict(n=d["n"], released=int(r.sum()), sp=q(sp), stand=int(d["stand"].sum()), t3=int(d["t3"].sum()),
                rpm=q(rpm), ang=float(np.median(ang)) if len(ang) else np.nan, back30=int((ang <= 30).sum()))


def line(s, tag):
    p = s["sp"]
    gate = " ".join(f"{g} {'pass' if p[1] * KMH >= k and s['stand'] >= 0.75 * s['n'] and s['t3'] >= 0.5 * s['n'] else 'no'}"
                    for g, k in GATES)
    return (f"{tag}: release p10/50/90 {p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f} m/s ({p[1] * KMH:.0f} km/h), stand+thrown "
            f"{s['stand']}/{s['n']}, T3+ stand {s['t3']}/{s['n']}, spin p10/50/90 {s['rpm'][0]:.0f}/{s['rpm'][1]:.0f}/"
            f"{s['rpm'][2]:.0f} rpm, axis vs backspin median {s['ang']:.0f} deg, within 30 deg {s['back30']}/{s['released']}; {gate}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    s = stats(load(a.csv))
    out = line(s, a.tag or a.csv)
    if a.ref:
        r = stats(load(a.ref))
        out += (f" | vs ref: release {s['sp'][1] - r['sp'][1]:+.2f} m/s, stand {s['stand'] - r['stand']:+d}, "
                f"T3+ stand {s['t3'] - r['t3']:+d}")
    print(out)
