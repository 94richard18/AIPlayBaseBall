"""Four-seam fastball line (user 11-06, criterion option (a)): one line per eval_stage --release_csv.

Release speed p10 / 50 / 90 (m/s and km/h), at rest and thrown, at rest and T3+, spin rpm p10 / 50 / 90 and the
backspin share: the angle between the spin axis and pure backspin (horizontal, perpendicular to the velocity, Magnus
lift up = v_hat x z_hat), median and the count within 30 deg (old criterion, kept for comparison). Spin criterion (user
11-07, Statcast style): TILT = signed angle of omega's projection onto the plane perpendicular to the release velocity,
from pure backspin b (12:00 = 0 deg, 30 deg per hour), within +-45 deg (11:30-1:30, symmetric: no handedness), AND SPIN
EFFICIENCY eta = |omega - (omega . v_hat) v_hat| / |omega| >= 0.85; the count passing both. Gates: M1 90 / M2 105 / M3 120 km/h release median,
stand >= 48/64, T3+ and stand >= 32/64 (identical start); spin is reported, not gated. --ref prints the delta vs a
reference csv (P3c: assets/game/p3c_identical_t18_release.csv).
--gate noisy128 (docs/plans/fastball_restart.md 1-2-1, the acceptance gate on the noisy 128 table): stand >= 0.75 n
(96/128), T3+ and stand >= 0.25 n (32/128); M3 = final adoption also needs the tilt median within +-45 deg AND the
efficiency median >= 0.85 (median statistic: the user's D5 decision, 11-07). The default gate and the default line are
unchanged.

python scripts/fastball_summary.py RELEASE_CSV [--ref REF_CSV] [--tag TAG] [--gate {default,noisy128}]
"""
import argparse
import csv

import numpy as np

KMH = 3.6
GATES = (("M1", 90.0), ("M2", 105.0), ("M3", 120.0))
TILT_MAX, EFF_MIN = 45.0, 0.85  # user 11-07: tilt 11:30-1:30, spin efficiency >= 0.85


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
    vh = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    wp = w - (w * vh).sum(1, keepdims=True) * vh  # omega's projection onto the plane perpendicular to the velocity
    eff = np.linalg.norm(wp, axis=1) / (np.linalg.norm(w, axis=1) + 1e-9)
    up = np.cross(vh, back)  # in-plane, 90 deg from b (b x v_hat points the Magnus lift up)
    tilt = np.degrees(np.arctan2((wp * up).sum(1), (wp * back).sum(1)))
    ok = (np.abs(tilt) <= TILT_MAX) & (eff >= EFF_MIN)
    q = lambda x: np.percentile(x, (10, 50, 90)) if len(x) else np.full(3, np.nan)
    return dict(n=d["n"], released=int(r.sum()), sp=q(sp), stand=int(d["stand"].sum()), t3=int(d["t3"].sum()),
                rpm=q(rpm), ang=float(np.median(ang)) if len(ang) else np.nan, back30=int((ang <= 30).sum()),
                tilt=float(np.median(tilt)) if len(tilt) else np.nan, eff=float(np.median(eff)) if len(eff) else np.nan,
                spin_ok=int(ok.sum()))


def clock(deg):
    m = round((12 * 60 + deg * 2) % (12 * 60))  # 30 deg per hour = 2 min per deg, 12:00 = 0 deg
    return f"{(m // 60) or 12}:{m % 60:02d}"


def gate_marks(s, mode="default"):
    p = s["sp"]
    t3_share = 0.5 if mode == "default" else 0.25
    base = lambda k: p[1] * KMH >= k and s["stand"] >= 0.75 * s["n"] and s["t3"] >= t3_share * s["n"]
    if mode == "default":
        return " ".join(f"{g} {'pass' if base(k) else 'no'}" for g, k in GATES)
    spin = abs(s["tilt"]) <= TILT_MAX and s["eff"] >= EFF_MIN  # NaN medians compare False
    marks = []
    for g, k in GATES:
        ok = base(k) and (spin if g == "M3" else True)
        marks.append(f"{g} {'pass' if ok else 'no'}")
    return "noisy128 gate: " + " ".join(marks)


def line(s, tag, mode="default"):
    p = s["sp"]
    gate = gate_marks(s, mode)
    return (f"{tag}: release p10/50/90 {p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f} m/s ({p[1] * KMH:.0f} km/h), stand+thrown "
            f"{s['stand']}/{s['n']}, T3+ stand {s['t3']}/{s['n']}, spin p10/50/90 {s['rpm'][0]:.0f}/{s['rpm'][1]:.0f}/"
            f"{s['rpm'][2]:.0f} rpm, axis vs backspin median {s['ang']:.0f} deg, within 30 deg {s['back30']}/{s['released']}, "
            f"tilt median {clock(s['tilt'])} ({s['tilt']:+.0f} deg), efficiency median {s['eff']:.2f}, tilt+eff ok "
            f"{s['spin_ok']}/{s['n']}; {gate}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gate", choices=("default", "noisy128"), default="default")
    a = ap.parse_args()
    s = stats(load(a.csv))
    out = line(s, a.tag or a.csv, a.gate)
    if a.ref:
        r = stats(load(a.ref))
        out += (f" | vs ref: release {s['sp'][1] - r['sp'][1]:+.2f} m/s, stand {s['stand'] - r['stand']:+d}, "
                f"T3+ stand {s['t3'] - r['t3']:+d}, tilt+eff ok {s['spin_ok'] - r['spin_ok']:+d}")
    print(out)
