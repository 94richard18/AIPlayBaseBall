"""13997: joint fair hits vs diag fair in-zone hits - launch against pitch height / side at the plate, timing, dz at contact (CPU).
Joint hit rows carry the post-hit ball in plate_y / plate_z, so the pitch crossing is pred_y / pred_z (= plate_* on misses);
joint timing_s is only set on misses."""
import csv
import json

import numpy as np

f = lambda x: float(x) if x not in ("", "nan", "None") else np.nan  # noqa: E731
b = list(csv.DictReader(open("logs/h31/joint_s6_13997.csv")))
J = {k: np.array([f(x[k]) for x in b]) for k in ("pred_y", "pred_z", "plate_y", "plate_z", "timing_s", "dz_m", "dperp_m", "launch_deg", "spray_deg", "carry_m", "exit_mps", "kmh")}
J["pred_y"] = -J["pred_y"]  # batter frame -> plate frame (pred_y = -plate_y on misses)
jh = np.array([x["hit"] == "True" for x in b]); jf = jh & (np.abs(J["spray_deg"]) <= 45)
mi = ~jh & np.isfinite(J["plate_z"])
print(f"pred vs plate on joint misses: |dz| med {np.nanmedian(np.abs(J['pred_z'][mi] - J['plate_z'][mi])):.3f} m, |dy| med {np.nanmedian(np.abs(J['pred_y'][mi] - J['plate_y'][mi])):.3f} m (n {mi.sum()})")
D = {k: np.array(v, float) for k, v in json.load(open("logs/diag_bat_contact_h31_13997.json"))["per_pitch"].items()}
z = (np.abs(D["plate_y"]) <= 0.2525) & (D["plate_z"] >= 0.50) & (D["plate_z"] <= 1.10)
dh = z & (D["hit"] > 0); df = dh & (np.abs(D["spray_contact"]) <= 45)
jz = (np.abs(J["pred_y"]) <= 0.2525) & (J["pred_z"] >= 0.50) & (J["pred_z"] <= 1.10)
m = lambda a: np.nanmedian(a)  # noqa: E731
q = lambda a: "/".join(f"{x:+.3f}" for x in np.nanpercentile(a, [25, 50, 75])) if np.isfinite(a).any() else "n/a"  # noqa: E731
print(f"joint: 64 pitches, pitch in zone {jz.sum()}, hits {jh.sum()}, fair {jf.sum()} (fair with pitch in zone {(jf & jz).sum()}); diag: in zone {z.sum()}, hits {dh.sum()}, fair {df.sum()}")
print(f"{'set':22s} {'n':>4s} {'launch q25/50/75':>18s} {'pitch z q25/50/75':>22s} {'pitch y':>22s} {'timing (s)':>22s} {'dz at contact':>22s}")
lq = lambda a: "/".join(f"{x:+.0f}" for x in np.nanpercentile(a, [25, 50, 75]))  # noqa: E731
rows = (("joint fair", J["launch_deg"], J["pred_z"], J["pred_y"], J["timing_s"], J["dz_m"], jf),
        ("joint fair, zone", J["launch_deg"], J["pred_z"], J["pred_y"], J["timing_s"], J["dz_m"], jf & jz),
        ("joint fair, z<0.70", J["launch_deg"], J["pred_z"], J["pred_y"], J["timing_s"], J["dz_m"], jf & (J["pred_z"] < 0.70)),
        ("joint fair, z>=0.70", J["launch_deg"], J["pred_z"], J["pred_y"], J["timing_s"], J["dz_m"], jf & (J["pred_z"] >= 0.70)),
        ("diag fair zone", D["launch_contact"], D["plate_z"], D["plate_y"], D["timing"], D["dz"], df),
        ("diag fair, z<0.70", D["launch_contact"], D["plate_z"], D["plate_y"], D["timing"], D["dz"], df & (D["plate_z"] < 0.70)),
        ("diag fair, z>=0.70", D["launch_contact"], D["plate_z"], D["plate_y"], D["timing"], D["dz"], df & (D["plate_z"] >= 0.70)))
for nm, L, PZ, PY, TM, DZ, sel in rows:
    print(f"{nm:22s} {sel.sum():4d} {lq(L[sel]):>18s} {q(PZ[sel]):>22s} {q(PY[sel]):>22s} {q(TM[sel]):>22s} {q(DZ[sel]):>22s}")
for nm, L, X, sel in (("joint fair", J["launch_deg"], {"pitch_z": J["pred_z"], "pitch_y": J["pred_y"], "dz": J["dz_m"]}, jf),
                      ("diag fair", D["launch_contact"], {"pitch_z": D["plate_z"], "pitch_y": D["plate_y"], "dz": D["dz"]}, df)):
    ok = sel & np.all([np.isfinite(v) for v in X.values()], 0) & np.isfinite(L)
    print(nm, f"(n {ok.sum()})", " ".join(f"r(launch,{k}) {np.corrcoef(L[ok], v[ok])[0, 1]:+.2f}" for k, v in X.items()),
          "| launch vs dz %.1f deg/cm" % (np.polyfit(X["dz"][ok], L[ok], 1)[0] / 100))
print("joint misses timing (s) q:", q(J["timing_s"][mi]), "| diag all in-zone timing q:", q(D["timing"][z]))
