import csv, json, sys
import numpy as np
R = list(csv.DictReader(open("assets/game/p3c_identical_t18_release.csv")))
y = np.array([float(r["aero_y"]) for r in R]); z = np.array([float(r["aero_z"]) for r in R]); sp = np.array([float(r["speed"]) * 3.6 for r in R])
inz = (np.abs(y) <= 0.2525) & (z >= 0.50) & (z <= 1.10)
print(f"P3c release set n {len(R)}: speed {sp.min():.1f}-{sp.max():.1f} km/h (med {np.median(sp):.1f}), in-zone {inz.mean():.2f}, "
      f"plate_y med {np.median(y):+.2f} [{y.min():+.2f},{y.max():+.2f}], plate_z med {np.median(z):.2f} [{z.min():.2f},{z.max():.2f}]")

def stat(L, C):
    ok = np.isfinite(L)
    if ok.sum() == 0: return "n 0"
    return f"n {ok.sum():3d} launch med {np.median(L[ok]):+6.1f} grounder {np.mean(L[ok] < 0):.2f} carry med {np.nanmedian(C[ok]):5.1f}"

for tag in sys.argv[1:]:
    P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{tag}.json"))["per_pitch"].items()}
    f = (P["hit"] > 0) & (np.abs(P["spray_contact"]) <= 45)
    L = P["launch_contact"]; C = P.get("carry_contact", np.full_like(L, np.nan))
    py, pz = P["plate_y"], P["plate_z"]
    z_in = (np.abs(py) <= 0.2525) & (pz >= 0.50) & (pz <= 1.10)
    print(f"{tag}: all fair {stat(L[f], C[f])}")
    for name, m in (("in-zone", z_in), ("out-zone", ~z_in), ("high z>0.80", pz > 0.80), ("low z<=0.80", pz <= 0.80),
                    ("inside y<0", py < 0), ("away y>=0", py >= 0)):
        print(f"   {name:12s} {stat(L[f & m], C[f & m])}")
