"""ArmAim scripted rows (planning 14, 10-30): realised share of the commanded offset from a wet / dry pair of
diag_bat_contact --aim_scripted_dump npz files (same seed: pitch k of env e is the same pitch in both), and the percept
decomposition of the ball's crossing y.
  realised(group) = sum_pitches (num_wet - num_dry) / sum den_wet,  num = sum_window (q - swing target) . offset
  percept terms (y, relative to the nominal target, m): (a) pitch_plate vs actual crossing, (b) gravity-only extrapolation
  of the TRUE state vs the actual crossing, (c) percept vs gravity-only of the true state (delay + noise), at the window's
  first and last step.
    python scripts/armaim_pair.py --wet a.npz --dry b.npz --label row4
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--wet", required=True)
ap.add_argument("--dry", default=None)
ap.add_argument("--label", default="")
args = ap.parse_args()


def load(f):
    z = np.load(f)
    return {n: z["rows"][:, i] for i, n in enumerate(z["names"])}


W = load(args.wet)
out = [f"{args.label}: {len(W['env'])} pitches"]
if args.dry:
    D = load(args.dry)
    kw = {(int(e), int(k)): i for i, (e, k) in enumerate(zip(W["env"], W["ep"]))}
    kd = {(int(e), int(k)): i for i, (e, k) in enumerate(zip(D["env"], D["ep"]))}
    both = sorted(set(kw) & set(kd))
    iw, idr = np.array([kw[k] for k in both]), np.array([kd[k] for k in both])
    s = []
    for g in [n[4:] for n in W if n.startswith("num_") and n != "num_z"] + ["z"]:
        den = W[f"den_{g}"][iw].sum()
        if den > 1e-8:
            s.append(f"{g} {(W[f'num_{g}'][iw] - D[f'num_{g}'][idr]).sum() / den:+.2f}")
    out.append(f"  realised (wet - dry, {len(both)} paired pitches): " + ", ".join(s))
a = W["plate_y"] - W["act_y"]
b0, b1 = W["grav_y_open"] - W["act_y"], W["grav_y_last"] - W["act_y"]
c0, c1 = W["perc_y_open"] - W["grav_y_open"], W["perc_y_last"] - W["grav_y_last"]
t0, t1 = W["perc_y_open"] - W["act_y"], W["perc_y_last"] - W["act_y"]
m = lambda v: f"mean {np.mean(v):+.3f} |.| {np.mean(np.abs(v)):.3f} p90 {np.percentile(np.abs(v), 90):.3f}"  # noqa: E731
out += [f"  (a) pitch_plate - actual crossing y: {m(a)}",
        f"  (b) gravity-only (true state) - actual: open {m(b0)}; last {m(b1)}",
        f"  (c) percept - gravity-only (true state): open {m(c0)}; last {m(c1)}",
        f"  total percept - actual: open {m(t0)}; last {m(t1)}",
        f"  actual crossing z - nominal z: mean {np.mean(W['act_z']):+.3f}; plate_z mean {np.mean(W['plate_z']):.3f}"]
print("\n".join(out))
