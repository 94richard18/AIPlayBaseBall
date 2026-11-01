"""E10 (14, 11-01 20:0x: batter strength back to 1.0x, saim off): one diag json -> the in-zone row 14 asked for.
in zone = |plate_y| <= 0.2525, 0.50 <= plate_z <= 1.10; hits by `hit`, fair = |spray| <= 45; D share = D_contact in
[0, 2.5 cm] on hits; fell by end = `fell` over in-zone pitches;
jitter (median / q90 over in-zone pitches, release -> contact): da = mean |a_t - a_t-1|, dqt = mean |dq_target| per step,
torso = RMS torso angular velocity high-passed at 5 Hz (rad/s). Usage: e10_row.py diag.json -> one line."""
import json
import sys

import numpy as np

P = {k: np.array(v, float) for k, v in json.load(open(sys.argv[1]))["per_pitch"].items()}
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
h = z & (P["hit"] > 0)
f = h & (np.abs(P["spray_contact"]) <= 45)
md = lambda x: np.nanmedian(x) if np.isfinite(x).any() else float("nan")
q75 = lambda x: np.nanpercentile(x, 75) if np.isfinite(x).any() else float("nan")
L, E, D = P["launch_contact"], P["ev_contact"], P["D_contact"]
print(f"n {z.sum()} hit {h.sum() / max(z.sum(), 1):.3f} fair {f.sum() / max(h.sum(), 1):.2f} "
      f"ss_contact {md(P['ss_contact'][h]):.1f} exit {md(E[h]):.1f} / q75 {q75(E[h]):.1f} "
      f"launch {md(L[h]):.1f} / q75 {q75(L[h]):.1f} fair>=45 {np.mean(L[f][np.isfinite(L[f])] >= 45) if f.any() else float('nan'):.3f} "
      f"dx {md(P['dx_contact'][h]):+.3f} D share {np.mean((D[h] >= 0) & (D[h] <= 0.025)) if h.any() else float('nan'):.3f} "
      f"fell by end {np.mean(P['fell'][z] > 0):.3f}"
      + ("".join(f" {k[4:]} {np.nanmedian(P[k][z]):.4f} / q90 {np.nanpercentile(P[k][z], 90):.4f}"
                 for k in ("jit_da", "jit_dqt", "jit_torso_hp")) if "jit_da" in P else " jitter n/a"))
