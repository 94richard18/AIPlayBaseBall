"""jit_groups.py <tag> [label]: one row of the --jit_groups diag terms (mean over the 1024 first episodes, rad/s RMS of the 5 Hz
high-passed body joint velocity per group) for the wait (start -> release), approach (release -> contact) and all (start -> contact)
windows, + wait torso_hp (body angular velocity) for the link to the earlier rows."""
import json
import sys

import numpy as np

P = {k: np.array(v, float) for k, v in json.load(open(f"logs/diag_bat_contact_{sys.argv[1]}.json"))["per_pitch"].items()}
m = lambda k: np.nanmean(P[k][np.isfinite(P[k])])
G = ("wrists", "arms", "feet", "legs", "torso")
row = " | ".join(f"{w}: " + " ".join(f"{g} {m(f'jit_g_{w}_{g}'):.3f}" for g in G) for w in ("all", "wait", "app"))
print(f"{sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]} | {row} | wait_torso_hp {m('jit_wait_torso_hp'):.2f}")
