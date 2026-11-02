"""FORM diag (14, 11-02 16:4x; user: the form looks awkward, the ball flies too far): diag json (--form_log --contact_log) ->
logs/h32/form_<name>.csv (env, clip, hit, exit, launch, carry, pose_err_rad, key_err_m = swing window clip start -> contact,
+ the after-contact pair and the legs / waist / arms pose split) and one table row. hit / exit / launch / carry as e10_row /
h31_carry: in zone, fair = |spray| <= 45, carry = drag + Magnus flight from contact (release 0.9 m); form = mean / q90 over all
hit episodes. Usage: h32_form.py diag.json name [out.csv] -> one line"""
import csv
import json
import sys

import numpy as np
import torch

from aibaseball.physics import PhysicsSpec
from aibaseball.physics.ball_flight import launch_state, simulate_flight

J = json.load(open(sys.argv[1])); name = sys.argv[2]
P = {k: np.array(v, float) for k, v in J["per_pitch"].items()}
names = J["summary"]["form"]["clip_names"]
n = len(P["hit"]); hit = P["hit"] > 0
E, L = P["ev_contact"], P["launch_contact"]
carry = np.full(n, np.nan)
ok = hit & np.isfinite(E) & np.isfinite(L)
if ok.any():
    v, w = launch_state(torch.tensor(E[ok], dtype=torch.float32), torch.tensor(L[ok], dtype=torch.float32),
                        torch.tensor(np.nan_to_num(P["backspin_contact"][ok]), dtype=torch.float32))
    pos = torch.zeros(int(ok.sum()), 3); pos[:, 2] = 0.9
    carry[ok] = simulate_flight(pos, v, w, PhysicsSpec()).carry.numpy()
cols = ["pose_sw", "key_sw", "pose_post", "key_post", "legs_sw", "waist_sw", "arms_sw"]
with open(sys.argv[3] if len(sys.argv) > 3 else f"logs/h32/form_{name}.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["env", "clip", "hit", "exit", "launch", "spray", "carry", "pose_err_rad", "key_err_m", "pose_err_post_rad",
                 "key_err_post_m", "pose_legs_rad", "pose_waist_rad", "pose_arms_rad"])
    for e in range(n):
        wr.writerow([e, names[int(P["clip_first"][e])], int(hit[e])] + [f"{x:.4f}" for x in (E[e], L[e], P["spray_contact"][e], carry[e])]
                    + [f"{P['form_' + c][e]:.4f}" for c in cols])
z = (np.abs(P["plate_y"]) <= 0.2525) & (P["plate_z"] >= 0.50) & (P["plate_z"] <= 1.10)
h = z & hit; fz = h & (np.abs(P["spray_contact"]) <= 45)
mq = lambda x: f"{np.nanmean(x[hit]):.3f}/{np.nanpercentile(x[hit], 90):.3f}"
print(f"{name} | n {z.sum()} hit {h.sum() / max(z.sum(), 1):.3f} | exit {np.nanmedian(E[h]):.1f} | carry {np.nanmedian(carry[fz]):.1f} "
      f"| launch {np.nanmedian(L[h]):+.1f} | pose {mq(P['form_pose_sw'])} | key {mq(P['form_key_sw'])} "
      f"| post pose {mq(P['form_pose_post'])} key {mq(P['form_key_post'])} | legs {mq(P['form_legs_sw'])} waist {mq(P['form_waist_sw'])} arms {mq(P['form_arms_sw'])}")
