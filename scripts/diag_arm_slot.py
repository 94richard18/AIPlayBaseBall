"""Arm-slot check (ONE OUTS 2026-10-25): athlete hand-marker heading / elevation around release, 2916_4 (slot 24 deg) vs 1562 (14-18 deg).

    python scripts/diag_arm_slot.py
"""
import os, sys
import numpy as np
ROOT = r"D:\MyProject\AIPlayBaseBall"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from aibaseball.mocap.obp import load_trial
D = os.path.join(ROOT, "data", "c3d_pitching")
files = {"2916_4": "001596_002916_73_191_012_FF_944.c3d", "1562_1": "000874_001562_73_211_003_FF_918.c3d",
         "1562_2": "000874_001562_73_211_008_FF_903.c3d", "1562_3": "000874_001562_73_211_009_FF_906.c3d"}
for tag, fn in files.items():
    tr = load_trial(os.path.join(D, fn), body_cutoff=20.0, bat_cutoff=30.0)
    h = tr.lm["r_hand"]; sh = tr.lm["r_shoulder"]; r = tr.rate; c = tr.contact
    v = np.gradient(h, 1.0 / r, axis=0)
    ft = np.median(np.r_[tr.lm.get("r_ankle", h)[:, 2]]) if "r_ankle" in tr.lm else 0.0
    print(f"{tag}: rate {r:.0f} Hz, height {tr.height:.2f} m, hand z at release {h[c,2]:.2f} (shoulder {sh[c,2]:.2f}), hand-above-shoulder {h[c,2]-sh[c,2]:+.2f} m")
    row = []
    for ms in (-30, -20, -10, 0, 10, 20, 30):
        i = c + int(round(ms / 1000 * r)); vi = v[i]
        hd = np.degrees(np.arctan2(vi[1], vi[0])); el = np.degrees(np.arctan2(vi[2], np.linalg.norm(vi[:2])))
        row.append(f"{ms:+d}ms hd {hd:+6.1f} el {el:+6.1f} {np.linalg.norm(vi):4.1f}")
    print("   " + "\n   ".join(row))
print("--- elevation as a function of heading (frames -40 .. +20 ms)")
for tag, fn in files.items():
    tr = load_trial(os.path.join(D, fn), body_cutoff=20.0, bat_cutoff=30.0)
    h = tr.lm["r_hand"]; r = tr.rate; c = tr.contact
    v = np.gradient(h, 1.0 / r, axis=0)[c - int(0.04 * r): c + int(0.02 * r)]
    hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])); el = np.degrees(np.arctan2(v[:, 2], np.linalg.norm(v[:, :2], axis=1)))
    sp = np.linalg.norm(v, axis=1)
    k = el.argmax()
    at = {x: np.interp(x, hd, el) for x in (-10, 0, 10)}
    rate = np.gradient(hd, 1000.0 / r)
    print(f"{tag}: peak el {el[k]:+.1f} at heading {hd[k]:+.1f}; el at heading -10/0/+10: "
          f"{at[-10]:+.1f} / {at[0]:+.1f} / {at[10]:+.1f}; heading sweep {np.interp(0, hd, rate):.1f} deg/ms; speed {np.interp(0, hd, sp):.1f} m/s")
