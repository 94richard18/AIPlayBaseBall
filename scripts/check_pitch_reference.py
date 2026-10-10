"""python scripts/check_pitch_reference.py [motion.npz]

Reference feasibility after the release: COM over the lead foot (heel -0.08 .. toe +0.16 from the foot link along x),
COM vertical speed / acceleration (> -g means the feet stay loaded), pelvis heading, knee, back foot."""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r"D:\MyProject\AIPlayBaseBall")
from aibaseball.mocap.centroidal import mound_z, reference_centroidal  # noqa: E402

f = sys.argv[1] if len(sys.argv) > 1 else r"D:MyProjectAIPlayBaseBallssetsmotionspitch_2916-4.npz"
cen = reference_centroidal(f, lambda x: mound_z(x, 0.058, 0.254, 1 / 12, 0.54))
d = np.load(f, allow_pickle=True)
t = d["time"]
rel = float(d["contact_time"])
n = list(d["joint_names"])
com = cen["com"]
dt = t[1] - t[0]
vz = np.gradient(com[:, 2], dt)
k = int(0.05 / dt)
vz_s = np.convolve(vz, np.ones(k) / k, mode="same")
az = np.convolve(np.gradient(vz_s, dt), np.ones(k) / k, mode="same")
bad = []
print(" dt   COM-leadlink x  y   COM z   vz    az    heading  knee  back x   y    z   contact")
for dtr in np.arange(0, 1.01, 0.05):
    i = min(np.searchsorted(t, rel + dtr), len(t) - 1)
    lead = d["key_pos"][i, 2]
    back = d["key_pos"][i, 3]
    q = d["root_quat"][i]
    M = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    hd = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    cx = com[i, 0] - lead[0]
    single = cen["contact"][i, 0] and not cen["contact"][i, 1]
    if dtr >= 0.35 and single and not (-0.08 <= cx <= 0.16):
        bad.append(f"COM off the lead foot at +{dtr:.2f}")
    if az[i] < -0.7 * 9.81:
        bad.append(f"near flight at +{dtr:.2f} (az {az[i]:.1f})")
    if dtr > 0.14 and vz_s[i] > 0.5:
        bad.append(f"COM rising fast at +{dtr:.2f} ({vz_s[i]:.2f} m/s)")
    print(f"+{dtr:.2f}  {cx:+.2f} {com[i, 1] - lead[1]:+.2f}  {com[i, 2]:.2f} {vz_s[i]:+.2f} {az[i]:+5.1f}  {hd:+6.1f} "
          f"{np.degrees(d['joint_pos'][i, n.index('l_knee')]):5.1f}  {back[0] - lead[0]:+.2f} {back[1] - lead[1]:+.2f} "
          f"{back[2]:.2f}  {cen['contact'][i].astype(int)}")
print("PASS" if not bad else "FAIL: " + "; ".join(bad))
